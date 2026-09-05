#include "hip_fast_sampler.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace engine::models::fish_audio::detail {
namespace {

// Keep this standalone HIP helper independent of ggml include paths. These
// values are the stable ggml_type ids used by the host side of this build.
constexpr int32_t kGgmlTypeF32 = 0;
constexpr int32_t kGgmlTypeF16 = 1;
constexpr int32_t kGgmlTypeBF16 = 30;

void check_hip(hipError_t status, const char * label) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(label) + ": " + hipGetErrorString(status));
    }
}

size_t slow_embedding_type_size(int32_t type) {
    if (type == kGgmlTypeF32) {
        return sizeof(float);
    }
    if (type == kGgmlTypeF16 || type == kGgmlTypeBF16) {
        return sizeof(uint16_t);
    }
    throw std::invalid_argument("HIP Fish slow embedding supports only F32/F16/BF16 tables");
}

struct Workspace {
    int32_t vocab_size = 0;
    HipFastTopKResult * packed = nullptr;
    uint32_t * chain_rng_words = nullptr;
    int32_t * chain_rng_consumed = nullptr;
    int32_t * chain_codes = nullptr;
    float * embeddings = nullptr;
    int32_t embedding_rows = 0;
    int32_t embedding_dim = 0;
};

constexpr int32_t kMaxSlowCodebooks = 16;

struct SlowFrameArgs {
    int32_t values[kMaxSlowCodebooks + 1]{};
};

struct SlowStepWorkspace {
    void * semantic_text_embeddings = nullptr;
    void * codebook_embeddings = nullptr;
    int32_t semantic_text_type = -1;
    int32_t codebook_type = -1;
    int32_t semantic_begin = 0;
    int32_t semantic_rows = 0;
    int32_t codebook_rows = 0;
    int32_t codebook_vocab_size = 0;
    int32_t dim = 0;
    int32_t num_codebooks = 0;
};

__device__ __forceinline__ bool better(float av, int32_t ai, float bv, int32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

// Exact top-k for Fish Fast-AR (4096 entries for S2 Pro). Keeping only the
// compact result on the host avoids copying and sorting the full logits tensor
// for every acoustic codebook step.
__global__ void exact_topk_kernel(
    const float * logits,
    int32_t count,
    int32_t top_k,
    HipFastTopKResult * result) {
    extern __shared__ unsigned char smem_raw[];
    float * shared_logits = reinterpret_cast<float *>(smem_raw);
    float * best_values = shared_logits + count;
    int32_t * best_indices = reinterpret_cast<int32_t *>(best_values + blockDim.x);
    double * denom_parts = reinterpret_cast<double *>(best_indices + blockDim.x);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    for (int32_t i = tid; i < count; i += static_cast<int32_t>(blockDim.x)) {
        const float v = logits[i];
        shared_logits[i] = isfinite(v) ? v : -INFINITY;
    }
    __syncthreads();

    float max_logit = -INFINITY;
    for (int32_t rank = 0; rank < top_k; ++rank) {
        float local_value = -INFINITY;
        int32_t local_index = INT32_MAX;
        for (int32_t i = tid; i < count; i += static_cast<int32_t>(blockDim.x)) {
            const float v = shared_logits[i];
            if (better(v, i, local_value, local_index)) {
                local_value = v;
                local_index = i;
            }
        }
        best_values[tid] = local_value;
        best_indices[tid] = local_index;
        __syncthreads();

        for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_value = best_values[tid + stride];
                const int32_t other_index = best_indices[tid + stride];
                if (better(other_value, other_index, best_values[tid], best_indices[tid])) {
                    best_values[tid] = other_value;
                    best_indices[tid] = other_index;
                }
            }
            __syncthreads();
        }

        const float selected_value = best_values[0];
        const int32_t selected_index = best_indices[0];
        if (tid == 0) {
            if (rank == 0) {
                max_logit = selected_value;
                result->max_logit = selected_value;
                result->count = top_k;
            }
            result->logits[rank] = selected_value;
            result->indices[rank] = selected_index;
            shared_logits[selected_index] = -INFINITY;
        }
        __syncthreads();
        if (rank == 0) {
            max_logit = result->max_logit;
        }
    }

    double local_denom = 0.0;
    for (int32_t i = tid; i < count; i += static_cast<int32_t>(blockDim.x)) {
        const float v = logits[i];
        if (isfinite(v)) {
            local_denom += static_cast<double>(expf(v - max_logit));
        }
    }
    denom_parts[tid] = local_denom;
    __syncthreads();
    for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            denom_parts[tid] += denom_parts[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result->denom = denom_parts[0];
    }
}

__global__ void prepare_fast_step_kernel(
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    uint16_t * device_mask) {
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    if (tid == 0) {
        device_position[0] = position;
    }
    if (position == 0) {
        for (int32_t i = tid; i < mask_length; i += static_cast<int32_t>(blockDim.x)) {
            device_mask[i] = (i == 0) ? static_cast<uint16_t>(0x0000) : static_cast<uint16_t>(0xfc00);
        }
    } else if (tid == 0 && position < mask_length) {
        device_mask[position] = static_cast<uint16_t>(0x0000);
    }
}

__global__ void prime_fast_from_device_kernel(
    const float * source,
    int32_t dim,
    int32_t mask_length,
    int32_t * device_position,
    uint16_t * device_mask,
    float * device_input) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (blockIdx.x == 0) {
        const int32_t tid = static_cast<int32_t>(threadIdx.x);
        if (tid == 0) {
            device_position[0] = 0;
        }
        for (int32_t mask_index = tid; mask_index < mask_length; mask_index += static_cast<int32_t>(blockDim.x)) {
            device_mask[mask_index] =
                (mask_index == 0) ? static_cast<uint16_t>(0x0000) : static_cast<uint16_t>(0xfc00);
        }
    }
    if (i < dim) {
        device_input[i] = source[i];
    }
}

__global__ void gather_embedding_kernel(
    const float * table,
    int32_t dim,
    int32_t row,
    float * output) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < dim) {
        output[i] = table[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(i)];
    }
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t raw) {
    const uint32_t sign = static_cast<uint32_t>(raw & 0x8000u) << 16;
    uint32_t exponent = static_cast<uint32_t>((raw >> 10) & 0x1fu);
    uint32_t mantissa = static_cast<uint32_t>(raw & 0x03ffu);
    uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            int32_t unbiased = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x03ffu;
            out = sign |
                  (static_cast<uint32_t>(unbiased + 127) << 23) |
                  (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    union {
        uint32_t u;
        float f;
    } bits{};
    bits.u = out;
    return bits.f;
}

__device__ __forceinline__ float load_slow_embedding_value(
    const void * table,
    int32_t type,
    size_t index) {
    if (type == kGgmlTypeF32) {
        return static_cast<const float *>(table)[index];
    }
    if (type == kGgmlTypeF16) {
        return fp16_bits_to_float(static_cast<const uint16_t *>(table)[index]);
    }
    if (type == kGgmlTypeBF16) {
        const uint16_t raw = static_cast<const uint16_t *>(table)[index];
        union {
            uint32_t u;
            float f;
        } bits{};
        bits.u = static_cast<uint32_t>(raw) << 16;
        return bits.f;
    }
    return 0.0F;
}

__global__ void build_slow_embedding_kernel(
    const void * semantic_text_embeddings,
    int32_t semantic_text_type,
    int32_t semantic_begin,
    int32_t semantic_rows,
    const void * codebook_embeddings,
    int32_t codebook_type,
    int32_t codebook_vocab_size,
    int32_t dim,
    int32_t num_codebooks,
    SlowFrameArgs frame,
    float semantic_scale,
    int32_t position,
    int32_t cache_slot,
    int32_t mask_length,
    int32_t * device_position,
    int32_t * device_cache_slot,
    uint16_t * device_mask,
    float * output) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i == 0) {
        *device_position = position;
        *device_cache_slot = cache_slot;
        if (cache_slot >= 0 && cache_slot < mask_length) {
            device_mask[cache_slot] = static_cast<uint16_t>(0x0000);
        }
    }
    if (i >= dim) {
        return;
    }
    const int32_t text_row = frame.values[0] - semantic_begin;
    if (text_row < 0 || text_row >= semantic_rows) {
        return;
    }
    float value = load_slow_embedding_value(
        semantic_text_embeddings,
        semantic_text_type,
        static_cast<size_t>(text_row) * static_cast<size_t>(dim) + static_cast<size_t>(i));
    // Preserve the CPU reference path's codebook accumulation order exactly.
    for (int32_t codebook = 0; codebook < num_codebooks; ++codebook) {
        const int32_t code = frame.values[codebook + 1];
        const size_t row = static_cast<size_t>(codebook) * static_cast<size_t>(codebook_vocab_size) +
                           static_cast<size_t>(code);
        value += load_slow_embedding_value(
            codebook_embeddings,
            codebook_type,
            row * static_cast<size_t>(dim) + static_cast<size_t>(i));
    }
    output[i] = value * semantic_scale;
}

__global__ void chain_prepare_kernel(
    const float * embedding_table,
    int32_t embedding_dim,
    int32_t code,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    uint16_t * device_mask,
    float * device_input) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        *device_position = position;
        if (position == 0) {
            for (int i = 0; i < mask_length; ++i) device_mask[i] = 0xfc00u;
        }
        if (position >= 0 && position < mask_length) device_mask[position] = 0u;
    }
    for (int i = tid; i < embedding_dim; i += static_cast<int>(blockDim.x)) {
        device_input[i] = embedding_table[static_cast<size_t>(code) * embedding_dim + i];
    }
}

__global__ void exact_topk_select_prepare_kernel(
    const float * logits,
    int32_t count,
    int32_t top_k,
    float temperature,
    float top_p,
    const uint32_t * rng_words,
    int32_t * rng_consumed,
    int32_t * selected_codes,
    int32_t output_slot,
    const float * embedding_table,
    int32_t embedding_dim,
    int32_t next_position,
    int32_t mask_length,
    int32_t * device_position,
    uint16_t * device_mask,
    float * device_input,
    HipFastTopKResult * result) {
    constexpr int kLocalItems = 16;
    extern __shared__ unsigned char smem_raw[];
    float * best_values = reinterpret_cast<float *>(smem_raw);
    int32_t * best_indices = reinterpret_cast<int32_t *>(best_values + blockDim.x);
    double * denom_parts = reinterpret_cast<double *>(best_indices + blockDim.x);
    float * sample_weights = reinterpret_cast<float *>(denom_parts + blockDim.x);

    const int tid = static_cast<int>(threadIdx.x);
    float local_values[kLocalItems];
    int32_t local_indices[kLocalItems];
    #pragma unroll
    for (int j = 0; j < kLocalItems; ++j) {
        local_values[j] = -INFINITY;
        local_indices[j] = INT32_MAX;
    }

    int local_count = 0;
    for (int i = tid; i < count; i += static_cast<int>(blockDim.x)) {
        const float raw = logits[i];
        const float v = isfinite(raw) ? raw : -INFINITY;
        int pos = local_count;
        if (pos >= kLocalItems) pos = kLocalItems - 1;
        if (local_count < kLocalItems) ++local_count;
        while (pos > 0 && better(v, i, local_values[pos - 1], local_indices[pos - 1])) {
            if (pos < kLocalItems) {
                local_values[pos] = local_values[pos - 1];
                local_indices[pos] = local_indices[pos - 1];
            }
            --pos;
        }
        if (pos < kLocalItems) {
            local_values[pos] = v;
            local_indices[pos] = i;
        }
    }

    int local_head = 0;
    float max_logit = -INFINITY;
    for (int rank = 0; rank < top_k; ++rank) {
        const bool have_local = local_head < local_count;
        best_values[tid] = have_local ? local_values[local_head] : -INFINITY;
        best_indices[tid] = have_local ? local_indices[local_head] : INT32_MAX;
        __syncthreads();
        for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float ov = best_values[tid + stride];
                const int oi = best_indices[tid + stride];
                if (better(ov, oi, best_values[tid], best_indices[tid])) {
                    best_values[tid] = ov;
                    best_indices[tid] = oi;
                }
            }
            __syncthreads();
        }
        const float selected_value = best_values[0];
        const int selected_index = best_indices[0];
        if (tid == 0) {
            if (rank == 0) {
                max_logit = selected_value;
                result->max_logit = selected_value;
                result->count = top_k;
            }
            result->logits[rank] = selected_value;
            result->indices[rank] = selected_index;
        }
        __syncthreads();
        if (rank == 0) max_logit = result->max_logit;
        if (have_local && local_indices[local_head] == selected_index) ++local_head;
        __syncthreads();
    }

    double local_denom = 0.0;
    for (int i = tid; i < count; i += static_cast<int>(blockDim.x)) {
        const float v = logits[i];
        if (isfinite(v)) local_denom += static_cast<double>(expf(v - max_logit));
    }
    denom_parts[tid] = local_denom;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) denom_parts[tid] += denom_parts[tid + stride];
        __syncthreads();
    }
    if (tid == 0) result->denom = denom_parts[0];
    __syncthreads();

    if (tid == 0) {
        double cumulative = 0.0;
        int kept = 0;
        for (int i = 0; i < top_k; ++i) {
            const float prob = static_cast<float>(static_cast<double>(expf(result->logits[i] - result->max_logit)) / result->denom);
            cumulative += static_cast<double>(prob);
            if (cumulative > static_cast<double>(top_p) && i != 0) break;
            ++kept;
        }
        int chosen = 0;
        if (kept >= 2) {
            const float tscale = temperature > 1.0e-5f ? temperature : 1.0e-5f;
            float fmax = -INFINITY;
            for (int i = 0; i < kept; ++i) fmax = fmax > result->logits[i] / tscale ? fmax : result->logits[i] / tscale;
            double fden = 0.0;
            for (int i = 0; i < kept; ++i) {
                const float q = expf(result->logits[i] / tscale - fmax);
                sample_weights[i] = q;
                fden += static_cast<double>(q);
            }
            double wsum = 0.0;
            for (int i = 0; i < kept; ++i) {
                const float probability = static_cast<float>(static_cast<double>(sample_weights[i]) / fden);
                sample_weights[i] = probability > 0.0f ? probability : 0.0f;
                wsum += static_cast<double>(sample_weights[i]);
            }
            const int off = *rng_consumed;
            const double r = 4294967296.0;
            double psum = static_cast<double>(rng_words[off]);
            psum += static_cast<double>(rng_words[off + 1]) * r;
            const double p = psum / (r * r);
            *rng_consumed = off + 2;
            double cp = 0.0;
            chosen = kept - 1;
            for (int i = 0; i < kept; ++i) {
                cp += static_cast<double>(sample_weights[i]) / wsum;
                if (i == kept - 1) cp = 1.0;
                if (cp >= p) { chosen = i; break; }
            }
        }
        const int code = result->indices[chosen];
        selected_codes[output_slot] = code;
        *device_position = next_position;
        if (next_position >= 0 && next_position < mask_length) device_mask[next_position] = 0u;
        best_indices[0] = code;
    }
    __syncthreads();
    const int selected = best_indices[0];
    for (int i = tid; i < embedding_dim; i += static_cast<int>(blockDim.x)) {
        device_input[i] = embedding_table[static_cast<size_t>(selected) * embedding_dim + i];
    }
}

void free_workspace(Workspace * ws) {
    if (ws == nullptr) {
        return;
    }
    if (ws->packed != nullptr) {
        (void) hipFree(ws->packed);
    }
    if (ws->embeddings != nullptr) {
        (void) hipFree(ws->embeddings);
    }
    if (ws->chain_rng_words != nullptr) {
        (void) hipFree(ws->chain_rng_words);
    }
    if (ws->chain_rng_consumed != nullptr) {
        (void) hipFree(ws->chain_rng_consumed);
    }
    if (ws->chain_codes != nullptr) {
        (void) hipFree(ws->chain_codes);
    }
    delete ws;
}

void free_slow_step_workspace(SlowStepWorkspace * ws) {
    if (ws == nullptr) {
        return;
    }
    if (ws->semantic_text_embeddings != nullptr) {
        (void) hipFree(ws->semantic_text_embeddings);
    }
    if (ws->codebook_embeddings != nullptr) {
        (void) hipFree(ws->codebook_embeddings);
    }
    delete ws;
}

}  // namespace

void * hip_fast_sampler_create(int32_t vocab_size) {
    if (vocab_size <= 0) {
        throw std::invalid_argument("HIP Fish fast sampler requires positive vocab_size");
    }
    auto * ws = new Workspace{};
    ws->vocab_size = vocab_size;
    try {
        check_hip(hipMalloc(&ws->packed, sizeof(HipFastTopKResult)), "hipMalloc Fish sampler result");
        check_hip(hipMalloc(&ws->chain_rng_words, 32 * sizeof(uint32_t)), "hipMalloc Fish chain rng");
        check_hip(hipMalloc(&ws->chain_rng_consumed, sizeof(int32_t)), "hipMalloc Fish chain consumed");
        check_hip(hipMalloc(&ws->chain_codes, 16 * sizeof(int32_t)), "hipMalloc Fish chain codes");
    } catch (...) {
        free_workspace(ws);
        throw;
    }
    return ws;
}

void hip_fast_sampler_destroy(void * workspace) {
    free_workspace(static_cast<Workspace *>(workspace));
}

void hip_fast_sampler_topk(
    void * workspace,
    const float * device_logits,
    int32_t top_k,
    HipFastTopKResult * host_result) {
    auto * ws = static_cast<Workspace *>(workspace);
    if (ws == nullptr || device_logits == nullptr || host_result == nullptr) {
        throw std::invalid_argument("HIP Fish fast sampler received a null argument");
    }
    if (top_k <= 0 || top_k > ws->vocab_size || top_k > kHipFastSamplerMaxTopK) {
        throw std::invalid_argument("HIP Fish fast sampler top_k is out of range");
    }

    constexpr int threads = 256;
    const size_t shared_bytes =
        static_cast<size_t>(ws->vocab_size) * sizeof(float) +
        static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t) + sizeof(double));
    hipLaunchKernelGGL(
        exact_topk_kernel,
        dim3(1),
        dim3(threads),
        shared_bytes,
        0,
        device_logits,
        ws->vocab_size,
        top_k,
        ws->packed);
    check_hip(hipGetLastError(), "exact_topk_kernel Fish sampler");
    check_hip(
        hipMemcpy(host_result, ws->packed, sizeof(HipFastTopKResult), hipMemcpyDeviceToHost),
        "hipMemcpy Fish sampler result");
}

void hip_fast_sampler_upload_embeddings(
    void * workspace,
    const float * host_embeddings,
    int32_t rows,
    int32_t dim) {
    auto * ws = static_cast<Workspace *>(workspace);
    if (ws == nullptr || host_embeddings == nullptr || rows <= 0 || dim <= 0) {
        throw std::invalid_argument("HIP Fish embedding upload received invalid arguments");
    }
    if (ws->embeddings != nullptr) {
        check_hip(hipFree(ws->embeddings), "hipFree old Fish embedding table");
        ws->embeddings = nullptr;
    }
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(dim) * sizeof(float);
    check_hip(hipMalloc(&ws->embeddings, bytes), "hipMalloc Fish embedding table");
    check_hip(hipMemcpy(ws->embeddings, host_embeddings, bytes, hipMemcpyHostToDevice), "hipMemcpy Fish embedding table");
    ws->embedding_rows = rows;
    ws->embedding_dim = dim;
}

void hip_fast_sampler_gather_embedding(
    void * workspace,
    int32_t row,
    float * device_output,
    void * stream_ptr) {
    auto * ws = static_cast<Workspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (ws == nullptr || ws->embeddings == nullptr || device_output == nullptr ||
        row < 0 || row >= ws->embedding_rows || ws->embedding_dim <= 0) {
        throw std::invalid_argument("HIP Fish embedding gather received invalid arguments");
    }
    constexpr int threads = 256;
    const int blocks = (ws->embedding_dim + threads - 1) / threads;
    hipLaunchKernelGGL(
        gather_embedding_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        ws->embeddings, ws->embedding_dim, row, device_output);
    check_hip(hipGetLastError(), "gather_embedding_kernel Fish sampler");
}

void * hip_slow_step_create() {
    return new SlowStepWorkspace{};
}

void hip_slow_step_destroy(void * workspace) {
    free_slow_step_workspace(static_cast<SlowStepWorkspace *>(workspace));
}

void hip_slow_step_upload_embeddings(
    void * workspace,
    const void * host_semantic_text_embeddings,
    size_t semantic_text_bytes,
    int32_t semantic_text_type,
    int32_t semantic_begin,
    int32_t semantic_rows,
    const void * host_codebook_embeddings,
    size_t codebook_bytes,
    int32_t codebook_type,
    int32_t codebook_rows,
    int32_t codebook_vocab_size,
    int32_t dim,
    int32_t num_codebooks) {
    auto * ws = static_cast<SlowStepWorkspace *>(workspace);
    if (ws == nullptr || host_semantic_text_embeddings == nullptr || host_codebook_embeddings == nullptr ||
        semantic_rows <= 0 || codebook_rows <= 0 || codebook_vocab_size <= 0 || dim <= 0 ||
        num_codebooks <= 0 || num_codebooks > kMaxSlowCodebooks ||
        codebook_rows != codebook_vocab_size * num_codebooks) {
        throw std::invalid_argument("HIP Fish slow embedding upload received invalid arguments");
    }
    const size_t text_value_bytes = slow_embedding_type_size(semantic_text_type);
    const size_t codebook_value_bytes = slow_embedding_type_size(codebook_type);
    const size_t expected_text_bytes =
        static_cast<size_t>(semantic_rows) * static_cast<size_t>(dim) * text_value_bytes;
    const size_t expected_codebook_bytes =
        static_cast<size_t>(codebook_rows) * static_cast<size_t>(dim) * codebook_value_bytes;
    if (semantic_text_bytes != expected_text_bytes || codebook_bytes != expected_codebook_bytes) {
        throw std::invalid_argument("HIP Fish slow embedding byte size mismatch");
    }
    if (ws->semantic_text_embeddings != nullptr) {
        check_hip(hipFree(ws->semantic_text_embeddings), "hipFree old Fish slow text embedding table");
        ws->semantic_text_embeddings = nullptr;
    }
    if (ws->codebook_embeddings != nullptr) {
        check_hip(hipFree(ws->codebook_embeddings), "hipFree old Fish slow codebook embedding table");
        ws->codebook_embeddings = nullptr;
    }
    try {
        check_hip(
            hipMalloc(&ws->semantic_text_embeddings, semantic_text_bytes),
            "hipMalloc Fish slow text embedding table");
        check_hip(
            hipMalloc(&ws->codebook_embeddings, codebook_bytes),
            "hipMalloc Fish slow codebook embedding table");
        check_hip(
            hipMemcpy(
                ws->semantic_text_embeddings,
                host_semantic_text_embeddings,
                semantic_text_bytes,
                hipMemcpyHostToDevice),
            "hipMemcpy Fish slow text embedding table");
        check_hip(
            hipMemcpy(ws->codebook_embeddings, host_codebook_embeddings, codebook_bytes, hipMemcpyHostToDevice),
            "hipMemcpy Fish slow codebook embedding table");
    } catch (...) {
        if (ws->semantic_text_embeddings != nullptr) {
            (void) hipFree(ws->semantic_text_embeddings);
            ws->semantic_text_embeddings = nullptr;
        }
        if (ws->codebook_embeddings != nullptr) {
            (void) hipFree(ws->codebook_embeddings);
            ws->codebook_embeddings = nullptr;
        }
        throw;
    }
    ws->semantic_text_type = semantic_text_type;
    ws->codebook_type = codebook_type;
    ws->semantic_begin = semantic_begin;
    ws->semantic_rows = semantic_rows;
    ws->codebook_rows = codebook_rows;
    ws->codebook_vocab_size = codebook_vocab_size;
    ws->dim = dim;
    ws->num_codebooks = num_codebooks;
}

void hip_slow_step_build_embedding(
    void * workspace,
    const int32_t * host_frame,
    int32_t frame_size,
    float semantic_scale,
    int32_t position,
    int32_t cache_slot,
    int32_t mask_length,
    int32_t * device_position,
    int32_t * device_cache_slot,
    void * device_mask,
    float * device_output,
    void * stream_ptr) {
    auto * ws = static_cast<SlowStepWorkspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (ws == nullptr || ws->semantic_text_embeddings == nullptr || ws->codebook_embeddings == nullptr ||
        host_frame == nullptr || device_position == nullptr || device_cache_slot == nullptr ||
        device_mask == nullptr || device_output == nullptr || ws->dim <= 0 || ws->num_codebooks <= 0 ||
        frame_size != ws->num_codebooks + 1 || position < 0 || cache_slot < 0 ||
        mask_length <= 0 || cache_slot >= mask_length) {
        throw std::invalid_argument("HIP Fish slow embedding build received invalid arguments");
    }
    const int32_t token = host_frame[0];
    if (token < ws->semantic_begin || token >= ws->semantic_begin + ws->semantic_rows) {
        throw std::invalid_argument("HIP Fish slow embedding token is outside the semantic range");
    }
    SlowFrameArgs frame{};
    frame.values[0] = token;
    for (int32_t codebook = 0; codebook < ws->num_codebooks; ++codebook) {
        const int32_t code = host_frame[codebook + 1];
        if (code < 0 || code >= ws->codebook_vocab_size) {
            throw std::invalid_argument("HIP Fish slow embedding code is out of range");
        }
        frame.values[codebook + 1] = code;
    }
    constexpr int threads = 256;
    const int blocks = (ws->dim + threads - 1) / threads;
    hipLaunchKernelGGL(
        build_slow_embedding_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        ws->semantic_text_embeddings,
        ws->semantic_text_type,
        ws->semantic_begin,
        ws->semantic_rows,
        ws->codebook_embeddings,
        ws->codebook_type,
        ws->codebook_vocab_size,
        ws->dim,
        ws->num_codebooks,
        frame,
        semantic_scale,
        position,
        cache_slot,
        mask_length,
        device_position,
        device_cache_slot,
        static_cast<uint16_t *>(device_mask),
        device_output);
    check_hip(hipGetLastError(), "build_slow_embedding_kernel Fish sampler");
}

void hip_fast_sampler_prepare_step(
    void * workspace,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    void * stream_ptr) {
    auto * ws = static_cast<Workspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (ws == nullptr || device_position == nullptr || device_mask == nullptr ||
        position < 0 || mask_length <= 0 || position >= mask_length) {
        throw std::invalid_argument("HIP Fish fast step preparation received invalid arguments");
    }
    constexpr int threads = 32;
    hipLaunchKernelGGL(
        prepare_fast_step_kernel,
        dim3(1), dim3(threads), 0, stream,
        position, mask_length, device_position, static_cast<uint16_t *>(device_mask));
    check_hip(hipGetLastError(), "prepare_fast_step_kernel Fish sampler");
}

void hip_fast_sampler_prime_device(
    void * workspace,
    const float * device_source,
    int32_t dim,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    float * device_input,
    void * stream_ptr) {
    auto * ws = static_cast<Workspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (ws == nullptr || device_source == nullptr || device_position == nullptr ||
        device_mask == nullptr || device_input == nullptr || dim <= 0 || mask_length <= 0) {
        throw std::invalid_argument("HIP Fish fast device prime received invalid arguments");
    }
    constexpr int threads = 256;
    const int blocks = (dim + threads - 1) / threads;
    hipLaunchKernelGGL(
        prime_fast_from_device_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        device_source, dim, mask_length, device_position,
        static_cast<uint16_t *>(device_mask), device_input);
    check_hip(hipGetLastError(), "prime_fast_from_device_kernel Fish sampler");
}

void hip_fast_sampler_chain_begin(
    void * workspace,
    const uint32_t * host_rng_words,
    int32_t rng_word_count,
    int32_t initial_code,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    float * device_input,
    void * stream_ptr) {
    auto * ws = static_cast<Workspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (!ws || !host_rng_words || rng_word_count <= 0 || rng_word_count > 32 || !ws->embeddings) throw std::invalid_argument("invalid Fish chain begin");
    check_hip(hipMemcpyAsync(ws->chain_rng_words, host_rng_words, static_cast<size_t>(rng_word_count)*sizeof(uint32_t), hipMemcpyHostToDevice, stream), "Fish chain rng upload");
    check_hip(hipMemsetAsync(ws->chain_rng_consumed, 0, sizeof(int32_t), stream), "Fish chain consumed reset");
    hipLaunchKernelGGL(chain_prepare_kernel, dim3(1), dim3(256), 0, stream,
        ws->embeddings, ws->embedding_dim, initial_code, position, mask_length,
        device_position, static_cast<uint16_t *>(device_mask), device_input);
    check_hip(hipGetLastError(), "Fish chain prepare kernel");
}

void hip_fast_sampler_chain_select_prepare(
    void * workspace,
    const float * device_logits,
    int32_t top_k,
    float temperature,
    float top_p,
    int32_t output_slot,
    int32_t next_position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    float * device_input,
    void * stream_ptr) {
    auto * ws = static_cast<Workspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (!ws || !device_logits || !ws->embeddings || output_slot < 0 || output_slot >= 16) throw std::invalid_argument("invalid Fish chain step");
    constexpr int threads = 256;
    if (ws->vocab_size > threads * 16 || top_k <= 0 || top_k > kHipFastSamplerMaxTopK) {
        throw std::invalid_argument("Fish chain local top-k shape is unsupported");
    }
    const size_t shared_bytes =
        static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t) + sizeof(double)) +
        static_cast<size_t>(kHipFastSamplerMaxTopK) * sizeof(float);
    hipLaunchKernelGGL(exact_topk_select_prepare_kernel, dim3(1), dim3(threads), shared_bytes, stream,
        device_logits, ws->vocab_size, top_k, temperature, top_p,
        ws->chain_rng_words, ws->chain_rng_consumed, ws->chain_codes, output_slot,
        ws->embeddings, ws->embedding_dim, next_position, mask_length,
        device_position, static_cast<uint16_t *>(device_mask), device_input, ws->packed);
    check_hip(hipGetLastError(), "Fish chain select kernel");
}

void hip_fast_sampler_chain_finish(
    void * workspace,
    int32_t count,
    int32_t * host_codes,
    int32_t * host_rng_words_consumed,
    void * stream_ptr) {
    auto * ws = static_cast<Workspace *>(workspace);
    auto stream = static_cast<hipStream_t>(stream_ptr);
    if (!ws || !host_codes || !host_rng_words_consumed || count <= 0 || count > 16) throw std::invalid_argument("invalid Fish chain finish");
    check_hip(hipMemcpyAsync(host_codes, ws->chain_codes, static_cast<size_t>(count)*sizeof(int32_t), hipMemcpyDeviceToHost, stream), "Fish chain codes read");
    check_hip(hipMemcpyAsync(host_rng_words_consumed, ws->chain_rng_consumed, sizeof(int32_t), hipMemcpyDeviceToHost, stream), "Fish chain consumed read");
    check_hip(hipStreamSynchronize(stream), "Fish chain stream sync");
}

}  // namespace engine::models::fish_audio::detail
