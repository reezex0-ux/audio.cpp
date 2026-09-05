#include "hip_fast_sampler.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace engine::models::fish_audio::detail {
namespace {

void check_hip(hipError_t status, const char * label) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(label) + ": " + hipGetErrorString(status));
    }
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

__device__ __forceinline__ bool better(float av, int32_t ai, float bv, int32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

// Exact top-k for Fish Fast-AR (vocab=4096). One block avoids a full radix sort.
// Shared-memory logits are marked -inf after each selected item, preserving a
// deterministic descending order with lower token id as the tie-breaker.
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

__global__ void exact_semantic_topk_kernel(
    const float * logits,
    int32_t semantic_begin,
    int32_t semantic_end,
    int32_t eos_index,
    int32_t candidate_count,
    int32_t top_k,
    HipFastTopKResult * result) {
    extern __shared__ unsigned char smem_raw[];
    float * shared_logits = reinterpret_cast<float *>(smem_raw);
    int32_t * shared_indices = reinterpret_cast<int32_t *>(shared_logits + candidate_count);
    float * best_values = reinterpret_cast<float *>(shared_indices + candidate_count);
    int32_t * best_indices = reinterpret_cast<int32_t *>(best_values + blockDim.x);
    double * denom_parts = reinterpret_cast<double *>(best_indices + blockDim.x);

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const bool eos_outside = eos_index < semantic_begin || eos_index > semantic_end;
    for (int32_t i = tid; i < candidate_count; i += static_cast<int32_t>(blockDim.x)) {
        int32_t source_index;
        if (eos_outside && i == 0) {
            source_index = eos_index;
        } else {
            const int32_t semantic_offset = eos_outside ? i - 1 : i;
            source_index = semantic_begin + semantic_offset;
        }
        const float v = logits[source_index];
        shared_logits[i] = isfinite(v) ? v : -INFINITY;
        shared_indices[i] = source_index;
    }
    __syncthreads();

    float max_logit = -INFINITY;
    for (int32_t rank = 0; rank < top_k; ++rank) {
        float local_value = -INFINITY;
        int32_t local_source_index = INT32_MAX;
        int32_t local_candidate_index = -1;
        for (int32_t i = tid; i < candidate_count; i += static_cast<int32_t>(blockDim.x)) {
            const float v = shared_logits[i];
            const int32_t source_index = shared_indices[i];
            if (better(v, source_index, local_value, local_source_index)) {
                local_value = v;
                local_source_index = source_index;
                local_candidate_index = i;
            }
        }
        best_values[tid] = local_value;
        best_indices[tid] = local_candidate_index;
        __syncthreads();

        for (int32_t stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const int32_t lhs_ci = best_indices[tid];
                const int32_t rhs_ci = best_indices[tid + stride];
                const float lhs_v = best_values[tid];
                const float rhs_v = best_values[tid + stride];
                const int32_t lhs_si = lhs_ci >= 0 ? shared_indices[lhs_ci] : INT32_MAX;
                const int32_t rhs_si = rhs_ci >= 0 ? shared_indices[rhs_ci] : INT32_MAX;
                if (better(rhs_v, rhs_si, lhs_v, lhs_si)) {
                    best_values[tid] = rhs_v;
                    best_indices[tid] = rhs_ci;
                }
            }
            __syncthreads();
        }

        const int32_t selected_candidate = best_indices[0];
        const float selected_value = best_values[0];
        if (tid == 0) {
            const int32_t selected_source = shared_indices[selected_candidate];
            if (rank == 0) {
                max_logit = selected_value;
                result->max_logit = selected_value;
                result->count = top_k;
            }
            result->logits[rank] = selected_value;
            result->indices[rank] = selected_source;
            shared_logits[selected_candidate] = -INFINITY;
        }
        __syncthreads();
        if (rank == 0) {
            max_logit = result->max_logit;
        }
    }

    double local_denom = 0.0;
    for (int32_t i = tid; i < candidate_count; i += static_cast<int32_t>(blockDim.x)) {
        const int32_t source_index = shared_indices[i];
        const float v = logits[source_index];
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
    if (ws == nullptr) return;
    if (ws->packed != nullptr) {
        (void) hipFree(ws->packed);
    }
    if (ws->embeddings != nullptr) (void) hipFree(ws->embeddings);
    if (ws->chain_rng_words != nullptr) (void) hipFree(ws->chain_rng_words);
    if (ws->chain_rng_consumed != nullptr) (void) hipFree(ws->chain_rng_consumed);
    if (ws->chain_codes != nullptr) (void) hipFree(ws->chain_codes);
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
        check_hip(hipMalloc(&ws->packed, sizeof(HipFastTopKResult)), "hipMalloc fish sampler result");
        check_hip(hipMalloc(&ws->chain_rng_words, 32 * sizeof(uint32_t)), "hipMalloc fish chain rng");
        check_hip(hipMalloc(&ws->chain_rng_consumed, sizeof(int32_t)), "hipMalloc fish chain consumed");
        check_hip(hipMalloc(&ws->chain_codes, 16 * sizeof(int32_t)), "hipMalloc fish chain codes");
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
    check_hip(hipGetLastError(), "exact_topk_kernel fish sampler");
    check_hip(
        hipMemcpy(host_result, ws->packed, sizeof(HipFastTopKResult), hipMemcpyDeviceToHost),
        "hipMemcpy fish sampler result");
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
        check_hip(hipFree(ws->embeddings), "hipFree old fish embedding table");
        ws->embeddings = nullptr;
    }
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(dim) * sizeof(float);
    check_hip(hipMalloc(&ws->embeddings, bytes), "hipMalloc fish embedding table");
    check_hip(hipMemcpy(ws->embeddings, host_embeddings, bytes, hipMemcpyHostToDevice), "hipMemcpy fish embedding table");
    ws->embedding_rows = rows;
    ws->embedding_dim = dim;
}

void hip_fast_sampler_gather_embedding(
    void * workspace,
    int32_t row,
    float * device_output) {
    auto * ws = static_cast<Workspace *>(workspace);
    if (ws == nullptr || ws->embeddings == nullptr || device_output == nullptr ||
        row < 0 || row >= ws->embedding_rows || ws->embedding_dim <= 0) {
        throw std::invalid_argument("HIP Fish embedding gather received invalid arguments");
    }
    constexpr int threads = 256;
    const int blocks = (ws->embedding_dim + threads - 1) / threads;
    hipLaunchKernelGGL(
        gather_embedding_kernel,
        dim3(blocks), dim3(threads), 0, 0,
        ws->embeddings, ws->embedding_dim, row, device_output);
    check_hip(hipGetLastError(), "gather_embedding_kernel fish sampler");
}

void hip_fast_sampler_prepare_step(
    void * workspace,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask) {
    auto * ws = static_cast<Workspace *>(workspace);
    if (ws == nullptr || device_position == nullptr || device_mask == nullptr ||
        position < 0 || mask_length <= 0 || position >= mask_length) {
        throw std::invalid_argument("HIP Fish fast step preparation received invalid arguments");
    }
    constexpr int threads = 32;
    hipLaunchKernelGGL(
        prepare_fast_step_kernel,
        dim3(1), dim3(threads), 0, 0,
        position, mask_length, device_position, static_cast<uint16_t *>(device_mask));
    check_hip(hipGetLastError(), "prepare_fast_step_kernel fish sampler");
}

void hip_semantic_sampler_topk(
    void * workspace,
    const float * device_logits,
    int32_t semantic_begin,
    int32_t semantic_end,
    int32_t eos_index,
    int32_t top_k,
    HipFastTopKResult * host_result) {
    auto * ws = static_cast<Workspace *>(workspace);
    if (ws == nullptr || device_logits == nullptr || host_result == nullptr) {
        throw std::invalid_argument("HIP Fish semantic sampler received a null argument");
    }
    if (semantic_begin < 0 || semantic_end < semantic_begin || eos_index < 0) {
        throw std::invalid_argument("HIP Fish semantic sampler received invalid token bounds");
    }
    const bool eos_outside = eos_index < semantic_begin || eos_index > semantic_end;
    const int32_t candidate_count = semantic_end - semantic_begin + 1 + (eos_outside ? 1 : 0);
    if (top_k <= 0 || top_k > candidate_count || top_k > kHipFastSamplerMaxTopK) {
        throw std::invalid_argument("HIP Fish semantic sampler top_k is out of range");
    }
    constexpr int threads = 256;
    const size_t shared_bytes =
        static_cast<size_t>(candidate_count) * (sizeof(float) + sizeof(int32_t)) +
        static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t) + sizeof(double));
    hipLaunchKernelGGL(
        exact_semantic_topk_kernel,
        dim3(1),
        dim3(threads),
        shared_bytes,
        0,
        device_logits,
        semantic_begin,
        semantic_end,
        eos_index,
        candidate_count,
        top_k,
        ws->packed);
    check_hip(hipGetLastError(), "exact_semantic_topk_kernel fish sampler");
    check_hip(
        hipMemcpy(host_result, ws->packed, sizeof(HipFastTopKResult), hipMemcpyDeviceToHost),
        "hipMemcpy fish semantic sampler result");
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
