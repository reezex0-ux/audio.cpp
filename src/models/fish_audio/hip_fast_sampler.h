#pragma once

#include <cstddef>
#include <cstdint>

namespace engine::models::fish_audio::detail {

constexpr int kHipFastSamplerMaxTopK = 256;

struct HipFastTopKResult {
    int32_t count = 0;
    float max_logit = 0.0F;
    double denom = 0.0;
    float logits[kHipFastSamplerMaxTopK]{};
    int32_t indices[kHipFastSamplerMaxTopK]{};
};

void * hip_fast_sampler_create(int32_t vocab_size);
void hip_fast_sampler_destroy(void * workspace);
void hip_fast_sampler_topk(
    void * workspace,
    const float * device_logits,
    int32_t top_k,
    HipFastTopKResult * host_result);

// Phase 1K: gather the semantic range plus optional EOS on-device, then
// return only the exact top-k logits/ids needed by the existing CPU sampler.
void hip_fast_sampler_semantic_topk(
    void * workspace,
    const float * device_logits,
    int32_t semantic_begin,
    int32_t semantic_end,
    int32_t eos_index,
    int32_t source_size,
    int32_t top_k,
    HipFastTopKResult * host_result,
    void * stream);

void hip_fast_sampler_upload_embeddings(
    void * workspace,
    const float * host_embeddings,
    int32_t rows,
    int32_t dim);

void hip_fast_sampler_gather_embedding(
    void * workspace,
    int32_t row,
    float * device_output,
    void * stream);

// Phase 1I: keep generated-frame slow embedding construction and per-step
// state preparation on the current ggml HIP stream. F32/F16/BF16 tables stay
// in their native storage type and selected values are accumulated as F32.
void * hip_slow_step_create();
void hip_slow_step_destroy(void * workspace);
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
    int32_t num_codebooks);
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
    void * stream);

void hip_fast_sampler_prepare_step(
    void * workspace,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    void * stream);

// Phase 1J: prime Fast-AR directly from a Slow-AR hidden vector already on
// the same HIP device, while preparing position/mask in the same kernel.
void hip_fast_sampler_prime_device(
    void * workspace,
    const float * device_source,
    int32_t dim,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask,
    float * device_input,
    void * stream);


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
    void * stream);

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
    void * stream);

void hip_fast_sampler_chain_finish(
    void * workspace,
    int32_t count,
    int32_t * host_codes,
    int32_t * host_rng_words_consumed,
    void * stream);

}  // namespace engine::models::fish_audio::detail
