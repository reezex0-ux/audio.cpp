#pragma once

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

void hip_fast_sampler_upload_embeddings(
    void * workspace,
    const float * host_embeddings,
    int32_t rows,
    int32_t dim);

void hip_fast_sampler_gather_embedding(
    void * workspace,
    int32_t row,
    float * device_output);

void hip_fast_sampler_prepare_step(
    void * workspace,
    int32_t position,
    int32_t mask_length,
    int32_t * device_position,
    void * device_mask);

void hip_semantic_sampler_topk(
    void * workspace,
    const float * device_logits,
    int32_t semantic_begin,
    int32_t semantic_end,
    int32_t eos_index,
    int32_t top_k,
    HipFastTopKResult * host_result);

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
