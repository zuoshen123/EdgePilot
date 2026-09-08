#pragma once

#include "edgepilot/backend/inference_backend.h"
#include <memory>

// Forward declare llama.cpp types
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;

namespace edgepilot {

/**
 * GGML 后端 — 基于 llama.cpp 的推理实现
 *
 * 支持 GGUF 格式模型 (TinyLlama, Phi-2, Llama 3 等)。
 * 在 Android 上使用 CPU 推理，在 iOS 上使用 Metal 加速。
 */
class GgmlBackend : public InferenceBackend {
public:
    GgmlBackend();
    ~GgmlBackend() override;

    // ---- InferenceBackend 接口实现 ----

    Status loadModel(const ModelConfig& config) override;
    void unload() override;
    bool isLoaded() const override;

    GenerateResult generate(const GenerateRequest& request) override;
    void generateAsync(const GenerateRequest& request, TokenCallback callback) override;
    void cancel() override;

    float prefill(const std::vector<int>& tokens) override;
    TokenResult decodeNext() override;
    std::vector<TokenResult> verifyTokens(const std::vector<int>& candidates) override;

    HardwareInfo getHardwareInfo() const override;
    KVCacheInfo getKVCacheInfo() const override;
    InferenceState getState() const override;
    std::string getName() const override;

    std::vector<uint8_t> exportKVCache() const override;
    bool importKVCache(const std::vector<uint8_t>& data) override;
    void clearKVCache() override;
    bool compressKVCache(int target_bits) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edgepilot
