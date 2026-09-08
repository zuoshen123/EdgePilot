#pragma once

#include "edgepilot/common/types.h"
#include <string>

namespace edgepilot {

/**
 * 推理后端抽象接口
 *
 * 所有推理引擎 (ExecuTorch, ncnn, GGML) 必须实现此接口。
 * 提供统一的模型加载、推理和资源管理 API。
 */
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    // ---- 生命周期 ----

    /// 加载模型，返回是否成功
    virtual Status loadModel(const ModelConfig& config) = 0;

    /// 释放模型和所有资源
    virtual void unload() = 0;

    /// 模型是否已加载
    virtual bool isLoaded() const = 0;

    // ---- 推理 ----

    /// 同步生成（阻塞直到完成或超时）
    virtual GenerateResult generate(const GenerateRequest& request) = 0;

    /// 异步生成（通过回调逐 token 返回）
    virtual void generateAsync(const GenerateRequest& request,
                               TokenCallback callback) = 0;

    /// 取消正在进行的推理
    virtual void cancel() = 0;

    /// Prefill 阶段：处理输入 tokens，填充 KV Cache
    /// 返回 prefill 吞吐量 (tokens/sec)
    virtual float prefill(const std::vector<int>& tokens) = 0;

    /// Decode 单个 token（用于投机采样的逐 token 验证）
    virtual TokenResult decodeNext() = 0;

    /// 批量验证 tokens（投机采样核心方法）
    /// 输入 K 个候选 token，返回每个 token 的接受/拒绝信息
    virtual std::vector<TokenResult> verifyTokens(
        const std::vector<int>& candidates) = 0;

    // ---- 硬件 & 状态 ----

    /// 获取硬件能力
    virtual HardwareInfo getHardwareInfo() const = 0;

    /// 获取 KV Cache 信息
    virtual KVCacheInfo getKVCacheInfo() const = 0;

    /// 获取当前推理状态
    virtual InferenceState getState() const = 0;

    /// 获取后端名称
    virtual std::string getName() const = 0;

    // ---- KV Cache 管理 ----

    /// 导出当前 KV Cache 状态（用于 Prompt Cache）
    virtual std::vector<uint8_t> exportKVCache() const = 0;

    /// 导入 KV Cache 状态
    virtual bool importKVCache(const std::vector<uint8_t>& data) = 0;

    /// 释放 KV Cache 内存
    virtual void clearKVCache() = 0;

    /// 压缩 KV Cache (FP16 -> INT4/INT8)
    virtual bool compressKVCache(int target_bits) = 0;

protected:
    InferenceState state_ = InferenceState::IDLE;
};

} // namespace edgepilot
