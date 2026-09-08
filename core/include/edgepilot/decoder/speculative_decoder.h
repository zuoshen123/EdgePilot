#pragma once

#include "edgepilot/backend/inference_backend.h"
#include "edgepilot/common/types.h"
#include <memory>

namespace edgepilot {

/**
 * 投机采样解码器 (Speculative Decoding)
 *
 * 核心思路: 用小模型 (Draft) 快速猜测多个 token，再用大模型 (Target) 并行验证。
 * 优势: TTFT 降低 40-60%，保持与纯大模型相同的输出质量。
 *
 * 流程:
 *   1. Draft model 自回归生成 K 个候选 token (快)
 *   2. Target model 对 K 个候选做一次 forward pass (并行验证)
 *   3. 从左到右找到第一个被拒绝的位置
 *   4. 接受之前的 token，从拒绝位置重新采样
 *   5. 重复直到生成完整回复
 */
class SpeculativeDecoder {
public:
    struct Config {
        int speculation_window;      // 每轮猜测的 token 数 (默认 4)
        float min_acceptance_rate;   // 最低接受率，低于此值回退到纯 target
        bool dynamic_window;         // 动态调整猜测窗口大小
        bool enable_prompt_cache;    // 是否启用 Prompt Cache
    };

    SpeculativeDecoder();
    ~SpeculativeDecoder();

    /// 初始化: 加载 Draft 和 Target 模型
    Status initialize(const ModelConfig& draft_config,
                      const ModelConfig& target_config,
                      const Config& spec_config);

    /// 释放所有资源
    void release();

    // ---- 推理 ----

    /// 同步生成
    GenerateResult generate(const GenerateRequest& request);

    /// 异步生成 (逐 token 回调)
    void generateAsync(const GenerateRequest& request, TokenCallback callback);

    /// 取消
    void cancel();

    // ---- 状态 ----

    bool isInitialized() const;
    float getAverageAcceptanceRate() const;
    int getCurrentWindowSize() const;

    /// 获取性能统计
    struct Stats {
        float avg_ttft_ms;
        float avg_itl_ms;
        float avg_tokens_per_sec;
        float avg_acceptance_rate;
        int total_draft_tokens;
        int total_accepted_tokens;
        int total_target_calls;
    };
    Stats getStats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace edgepilot
