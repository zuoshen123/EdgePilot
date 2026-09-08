#include "edgepilot/decoder/speculative_decoder.h"
#include <atomic>
#include <cmath>

namespace edgepilot {

struct SpeculativeDecoder::Impl {
    std::unique_ptr<InferenceBackend> draft_model;
    std::unique_ptr<InferenceBackend> target_model;
    Config config;

    std::atomic<bool> initialized{false};
    std::atomic<bool> cancelled{false};

    // 统计
    std::atomic<int> total_draft_tokens{0};
    std::atomic<int> total_accepted_tokens{0};
    std::atomic<int> total_target_calls{0};
    std::atomic<float> avg_acceptance_rate{0.0f};
    std::atomic<int> current_window_size{4};

    /// 动态调整猜测窗口: 接受率高 → 增大窗口，接受率低 → 减小窗口
    void adjustWindow(float acceptance_rate) {
        if (!config.dynamic_window) return;

        int window = current_window_size.load();
        if (acceptance_rate > 0.7f && window < 8) {
            current_window_size.store(window + 1);
        } else if (acceptance_rate < 0.3f && window > 2) {
            current_window_size.store(window - 1);
        }
    }
};

SpeculativeDecoder::SpeculativeDecoder()
    : impl_(std::make_unique<Impl>()) {}

SpeculativeDecoder::~SpeculativeDecoder() {
    release();
}

Status SpeculativeDecoder::initialize(const ModelConfig& draft_config,
                                       const ModelConfig& target_config,
                                       const Config& spec_config) {
    impl_->config = spec_config;
    impl_->current_window_size.store(spec_config.speculation_window);

    // TODO: 实际加载模型
    // Phase 2 实现: 接入真实后端
    // impl_->draft_model = BackendFactory::create(draft_config.backend);
    // impl_->target_model = BackendFactory::create(target_config.backend);

    impl_->initialized.store(true);
    return Status::OK;
}

void SpeculativeDecoder::release() {
    impl_->cancelled.store(true);
    if (impl_->draft_model) impl_->draft_model->unload();
    if (impl_->target_model) impl_->target_model->unload();
    impl_->initialized.store(false);
}

GenerateResult SpeculativeDecoder::generate(const GenerateRequest& request) {
    GenerateResult result{};
    if (!impl_->initialized.load()) return result;

    auto start = std::chrono::steady_clock::now();
    std::vector<int> generated;
    std::vector<float> itl_samples;

    // 1. Prefill 阶段
    auto prefill_start = std::chrono::steady_clock::now();
    // draft_model->prefill(request.tokens);
    // target_model->prefill(request.tokens);
    auto prefill_end = std::chrono::steady_clock::now();
    result.ttft_ms = std::chrono::duration<float, std::milli>(
        prefill_end - prefill_start).count();

    // 2. 投机解码循环
    int K = impl_->current_window_size.load();
    int total_proposed = 0;
    int total_accepted = 0;

    while (static_cast<int>(generated.size()) < request.max_new_tokens
           && !impl_->cancelled.load()) {

        // 2a. Draft 模型快速生成 K 个候选
        auto draft_start = std::chrono::steady_clock::now();
        std::vector<int> candidates;
        // TODO: draft_model->generate K tokens
        // 临时: 模拟生成
        for (int i = 0; i < K; i++) {
            candidates.push_back(0);  // placeholder token
        }
        total_proposed += K;
        auto draft_end = std::chrono::steady_clock::now();

        // 2b. Target 模型并行验证
        auto verify_start = std::chrono::steady_clock::now();
        std::vector<TokenResult> verification;
        // TODO: target_model->verifyTokens(candidates)
        // 临时: 模拟全部接受
        for (int i = 0; i < K; i++) {
            TokenResult tr;
            tr.token_id = candidates[i];
            tr.is_eos = false;
            tr.probability = 0.9f;
            verification.push_back(tr);
        }
        auto verify_end = std::chrono::steady_clock::now();

        // 2c. 接受验证通过的 token
        int accepted_in_round = 0;
        for (const auto& v : verification) {
            if (v.is_eos) break;
            generated.push_back(v.token_id);
            accepted_in_round++;
        }
        total_accepted += accepted_in_round;

        // 记录 ITL
        float round_time = std::chrono::duration<float, std::milli>(
            verify_end - draft_start).count();
        if (accepted_in_round > 0) {
            itl_samples.push_back(round_time / accepted_in_round);
        }

        // 2d. 接受率反馈
        float acceptance_rate =
            static_cast<float>(accepted_in_round) / K;
        impl_->adjustWindow(acceptance_rate);

        // 3. 如果接受率太低，回退到纯 target
        if (acceptance_rate < impl_->config.min_acceptance_rate) {
            // TODO: 切换到 target_model 自回归生成
            break;
        }
    }

    // 计算最终统计
    auto end = std::chrono::steady_clock::now();
    result.generated_tokens = generated;
    result.total_tokens = static_cast<int>(generated.size());
    result.total_time_ms = std::chrono::duration<float, std::milli>(
        end - start).count();
    result.tokens_per_sec = (result.total_time_ms > 0)
        ? result.total_tokens * 1000.0f / result.total_time_ms
        : 0;
    result.draft_tokens_proposed = total_proposed;
    result.draft_tokens_accepted = total_accepted;
    result.acceptance_rate = (total_proposed > 0)
        ? static_cast<float>(total_accepted) / total_proposed
        : 0;

    // 更新全局统计
    impl_->total_draft_tokens += total_proposed;
    impl_->total_accepted_tokens += total_accepted;
    impl_->total_target_calls++;
    impl_->avg_acceptance_rate.store(result.acceptance_rate);

    return result;
}

void SpeculativeDecoder::generateAsync(const GenerateRequest& request,
                                        TokenCallback callback) {
    // 异步模式: 在独立线程中运行 generate，
    // 每产生一个 token 就通过 callback 返回
    // TODO: 实现异步版本
    auto result = generate(request);
    for (const auto& token_id : result.generated_tokens) {
        TokenResult tr;
        tr.token_id = token_id;
        tr.is_eos = false;
        if (callback) callback(tr);
    }
}

void SpeculativeDecoder::cancel() {
    impl_->cancelled.store(true);
}

bool SpeculativeDecoder::isInitialized() const {
    return impl_->initialized.load();
}

float SpeculativeDecoder::getAverageAcceptanceRate() const {
    return impl_->avg_acceptance_rate.load();
}

int SpeculativeDecoder::getCurrentWindowSize() const {
    return impl_->current_window_size.load();
}

SpeculativeDecoder::Stats SpeculativeDecoder::getStats() const {
    Stats s{};
    s.avg_acceptance_rate = impl_->avg_acceptance_rate.load();
    s.total_draft_tokens = impl_->total_draft_tokens.load();
    s.total_accepted_tokens = impl_->total_accepted_tokens.load();
    s.total_target_calls = impl_->total_target_calls.load();
    // TODO: 计算 avg_ttft_ms, avg_itl_ms, avg_tokens_per_sec
    return s;
}

} // namespace edgepilot
