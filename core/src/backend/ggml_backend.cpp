#include "edgepilot/backend/ggml_backend.h"

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "edgepilot/common/log.h"

#ifdef __ANDROID__
#include <sys/sysinfo.h>
#endif

namespace edgepilot {

// 流式/同步早退共用：失败必须让用户看见（N4，spec §⑤）
static constexpr const char* kPrefillError = "[错误] prefill 失败（可能超出上下文）";

// nearest-rank 分位数：idx = ceil(q*n) - 1（升序数组）
static double percentile_nearest_rank(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) return 0.0;
    size_t idx = static_cast<size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    if (idx == 0) idx = 1;
    if (idx > sorted.size()) idx = sorted.size();
    return sorted[idx - 1];
}

// 由逐 token 生成时刻序列计算 ITL 统计（序列本身 + avg/P50/P90/P99）
static void compute_itl_metrics(GenerateResult& r, const std::vector<double>& token_times) {
    r.itl_series_ms.clear();
    for (size_t i = 1; i < token_times.size(); ++i)
        r.itl_series_ms.push_back(token_times[i] - token_times[i - 1]);

    if (r.itl_series_ms.empty()) {
        r.itl_avg_ms = r.itl_p50_ms = r.itl_p90_ms = r.itl_p99_ms = 0.0f;
        return;
    }
    double sum = 0.0;
    for (double v : r.itl_series_ms) sum += v;
    r.itl_avg_ms = static_cast<float>(sum / static_cast<double>(r.itl_series_ms.size()));

    auto sorted = r.itl_series_ms;
    std::sort(sorted.begin(), sorted.end());
    r.itl_p50_ms = static_cast<float>(percentile_nearest_rank(sorted, 0.50));
    r.itl_p90_ms = static_cast<float>(percentile_nearest_rank(sorted, 0.90));
    r.itl_p99_ms = static_cast<float>(percentile_nearest_rank(sorted, 0.99));
}

// ============================================================================
// GgmlBackend::Impl — 内部实现
// ============================================================================
struct GgmlBackend::Impl {
    // ---- llama.cpp 对象 ----
    llama_model*    model    = nullptr;
    llama_context*  ctx      = nullptr;
    llama_sampler*  sampler  = nullptr;
    const llama_vocab* vocab = nullptr;

    // ---- 配置 ----
    ModelConfig config;

    // ---- 生成状态 ----
    std::mutex  mtx;
    std::atomic<bool> cancelled{false};
    int32_t     n_past    = 0;    // KV cache 中已有的 token 数

    // ---- 指标 ----
    int64_t     total_tokens_generated = 0;
    double      total_generation_ms    = 0.0;
    std::vector<double> itl_history;
    GenerateResult last_metrics{};

    // ---- 工具方法 ----

    static double now_ms() {
        using namespace std::chrono;
        return duration<double, std::milli>(
            steady_clock::now().time_since_epoch()
        ).count();
    }

    llama_sampler* buildSampler() {
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler* chain = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        return chain;
    }

    std::string tokenToPiece(llama_token token) {
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n <= 0) return "";
        return std::string(buf, n);
    }

    std::vector<llama_token> tokenize(const std::string& text, bool add_special = true) {
        int n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, add_special, false);
        if (n == 0) return {};
        if (n < 0) n = -n;  // 负数表示需要的 token 数量
        std::vector<llama_token> tokens(n);
        llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                       tokens.data(), n, add_special, false);
        return tokens;
    }

    void resetKVCache() {
        if (ctx) {
            llama_memory_clear(llama_get_memory(ctx), true);
            n_past = 0;
        }
    }
};

// ============================================================================
// 构造 / 析构
// ============================================================================
GgmlBackend::GgmlBackend() : impl_(std::make_unique<Impl>()) {
    llama_backend_init();
}

GgmlBackend::~GgmlBackend() {
    unload();
}

// ============================================================================
// loadModel
// ============================================================================
Status GgmlBackend::loadModel(const ModelConfig& config) {
    if (impl_->model) unload();

    impl_->config = config;
    state_ = InferenceState::LOADING;

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = config.gpu_layers;

    impl_->model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (!impl_->model) {
        state_ = InferenceState::ERROR;
        return Status::MODEL_LOAD_FAILED;
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);
    EP_LOGI("vocab=%p", (void*)impl_->vocab);

    auto cparams = llama_context_default_params();
    cparams.n_ctx   = config.context_length > 0 ? config.context_length : 2048;
    cparams.n_batch = config.batch_size > 0 ? config.batch_size : 512;
    cparams.n_threads = config.threads > 0 ? config.threads
                        : std::min(static_cast<int32_t>(std::thread::hardware_concurrency()), 4);
    cparams.n_threads_batch = cparams.n_threads;

    impl_->ctx = llama_init_from_model(impl_->model, cparams);
    if (!impl_->ctx) {
        llama_model_free(impl_->model);
        impl_->model = nullptr;
        state_ = InferenceState::ERROR;
        return Status::MODEL_LOAD_FAILED;
    }

    impl_->sampler = impl_->buildSampler();
    impl_->n_past = 0;
    impl_->total_tokens_generated = 0;
    impl_->total_generation_ms = 0.0;
    impl_->itl_history.clear();

    state_ = InferenceState::IDLE;
    return Status::OK;
}

void GgmlBackend::unload() {
    if (impl_->sampler) { llama_sampler_free(impl_->sampler); impl_->sampler = nullptr; }
    if (impl_->ctx)     { llama_free(impl_->ctx);              impl_->ctx = nullptr; }
    if (impl_->model)   { llama_model_free(impl_->model);      impl_->model = nullptr; }
    impl_->vocab = nullptr;
    impl_->n_past = 0;
    state_ = InferenceState::IDLE;
}

bool GgmlBackend::isLoaded() const {
    return impl_->model != nullptr && impl_->ctx != nullptr;
}

// ============================================================================
// generate — 同步生成
// ============================================================================
GenerateResult GgmlBackend::generate(const GenerateRequest& request) {
    GenerateResult result{};
    if (!isLoaded()) return result;

    state_ = InferenceState::DECODING;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->cancelled = false;

    double gen_start = Impl::now_ms();
    std::vector<double> token_times;

    // Tokenize prompt
    auto prompt_tokens = impl_->tokenize(request.prompt, true);
    EP_LOGI("tokenize: %d tokens", (int)prompt_tokens.size());
    if (prompt_tokens.empty()) {
        result.generated_text = kPrefillError;
        state_ = InferenceState::IDLE;
        return result;
    }

    // Prefill
    impl_->resetKVCache();
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            static_cast<int32_t>(prompt_tokens.size()));
    int prefill_ret = llama_decode(impl_->ctx, batch);
    EP_LOGI("prefill decode ret=%d", prefill_ret);
    if (prefill_ret != 0) {
        result.generated_text = kPrefillError;
        state_ = InferenceState::IDLE;
        return result;
    }
    impl_->n_past += static_cast<int32_t>(prompt_tokens.size());

    double first_token_time = 0.0;
    int32_t n_predict = request.max_new_tokens > 0 ? request.max_new_tokens : 256;
    llama_token new_token_id = 0;

    for (int32_t i = 0; i < n_predict; ++i) {
        if (impl_->cancelled) break;

        new_token_id = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);
        if (i < 3) EP_LOGI("sample[%d] id=%d eog=%d", i, new_token_id, llama_vocab_is_eog(impl_->vocab, new_token_id));
        if (llama_vocab_is_eog(impl_->vocab, new_token_id)) break;

        if (i == 0) {
            first_token_time = Impl::now_ms();
            result.ttft_ms = static_cast<float>(first_token_time - gen_start);
        }

        std::string piece = impl_->tokenToPiece(new_token_id);
        result.generated_text += piece;
        result.generated_tokens.push_back(static_cast<int>(new_token_id));
        result.total_tokens++;
        token_times.push_back(Impl::now_ms());

        llama_batch next_batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(impl_->ctx, next_batch) != 0) break;
        impl_->n_past++;
    }

    double gen_end = Impl::now_ms();
    result.total_time_ms = static_cast<float>(gen_end - gen_start);
    if (result.total_tokens > 0 && result.total_time_ms > 0) {
        result.tokens_per_sec = result.total_tokens / (result.total_time_ms / 1000.0f);
    }

    compute_itl_metrics(result, token_times);
    impl_->last_metrics = result;   // generate() 顶部已全程持有 impl_->mtx，此处直接赋值、切勿再次加锁（非递归锁会死锁）

    impl_->total_tokens_generated += result.total_tokens;
    impl_->total_generation_ms += result.total_time_ms;

    state_ = InferenceState::IDLE;
    return result;
}

// ============================================================================
// generateAsync — 流式生成
// ============================================================================
void GgmlBackend::generateAsync(const GenerateRequest& request, TokenCallback callback) {
    if (!isLoaded()) {
        TokenResult tr{};
        tr.is_eos = true;
        callback(tr);
        return;
    }

    state_ = InferenceState::DECODING;
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->cancelled = false;

    double gen_start = Impl::now_ms();
    std::vector<double> token_times;
    GenerateResult run{};

    auto prompt_tokens = impl_->tokenize(request.prompt, true);
    if (prompt_tokens.empty()) {
        run.generated_text = kPrefillError;
        run.total_time_ms = static_cast<float>(Impl::now_ms() - gen_start);
        compute_itl_metrics(run, token_times);
        impl_->last_metrics = run;  // 失败早退也定格本次(全零)指标，防 getLastMetrics 吐陈旧数据
        TokenResult tr{}; tr.is_eos = true;
        callback(tr);
        state_ = InferenceState::IDLE;
        return;
    }

    impl_->resetKVCache();
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            static_cast<int32_t>(prompt_tokens.size()));
    if (llama_decode(impl_->ctx, batch) != 0) {
        run.generated_text = kPrefillError;
        run.total_time_ms = static_cast<float>(Impl::now_ms() - gen_start);
        compute_itl_metrics(run, token_times);
        impl_->last_metrics = run;  // 同上：prompt 超长致 prefill 失败时，onDone 报本次全零而非上次结果
        TokenResult tr{}; tr.is_eos = true;
        callback(tr);
        state_ = InferenceState::IDLE;
        return;
    }
    impl_->n_past += static_cast<int32_t>(prompt_tokens.size());

    int32_t n_predict = request.max_new_tokens > 0 ? request.max_new_tokens : 256;
    llama_token new_token_id = 0;

    for (int32_t i = 0; i < n_predict; ++i) {
        if (impl_->cancelled) {
            TokenResult tr{}; tr.is_eos = true;
            callback(tr);
            break;
        }

        new_token_id = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);
        if (llama_vocab_is_eog(impl_->vocab, new_token_id)) {
            TokenResult tr{}; tr.is_eos = true;
            callback(tr);
            break;
        }

        TokenResult tr{};
        tr.token_id = static_cast<int>(new_token_id);
        tr.token_text = impl_->tokenToPiece(new_token_id);
        tr.is_eos = false;

        token_times.push_back(Impl::now_ms());
        run.generated_text += tr.token_text;
        run.generated_tokens.push_back(tr.token_id);
        run.total_tokens++;
        if (run.total_tokens == 1)
            run.ttft_ms = static_cast<float>(Impl::now_ms() - gen_start);

        callback(tr);

        llama_batch next_batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(impl_->ctx, next_batch) != 0) {
            TokenResult end_tr{}; end_tr.is_eos = true;
            callback(end_tr);
            break;
        }
        impl_->n_past++;
    }

    run.total_time_ms = static_cast<float>(Impl::now_ms() - gen_start);
    if (run.total_tokens > 0 && run.total_time_ms > 0)
        run.tokens_per_sec = run.total_tokens / (run.total_time_ms / 1000.0f);
    compute_itl_metrics(run, token_times);
    impl_->last_metrics = run;      // 全程持锁，安全
    impl_->total_tokens_generated += run.total_tokens;
    impl_->total_generation_ms += run.total_time_ms;

    state_ = InferenceState::IDLE;
}

void GgmlBackend::cancel() {
    impl_->cancelled = true;
}

// ============================================================================
// prefill
// ============================================================================
float GgmlBackend::prefill(const std::vector<int>& tokens) {
    if (!isLoaded() || tokens.empty()) return -1.0f;

    auto t0 = Impl::now_ms();
    impl_->resetKVCache();

    std::vector<llama_token> ltokens(tokens.begin(), tokens.end());
    llama_batch batch = llama_batch_get_one(ltokens.data(),
                                            static_cast<int32_t>(ltokens.size()));
    if (llama_decode(impl_->ctx, batch) != 0) return -1.0f;
    impl_->n_past += static_cast<int32_t>(tokens.size());

    return static_cast<float>(Impl::now_ms() - t0);
}

// ============================================================================
// decodeNext
// ============================================================================
TokenResult GgmlBackend::decodeNext() {
    TokenResult tr{};
    if (!isLoaded()) return tr;

    llama_token id = llama_sampler_sample(impl_->sampler, impl_->ctx, -1);

    tr.token_id = static_cast<int>(id);
    tr.token_text = impl_->tokenToPiece(id);
    tr.is_eos = llama_vocab_is_eog(impl_->vocab, id);

    float* logits = llama_get_logits(impl_->ctx);
    if (logits) {
        int n_vocab = llama_vocab_n_tokens(impl_->vocab);
        float max_logit = -1e9f;
        for (int i = 0; i < n_vocab; ++i)
            if (logits[i] > max_logit) max_logit = logits[i];
        float sum_exp = 0.0f;
        for (int i = 0; i < n_vocab; ++i)
            sum_exp += std::exp(logits[i] - max_logit);
        tr.probability = std::exp(logits[id] - max_logit) / sum_exp;
    }

    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_decode(impl_->ctx, batch);
    impl_->n_past++;

    return tr;
}

// ============================================================================
// verifyTokens
// ============================================================================
std::vector<TokenResult> GgmlBackend::verifyTokens(const std::vector<int>& candidates) {
    std::vector<TokenResult> results;
    if (!isLoaded() || candidates.empty()) return results;

    float* logits = llama_get_logits(impl_->ctx);
    int n_vocab = llama_vocab_n_tokens(impl_->vocab);

    for (int token_id : candidates) {
        TokenResult tr{};
        tr.token_id = token_id;
        tr.token_text = impl_->tokenToPiece(static_cast<llama_token>(token_id));

        if (logits && token_id >= 0 && token_id < n_vocab) {
            float max_logit = -1e9f;
            for (int i = 0; i < n_vocab; ++i)
                if (logits[i] > max_logit) max_logit = logits[i];
            float sum_exp = 0.0f;
            for (int i = 0; i < n_vocab; ++i)
                sum_exp += std::exp(logits[i] - max_logit);
            tr.probability = std::exp(logits[token_id] - max_logit) / sum_exp;
        }

        llama_token lid = static_cast<llama_token>(token_id);
        llama_batch batch = llama_batch_get_one(&lid, 1);
        llama_decode(impl_->ctx, batch);
        impl_->n_past++;

        results.push_back(tr);
    }
    return results;
}

// ============================================================================
// 信息查询
// ============================================================================
HardwareInfo GgmlBackend::getHardwareInfo() const {
    HardwareInfo info{};
    info.soc_name = "llama.cpp";

#ifdef __ANDROID__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_memory_bytes = si.totalram * si.mem_unit;
        info.available_memory_bytes = si.freeram * si.mem_unit;
    }
#else
    info.total_memory_bytes = 8ULL * 1024 * 1024 * 1024;
    info.available_memory_bytes = 4ULL * 1024 * 1024 * 1024;
#endif

    if (impl_->model) {
        char desc[256];
        llama_model_desc(impl_->model, desc, sizeof(desc));
        info.soc_name = std::string(desc);
    }
    info.gpu_name = "N/A";
    return info;
}

KVCacheInfo GgmlBackend::getKVCacheInfo() const {
    KVCacheInfo info{};
    if (impl_->ctx && impl_->model) {
        info.max_seq_len = static_cast<int>(llama_n_ctx(impl_->ctx));
        info.num_layers  = static_cast<int>(llama_model_n_layer(impl_->model));
        const int n_head     = static_cast<int>(llama_model_n_head(impl_->model));
        const int n_head_kv  = static_cast<int>(llama_model_n_head_kv(impl_->model));
        const int n_embd     = static_cast<int>(llama_model_n_embd(impl_->model));
        info.num_heads = n_head;
        info.head_dim  = n_head > 0 ? n_embd / n_head : 0;
        // K+V × 层 × kv头 × head_dim × F16(2B)；默认缓存类型 F16（cparams 未显式量化）
        const size_t per_tok = 2ULL * static_cast<size_t>(info.num_layers)
                               * static_cast<size_t>(n_head_kv > 0 ? n_head_kv : n_head)
                               * static_cast<size_t>(info.head_dim) * 2ULL;
        info.used_memory_bytes  = static_cast<size_t>(impl_->n_past) * per_tok;
        info.total_memory_bytes = static_cast<size_t>(info.max_seq_len) * per_tok;
    }
    return info;
}

InferenceState GgmlBackend::getState() const { return state_; }

std::string GgmlBackend::getName() const { return "llama.cpp (GGML)"; }

GenerateResult GgmlBackend::getLastMetrics() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->last_metrics;
}

std::vector<uint8_t> GgmlBackend::exportKVCache() const { return {}; }
bool GgmlBackend::importKVCache(const std::vector<uint8_t>&) { return false; }
void GgmlBackend::clearKVCache() { impl_->resetKVCache(); }
bool GgmlBackend::compressKVCache(int) { return false; }

} // namespace edgepilot
