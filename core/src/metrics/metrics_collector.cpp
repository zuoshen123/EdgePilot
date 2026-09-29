#include "edgepilot/metrics/metrics_collector.h"
#include "edgepilot/platform/platform_sampler.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <unordered_map>
#include <sstream>

namespace edgepilot {

namespace {
// JSON 安全字符串：仅保留内核目录/类型名的合法字符集（spec §②：zones/power_path 来自 sysfs）
std::string json_sanitize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || c=='-' || c=='_' || c=='.' || c=='/') out += c;  // '/' 保留：power_path 为绝对路径（R8）
    }
    return out;
}
} // namespace

// ============================================================
// MetricsCollector 实现
// ============================================================

struct MetricsCollector::Impl {
    Config config;
    std::atomic<bool> running{false};
    std::thread collect_thread;

    mutable std::mutex mutex;
    std::unordered_map<int, InferenceRecord> active_records;
    std::vector<InferenceRecord> history;

    MetricsCallback metrics_callback;

    // ---- v0.3 采样窗口 (spec §②) ----
    static constexpr size_t kMaxWinSamples = 20000;   // 每通道上限 ≈33min@10Hz，越界丢最旧
    PlatformSampler sampler;
    bool sampler_probed = false;
    std::mutex win_mtx;                                // 独立于 mutex：避免与记录锁交叉
    bool window_open = false;
    uint64_t window_seq = 0;   // 窗世代：beginWindow 递增，封死 end→begin 背靠背时旧窗样本混入新窗（R8）
    std::vector<PowerSample> win_power;
    std::vector<ThermalSample> win_thermal;
    std::vector<MemSample> win_mem;

    void ensureProbed() {   // 调用方持 win_mtx
        if (!sampler_probed) { sampler.probe(); sampler_probed = true; }
    }

    /// 后台采集线程
    void collectionLoop() {
        while (running.load()) {
            bool open;
            uint64_t seq;
            { std::lock_guard<std::mutex> lk(win_mtx); open = window_open; seq = window_seq; }
            if (open) {
                PowerSample ps; ThermalSample ts; MemSample ms;
                const bool rp = sampler.readPower(ps);
                const bool rt = sampler.readThermal(ts);
                const bool rm = sampler.readMem(ms);
                std::lock_guard<std::mutex> lk(win_mtx);
                // 二次确认：endWindow 可能在三读之间发生；seq 相等排除 end→begin 背靠背的跨窗混入（R8）
                if (window_open && window_seq == seq) {
                    if (rp) {
                        if (win_power.size() >= kMaxWinSamples) win_power.erase(win_power.begin());
                        win_power.push_back(ps);
                    }
                    if (rt) {
                        if (win_thermal.size() >= kMaxWinSamples) win_thermal.erase(win_thermal.begin());
                        win_thermal.push_back(ts);
                    }
                    if (rm) {
                        if (win_mem.size() >= kMaxWinSamples) win_mem.erase(win_mem.begin());
                        win_mem.push_back(ms);
                    }
                }
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config.sample_interval_ms));
        }
    }
};

MetricsCollector::MetricsCollector()
    : impl_(std::make_unique<Impl>()) {}

MetricsCollector::~MetricsCollector() {
    stop();
}

Status MetricsCollector::initialize(const Config& config) {
    impl_->config = config;
    if (impl_->config.sample_interval_ms <= 0) impl_->config.sample_interval_ms = 100;
    if (impl_->config.max_history_size <= 0)   impl_->config.max_history_size = 100;
    return Status::OK;
}

void MetricsCollector::start() {
    if (impl_->running.load()) return;
    impl_->running.store(true);
    impl_->collect_thread = std::thread(&Impl::collectionLoop, impl_.get());
}

void MetricsCollector::stop() {
    impl_->running.store(false);
    if (impl_->collect_thread.joinable()) {
        impl_->collect_thread.join();
    }
}

bool MetricsCollector::isRunning() const {
    return impl_->running.load();
}

void MetricsCollector::beginInference(int request_id, int agent_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    InferenceRecord record{};
    record.request_id = request_id;
    record.agent_id = agent_id;
    record.start_time = std::chrono::steady_clock::now();
    impl_->active_records[request_id] = record;
}

void MetricsCollector::recordTTFT(int request_id, float ttft_ms) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->active_records.find(request_id);
    if (it != impl_->active_records.end()) {
        it->second.ttft_ms = ttft_ms;
    }
}

void MetricsCollector::recordITL(int request_id, float itl_ms) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->active_records.find(request_id);
    if (it != impl_->active_records.end()) {
        it->second.itl_samples_ms.push_back(itl_ms);
    }
}

void MetricsCollector::endInference(int request_id,
                                     const GenerateResult& result) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->active_records.find(request_id);
    if (it == impl_->active_records.end()) return;

    auto& record = it->second;
    record.end_time = std::chrono::steady_clock::now();
    record.e2e_latency_ms = std::chrono::duration<float, std::milli>(
        record.end_time - record.start_time).count();
    record.tokens_per_sec = result.tokens_per_sec;
    record.total_tokens = result.total_tokens;
    record.draft_tokens_proposed = result.draft_tokens_proposed;
    record.draft_tokens_accepted = result.draft_tokens_accepted;
    record.acceptance_rate = result.acceptance_rate;

    // 计算 ITL 统计
    record.itl_stats = calculateITLStats(record.itl_samples_ms);

    // 移到历史
    impl_->history.push_back(record);
    impl_->active_records.erase(it);

    // 限制历史大小
    if (static_cast<int>(impl_->history.size()) > impl_->config.max_history_size) {
        impl_->history.erase(impl_->history.begin());
    }
}

MetricsCollector::ITLStats MetricsCollector::calculateITLStats(
    const std::vector<float>& samples) {
    if (samples.empty()) return {};

    ITLStats stats{};
    stats.sample_count = static_cast<int>(samples.size());

    std::vector<float> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    stats.min_ms = sorted.front();
    stats.max_ms = sorted.back();
    stats.avg_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0f)
                   / sorted.size();

    auto pct = [&](float p) {
        return sorted[static_cast<size_t>(p * (sorted.size() - 1))];
    };
    stats.p50_ms = pct(0.50f);
    stats.p95_ms = pct(0.95f);
    stats.p99_ms = pct(0.99f);

    return stats;
}

PerformanceMetrics MetricsCollector::getRecentMetrics(int count) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    PerformanceMetrics metrics{};

    if (impl_->history.empty()) return metrics;

    int n = std::min(count, static_cast<int>(impl_->history.size()));
    auto begin = impl_->history.end() - n;

    float sum_ttft = 0, sum_itl = 0, sum_tps = 0;
    for (auto it = begin; it != impl_->history.end(); ++it) {
        sum_ttft += it->ttft_ms;
        sum_itl += it->itl_stats.avg_ms;
        sum_tps += it->tokens_per_sec;
    }

    metrics.ttft_ms = sum_ttft / n;
    metrics.itl_avg_ms = sum_itl / n;
    metrics.tokens_per_sec = sum_tps / n;

    return metrics;
}

MetricsCollector::InferenceRecord MetricsCollector::getInferenceRecord(
    int request_id) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->active_records.find(request_id);
    if (it != impl_->active_records.end()) return it->second;

    for (const auto& record : impl_->history) {
        if (record.request_id == request_id) return record;
    }
    return {};
}

std::vector<MetricsCollector::InferenceRecord>
MetricsCollector::getHistory() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->history;
}

std::string MetricsCollector::exportToJSON() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::ostringstream oss;
    oss << "{\"records\":[";
    for (size_t i = 0; i < impl_->history.size(); i++) {
        const auto& r = impl_->history[i];
        if (i > 0) oss << ",";
        oss << "{"
            << "\"request_id\":" << r.request_id << ","
            << "\"ttft_ms\":" << r.ttft_ms << ","
            << "\"itl_avg_ms\":" << r.itl_stats.avg_ms << ","
            << "\"itl_p99_ms\":" << r.itl_stats.p99_ms << ","
            << "\"e2e_latency_ms\":" << r.e2e_latency_ms << ","
            << "\"tokens_per_sec\":" << r.tokens_per_sec << ","
            << "\"acceptance_rate\":" << r.acceptance_rate
            << "}";
    }
    oss << "]}";
    return oss.str();
}

void MetricsCollector::exportToSQLite(const std::string& path) const {
    // TODO: 写入 SQLite
    // 表结构: inference_records (request_id, agent_id, ttft_ms, itl_avg_ms, ...)
    (void)path;
}

void MetricsCollector::exportToCSV(const std::string& path) const {
    // TODO: 写入 CSV
    (void)path;
}

std::string MetricsCollector::samplerProbeJson() {
    std::lock_guard<std::mutex> lk(impl_->win_mtx);
    impl_->ensureProbed();
    const Capability& cap = impl_->sampler.capability();
    std::string j = "{\"power\":\"";      j += samplerStatusToString(cap.power);
    j += "\",\"thermal\":\"";             j += samplerStatusToString(cap.thermal);
    j += "\",\"mem\":\"";                 j += samplerStatusToString(cap.mem);
    j += "\",\"cpu\":\"";                 j += samplerStatusToString(cap.cpu);
    j += "\",\"power_path\":\"";          j += json_sanitize(cap.power_path);
    j += "\",\"zones\":[";
    for (size_t i = 0; i < cap.cpu_zone_types.size(); ++i) {
        if (i) j += ",";
        j += "\"" + json_sanitize(cap.cpu_zone_types[i]) + "\"";
    }
    j += "]}";
    return j;
}

void MetricsCollector::beginWindow() {
    std::lock_guard<std::mutex> lk(impl_->win_mtx);
    impl_->ensureProbed();
    impl_->win_power.clear();
    impl_->win_thermal.clear();
    impl_->win_mem.clear();
    ++impl_->window_seq;   // 在途旧窗样本的二次确认将因 seq 不符被丢弃（R8）
    impl_->window_open = true;
}

std::string MetricsCollector::endWindowJson() {
    std::lock_guard<std::mutex> lk(impl_->win_mtx);
    impl_->window_open = false;
    std::ostringstream o;
    o << std::fixed << std::setprecision(3);
    o << "{\"power\":[";
    for (size_t i = 0; i < impl_->win_power.size(); ++i) {
        if (i) o << ",";
        o << "[" << impl_->win_power[i].t_ms << "," << impl_->win_power[i].power_mw << "]";
    }
    o << "],\"thermal\":[";
    for (size_t i = 0; i < impl_->win_thermal.size(); ++i) {
        if (i) o << ",";
        o << "[" << impl_->win_thermal[i].t_ms << "," << impl_->win_thermal[i].cpu_max_c << "]";
    }
    o << "],\"mem\":[";
    for (size_t i = 0; i < impl_->win_mem.size(); ++i) {
        if (i) o << ",";
        o << "[" << impl_->win_mem[i].t_ms << "," << impl_->win_mem[i].pss_kb
          << "," << impl_->win_mem[i].vmrss_kb << "]";
    }
    o << "],\"n\":{\"power\":" << impl_->win_power.size()
      << ",\"thermal\":" << impl_->win_thermal.size()
      << ",\"mem\":" << impl_->win_mem.size() << "}}";
    impl_->win_power.clear();
    impl_->win_thermal.clear();
    impl_->win_mem.clear();
    return o.str();
}

void MetricsCollector::setMetricsCallback(MetricsCallback callback) {
    impl_->metrics_callback = std::move(callback);
}

// ============================================================
// Profiler 实现 (简化版)
// ============================================================

static std::unordered_map<Profiler::Event, Profiler::EventStats> g_event_stats;

void Profiler::beginEvent(Event event) {
    // TODO: 记录时间戳
    (void)event;
}

void Profiler::endEvent(Event event) {
    // TODO: 计算耗时并更新统计
    (void)event;
}

Profiler::EventStats Profiler::getEventStats(Event event) {
    auto it = g_event_stats.find(event);
    if (it != g_event_stats.end()) return it->second;
    return {};
}

std::string Profiler::exportChromeTrace() {
    // TODO: 生成 Chrome Tracing 格式 JSON
    return "{}";
}

std::string Profiler::generateReport() {
    // TODO: 生成可读报告
    return "Profiler report not yet implemented";
}

void Profiler::reset() {
    g_event_stats.clear();
}

} // namespace edgepilot
