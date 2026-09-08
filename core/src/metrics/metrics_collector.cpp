#include "edgepilot/metrics/metrics_collector.h"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <sstream>

namespace edgepilot {

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

    /// 后台采集线程
    void collectionLoop() {
        while (running.load()) {
            // 采集系统级指标 (CPU/GPU/内存/温度)
            // TODO: 从平台 API 读取

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
