#include "edgepilot/scheduler/agent_scheduler.h"
#include <queue>
#include <unordered_map>
#include <atomic>

namespace edgepilot {

// ============================================================
// AgentScheduler 实现
// ============================================================

struct AgentScheduler::Impl {
    Config config;
    InferenceBackend* backend = nullptr;

    // 请求队列 (优先级队列)
    struct ComparePriority {
        bool operator()(const AgentRequest& a, const AgentRequest& b) {
            // 优先级数字越小越优先
            if (a.priority != b.priority)
                return static_cast<int>(a.priority) > static_cast<int>(b.priority);
            // 同优先级按提交时间排序 (FIFO)
            return false;
        }
    };
    std::priority_queue<AgentRequest, std::vector<AgentRequest>,
                        ComparePriority> request_queue;

    // 活跃推理
    struct ActiveInference {
        AgentRequest request;
        std::chrono::steady_clock::time_point start_time;
        GenerateResult result;
        std::atomic<bool> completed{false};
    };
    std::unordered_map<int, std::shared_ptr<ActiveInference>> active_inferences;

    // 状态
    std::atomic<int> next_request_id{1};
    std::atomic<DegradationLevel> current_degradation{0};
    std::atomic<float> thermal_pressure{0.0f};
    std::atomic<float> memory_pressure{0.0f};

    mutable std::mutex mutex;
    std::condition_variable cv;

    StatusCallback status_callback;

    // ---- 调度逻辑 ----

    /// 检查是否可以接受新请求
    bool canAccept(const AgentRequest& request) {
        // 检查并发数
        if (active_inferences.size() >=
            static_cast<size_t>(config.max_concurrent_inference)) {
            return false;
        }

        // 检查内存
        if (memory_pressure.load() > config.memory_warning_threshold) {
            if (request.priority > Priority::HIGH) {
                return false;  // 低优先级请求被拒绝
            }
        }

        // 检查温度
        if (thermal_pressure.load() > config.thermal_critical_threshold) {
            if (request.priority > Priority::REALTIME) {
                return false;  // 只有 REALTIME 不被拒绝
            }
        }

        return true;
    }

    /// 计算当前降级级别
    DegradationLevel calculateDegradation() {
        float thermal = thermal_pressure.load();
        float memory = memory_pressure.load();

        if (thermal > config.thermal_critical_threshold) return 4;
        if (thermal > config.thermal_warning_threshold) return 3;
        if (memory > config.memory_warning_threshold) return 2;
        if (thermal > 0.5f || memory > 0.5f) return 1;
        return 0;
    }

    /// 根据降级级别调整请求
    void applyDegradation(AgentRequest& request, DegradationLevel level) {
        switch (level) {
            case 0:
                // 无降级
                break;
            case 1:
                // 减小 batch size
                request.inference_request.max_new_tokens =
                    std::min(request.inference_request.max_new_tokens, 256);
                break;
            case 2:
                // 降低精度 (通过调整配置)
                request.inference_request.max_new_tokens =
                    std::min(request.inference_request.max_new_tokens, 128);
                break;
            case 3:
                // 切换到更小模型 (标记降级)
                request.allow_degradation = true;
                break;
            case 4:
                // 暂停低优先级
                request.allow_degradation = true;
                break;
        }
    }
};

AgentScheduler::AgentScheduler()
    : impl_(std::make_unique<Impl>()) {}

AgentScheduler::~AgentScheduler() = default;

Status AgentScheduler::initialize(const Config& config,
                                   InferenceBackend* backend) {
    impl_->config = config;
    impl_->backend = backend;
    return Status::OK;
}

int AgentScheduler::submitRequest(const AgentRequest& request) {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    int request_id = impl_->next_request_id++;

    if (!impl_->canAccept(request)) {
        // 加入队列等待
        impl_->request_queue.push(request);
    } else {
        // 立即执行
        auto active = std::make_shared<Impl::ActiveInference>();
        active->request = request;
        active->start_time = std::chrono::steady_clock::now();
        impl_->active_inferences[request_id] = active;

        // 计算降级
        auto degradation = impl_->calculateDegradation();
        impl_->current_degradation.store(degradation);
        impl_->applyDegradation(active->request, degradation);
    }

    impl_->cv.notify_all();
    return request_id;
}

GenerateResult AgentScheduler::waitForResult(
    int request_id, std::chrono::milliseconds timeout) {

    std::unique_lock<std::mutex> lock(impl_->mutex);

    auto it = impl_->active_inferences.find(request_id);
    if (it == impl_->active_inferences.end()) {
        return {};  // 请求不存在
    }

    auto& active = it->second;
    auto deadline = std::chrono::steady_clock::now() + timeout;

    // 等待完成
    if (!impl_->cv.wait_until(lock, deadline, [&] {
        return active->completed.load();
    })) {
        return {};  // 超时
    }

    return active->result;
}

void AgentScheduler::cancelRequest(int request_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->active_inferences.find(request_id);
    if (it != impl_->active_inferences.end()) {
        it->second->completed.store(true);
        if (impl_->status_callback) {
            impl_->status_callback(request_id, InferenceState::CANCELLED, {});
        }
    }
}

InferenceState AgentScheduler::queryState(int request_id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->active_inferences.find(request_id);
    if (it == impl_->active_inferences.end()) {
        return InferenceState::IDLE;
    }
    return it->second->completed.load()
        ? InferenceState::COMPLETED
        : InferenceState::DECODING;
}

AgentScheduler::ResourceStatus AgentScheduler::getResourceStatus() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ResourceStatus status{};
    status.pending_requests = static_cast<int>(impl_->request_queue.size());
    status.active_inferences = static_cast<int>(impl_->active_inferences.size());
    status.current_degradation = impl_->current_degradation.load();
    status.thermal_pressure = impl_->thermal_pressure.load();
    status.memory_pressure = impl_->memory_pressure.load();
    return status;
}

void AgentScheduler::forceDegradationLevel(DegradationLevel level) {
    impl_->current_degradation.store(level);
}

void AgentScheduler::setStatusCallback(StatusCallback callback) {
    impl_->status_callback = std::move(callback);
}

// ============================================================
// PowerManager 实现 (简化版)
// ============================================================

void PowerManager::setThermalPolicy(const ThermalPolicy& policy) {
    // TODO: 注册到系统温度监控
    (void)policy;
}

PowerManager::PowerSnapshot PowerManager::getCurrentPower() {
    PowerSnapshot snapshot{};
    // TODO: 从系统 API 读取实际数据
    // Android: BatteryManager + HardwarePropertiesManager
    // iOS: IOKit + ThermalState
    return snapshot;
}

DegradationLevel PowerManager::recommendDegradation() {
    auto snap = getCurrentPower();
    if (snap.thermal_status >= 3) return 3;
    if (snap.thermal_status >= 2) return 2;
    if (snap.thermal_status >= 1) return 1;
    return 0;
}

float PowerManager::energyPerToken(float power_mw, float tokens_per_sec) {
    if (tokens_per_sec <= 0) return 0;
    return power_mw / tokens_per_sec;  // mJ per token
}

} // namespace edgepilot
