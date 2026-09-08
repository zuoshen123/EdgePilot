#include "edgepilot/backend/backend_factory.h"
#include "edgepilot/backend/ggml_backend.h"

#include <algorithm>
#include <thread>

#ifdef __ANDROID__
#include <sys/sysinfo.h>
#include <android/log.h>
#define LOG_TAG "EdgePilot"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...)
#define LOGW(...)
#endif

namespace edgepilot {

HardwareInfo BackendFactory::detectHardware() {
    HardwareInfo info{};
    info.soc_name = "ARM CPU";
    info.gpu_name = "Unknown";

#ifdef __ANDROID__
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        info.total_memory_bytes = si.totalram * si.mem_unit;
        info.available_memory_bytes = si.freeram * si.mem_unit;
    }
#elif defined(__APPLE__)
    info.soc_name = "Apple Silicon";
    info.total_memory_bytes = 8ULL * 1024 * 1024 * 1024;
    info.available_memory_bytes = 4ULL * 1024 * 1024 * 1024;
    info.gpu_name = "Apple GPU";
#else
    info.total_memory_bytes = 4ULL * 1024 * 1024 * 1024;
    info.available_memory_bytes = 2ULL * 1024 * 1024 * 1024;
#endif
    return info;
}

std::unique_ptr<InferenceBackend> BackendFactory::createOptimal() {
    LOGI("BackendFactory::createOptimal() — 使用 llama.cpp (GGML) 后端");
    return std::make_unique<GgmlBackend>();
}

std::unique_ptr<InferenceBackend> BackendFactory::create(BackendType type) {
    switch (type) {
        case BackendType::GGML:
            return std::make_unique<GgmlBackend>();
        default:
            LOGW("后端类型 (%d) 不可用，回退 GGML", static_cast<int>(type));
            return std::make_unique<GgmlBackend>();
    }
}

std::unique_ptr<InferenceBackend> BackendFactory::createForDevice(ComputeDevice device) {
    return std::make_unique<GgmlBackend>();
}

std::vector<BackendType> BackendFactory::listAvailable() {
    return { BackendType::GGML };
}

ModelConfig BackendFactory::getRecommendedConfig(const HardwareInfo& hw_info) {
    ModelConfig config{};
    config.backend = BackendType::GGML;
    config.threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 4);
    config.use_mmap = true;
    config.kv_cache_bits = 16;
    config.enable_flash_attn = false;

    size_t ram_mb = hw_info.total_memory_bytes / (1024 * 1024);

    if (ram_mb >= 8192) {
        config.context_length = 4096;
        config.batch_size = 512;
        config.gpu_layers = 99;  // 尝试全部 offload
        LOGI("推荐: 高端 (%zuMB), ctx=4096, GPU=ON", ram_mb);
    } else if (ram_mb >= 6144) {
        config.context_length = 2048;
        config.batch_size = 256;
        config.gpu_layers = 0;
        LOGI("推荐: 中端 (%zuMB), ctx=2048, GPU=OFF", ram_mb);
    } else {
        config.context_length = 1024;
        config.batch_size = 128;
        config.gpu_layers = 0;
        LOGI("推荐: 低端 (%zuMB), ctx=1024, GPU=OFF", ram_mb);
    }
    return config;
}

} // namespace edgepilot
