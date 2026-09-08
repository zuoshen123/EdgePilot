// EdgePilotBridge.mm
// ObjC++ 桥接实现

#import "EdgePilotBridge.h"
#include "edgepilot/backend/backend_factory.h"
#include "edgepilot/decoder/speculative_decoder.h"
#include "edgepilot/cache/prompt_cache.h"
#include "edgepilot/metrics/metrics_collector.h"

using namespace edgepilot;

static std::unique_ptr<InferenceBackend> g_backend;
static std::unique_ptr<SpeculativeDecoder> g_decoder;
static std::unique_ptr<MetricsCollector> g_metrics;

@implementation EdgePilotBridge

+ (NSString *)detectHardware {
    auto hw = BackendFactory::detectHardware();
    NSString *info = [NSString stringWithFormat:
        @"SoC: %s, Memory: %zuMB, GPU: %s, NPU: %s",
        hw.soc_name.c_str(),
        hw.total_memory_bytes / (1024 * 1024),
        hw.has_gpu ? "yes" : "no",
        hw.has_npu ? "yes" : "no"];
    return info;
}

+ (NSInteger)loadModel:(NSString *)modelPath contextLength:(NSInteger)contextLength {
    std::string path = [modelPath UTF8String];

    auto config = BackendFactory::getRecommendedConfig(path);
    config.context_length = static_cast<int>(contextLength);

    g_backend = BackendFactory::createOptimal();
    if (!g_backend) return -1;

    auto status = g_backend->loadModel(config);
    return static_cast<NSInteger>(status);
}

+ (void)unloadModel {
    if (g_backend) {
        g_backend->unload();
        g_backend.reset();
    }
}

+ (BOOL)isModelLoaded {
    return g_backend && g_backend->isLoaded();
}

+ (NSString *)generate:(NSString *)prompt
             maxTokens:(NSInteger)maxTokens
           temperature:(float)temperature {
    if (!g_backend || !g_backend->isLoaded()) {
        return @"ERROR: Engine not initialized";
    }

    GenerateRequest request;
    request.prompt = [prompt UTF8String];
    request.max_new_tokens = static_cast<int>(maxTokens);
    request.temperature = temperature;

    auto result = g_backend->generate(request);
    return [NSString stringWithUTF8String:result.generated_text.c_str()];
}

+ (NSInteger)initSpeculativeWithDraft:(NSString *)draftPath
                               target:(NSString *)targetPath
                               window:(NSInteger)window {
    g_decoder = std::make_unique<SpeculativeDecoder>();

    ModelConfig draft_config;
    draft_config.model_path = [draftPath UTF8String];
    draft_config.backend = BackendType::GGML;

    ModelConfig target_config;
    target_config.model_path = [targetPath UTF8String];
    target_config.backend = BackendType::GGML;

    SpeculativeDecoder::Config spec_config;
    spec_config.speculation_window = static_cast<int>(window);
    spec_config.dynamic_window = true;
    spec_config.min_acceptance_rate = 0.2f;

    auto status = g_decoder->initialize(draft_config, target_config, spec_config);
    return static_cast<NSInteger>(status);
}

+ (float)getAcceptanceRate {
    return g_decoder ? g_decoder->getAverageAcceptanceRate() : 0.0f;
}

+ (void)startMetrics {
    if (!g_metrics) {
        g_metrics = std::make_unique<MetricsCollector>();
        MetricsCollector::Config cfg;
        cfg.sample_interval_ms = 100;
        cfg.enable_sqlite = false;
        cfg.max_history_size = 1000;
        g_metrics->initialize(cfg);
    }
    g_metrics->start();
}

+ (void)stopMetrics {
    if (g_metrics) g_metrics->stop();
}

+ (NSString *)getMetricsJSON {
    if (!g_metrics) return @"{}";
    auto json = g_metrics->exportToJSON();
    return [NSString stringWithUTF8String:json.c_str()];
}

@end
