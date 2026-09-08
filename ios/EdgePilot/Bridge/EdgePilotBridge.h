// EdgePilotBridge.h
// ObjC++ 桥接头文件，连接 Swift UI 与 C++ 核心

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface EdgePilotBridge : NSObject

// 硬件检测
+ (NSString *)detectHardware;

// 引擎生命周期
+ (NSInteger)loadModel:(NSString *)modelPath contextLength:(NSInteger)contextLength;
+ (void)unloadModel;
+ (BOOL)isModelLoaded;

// 推理
+ (NSString *)generate:(NSString *)prompt
             maxTokens:(NSInteger)maxTokens
           temperature:(float)temperature;

// 投机采样
+ (NSInteger)initSpeculativeWithDraft:(NSString *)draftPath
                               target:(NSString *)targetPath
                             window:(NSInteger)window;
+ (float)getAcceptanceRate;

// 性能指标
+ (void)startMetrics;
+ (void)stopMetrics;
+ (NSString *)getMetricsJSON;

@end

NS_ASSUME_NONNULL_END
