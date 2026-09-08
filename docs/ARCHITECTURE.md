# EdgePilot 架构文档

## 项目结构

```
EdgePilot/
├── core/                          # C++ 核心推理引擎 (跨平台)
│   ├── include/edgepilot/
│   │   ├── common/types.h         # 基础类型定义
│   │   ├── backend/
│   │   │   ├── inference_backend.h # 推理后端抽象接口
│   │   │   └── backend_factory.h  # 后端工厂 (自动检测硬件)
│   │   ├── decoder/
│   │   │   └── speculative_decoder.h # 投机采样解码器
│   │   ├── cache/
│   │   │   └── prompt_cache.h     # Prompt Cache + KV Cache 压缩
│   │   ├── scheduler/
│   │   │   └── agent_scheduler.h  # 多 Agent 并发调度器
│   │   └── metrics/
│   │       └── metrics_collector.h # 性能指标采集器
│   └── src/                       # 实现文件
│       ├── backend/
│       ├── decoder/
│       ├── cache/
│       ├── scheduler/
│       ├── metrics/
│       └── jni/                   # Android JNI 绑定
│
├── android/                       # Android App (Kotlin + Compose)
│   ├── app/src/main/java/com/edgepilot/
│   │   ├── MainActivity.kt
│   │   ├── native/NativeEngine.kt # JNI 桥接
│   │   ├── ui/
│   │   │   ├── theme/Theme.kt
│   │   │   └── screens/
│   │   │       ├── HomeScreen.kt  # 主界面
│   │   │       └── MetricsScreen.kt # 指标面板
│   │   └── viewmodel/
│   │       └── BenchmarkViewModel.kt
│   └── app/build.gradle.kts
│
├── ios/                           # iOS App (Swift + SwiftUI)
│   └── EdgePilot/
│       ├── Bridge/
│       │   ├── EdgePilotBridge.h  # ObjC++ 桥接头
│       │   └── EdgePilotBridge.mm # 桥接实现
│       ├── Views/ContentView.swift
│       ├── ViewModels/BenchmarkViewModel.swift
│       └── EdgePilotApp.swift
│
├── models/                        # 模型文件 (不入版本控制)
├── tools/                         # 工具脚本
│   ├── download_model.sh
│   └── benchmark.py
└── docs/
```

## 数据流

```
用户输入 Prompt
       │
       ▼
┌──────────────┐
│  Scheduler   │ ← 优先级排队 / 热感知降级
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│ Prompt Cache │────▶│ Cache Hit?   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       │ Hit                │ Miss
       ▼                    ▼
  复用 KV Cache      ┌──────────────┐
                     │  Speculative │
                     │   Decoding   │
                     │              │
                     │  Draft → K   │
                     │  Target → ✓✗ │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  KV Cache    │
                     │  INT4 压缩   │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Backend    │
                     │ (ExecuTorch/ │
                     │  ncnn/GGML)  │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Metrics     │
                     │  Collector   │
                     │ (TTFT/ITL/   │
                     │  Power/Temp) │
                     └──────────────┘
```

## 核心设计决策

### 1. 为什么用 C++ 做核心?
- 性能关键路径需要零开销抽象
- 跨 Android/iOS 共享代码
- 与 ExecuTorch/ncnn/GGML 的 C++ API 无缝对接

### 2. 为什么需要 Speculative Decoding?
- TTFT 是用户体验最关键的指标
- 小模型 (1.5B) 生成 + 大模型 (7B) 验证 = 质量不损失，速度提升 40-60%
- 动态窗口调整适应不同 prompt 复杂度

### 3. KV Cache 为什么用 INT4?
- FP16 KV Cache 在 7B 模型 + 4K context 下需要 1GB
- INT4 压缩到 256MB，perplexity 增加 <0.5%
- 配合分页管理，活跃 context 只需 64MB

### 4. 为什么需要热感知调度?
- 手机持续高负载推理会导致温度升高
- 系统降频后性能断崖式下降
- 主动降级比被动降频体验更好
