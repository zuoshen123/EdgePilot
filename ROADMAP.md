# EdgePilot 版本迭代规划

> 方法论贯穿所有版本：**先建立可信测量，再做单点优化，用指标验证收益**。

```
v0.1 链路打通 → v0.2 数据可信 → v0.3 真机基线 → v0.4/0.5 优化出收益 → v1.0 可交付
```

## 当前状态（v0.1，已完成）

- ✅ llama.cpp 静态链接进 Android，JNI 桥接层打通
- ✅ 完整推理链路：模型加载 → tokenize → prefill → 采样循环 → KV cache 续写
- ✅ 核心指标采集：TTFT / ITL / 吞吐量，经 JSON 回传至 UI
- ✅ x86_64 模拟器端到端验证通过（~0.7–0.9 tok/s）

---

## v0.2 — 数据可信（偿还技术债）

| 事项 | 现状 | 目标 |
|---|---|---|
| 调试日志 | 推理热路径中残留临时日志 | 收敛为可开关的编译期日志宏 |
| 流式输出 | `generate()` 同步一次性返回 | JNI 回调逐 token 推至 Kotlin，UI 打字机效果 |
| 指标图表 | 功耗曲线 / token-时间图为预置数据 | 接真实 `BenchmarkResult` 序列渲染 |
| ITL P99 | 均值 × 1.2 占位估算 | C++ 侧记录逐 token 时间戳，真实计算 P50/P90/P99 |
| 接受率指标 | 返回固定值 | 无实测前标注"待测"，不展示假数据 |

**验收标准**：指标 Tab 每个数字均可追溯到一次真实推理，UI 无预置数据。

---

## v0.3 — 真机基线 + 功耗/温度采集

当前性能数据来自 x86 模拟器转译，需在真实设备上重建基线：

- **真机测试**：arm64-v8a 物理设备（至少 1–2 款不同 SoC），产出 TTFT / ITL / 吞吐量真实基线
- **功耗采集**：`BatteryManager` / `dumpsys battery` 读取电流电压 → 计算 **tokens/Joule**（每焦耳生成 token 数）
- **热节流监控**：读取 `/sys/class/thermal/` 温度，记录连续推理的性能衰减曲线
- **内存剖析**：推理全程 RSS/PSS 采样，拆解权重 / KV cache / 激活各自占比
- **自动化 Benchmark Harness**：预置 prompt 集（短 / 中 / 长上下文），一键跑完整套件导出 CSV/JSON

**验收标准**：产出真机性能报告（prompt 长度 × 线程数 的 TTFT/ITL/功耗矩阵）。

---

## v0.4 — KV Cache 与 Prompt Cache 优化

将 `inference_backend.h` 中预留的缓存接口从声明变为实现：

- **Session 化多轮对话**：KV cache 跨轮复用，仅 prefill 增量部分，量化对比有/无 prompt cache 的 TTFT
- **KV cache 持久化**：`exportKVCache` / `importKVCache` 落地，支持会话热恢复
- **KV 量化**：cache F16 → Q8/Q4（`compressKVCache`），长上下文内存占用减半
- **配置推荐器**：基于硬件检测数据，给出不同 RAM 设备下的 n_ctx / 线程数 / 量化方案建议

**验收标准**：多轮对话 TTFT 降低 ≥ 70%；4K 上下文内存占用降低 ≥ 40%。

---

## v0.5 — 推测解码（Speculative Decoding）

- 双模型架构：小模型起草 N 个 token，主模型单次 decode 并行验证
- `acceptanceRate` 接入真实统计（当前为占位字段）
- 调参：草稿长度、采样策略对接受率的影响
- 可视化：逐 token 时间戳图，展示推测解码前后的阶梯状加速模式

**验收标准**：decode 阶段端到端提速 ≥ 1.5×，acceptance_rate ≥ 0.6。

---

## v1.0 — 后端扩展 + 工程化发布

- **GPU 后端**：llama.cpp OpenCL / Vulkan 编译开关（`GGML_OPENCL=ON`），同设备 CPU vs GPU 性能矩阵
- **NPU 通路调研**：高通 QNN / NNAPI 集成可行性分析报告
- **模型管理**：App 内下载 + 断点续传 + 完整性校验，免 adb 手动推送
- **配置面板**：n_ctx / n_threads / 量化方案 / 采样参数 UI 可调，即改即测
- **工程化**：CI（GitHub Actions 构建 + 静态检查）、纯逻辑层单元测试、Release APK、版本 tag、CHANGELOG

**验收标准**：按 README 指引 30 分钟内跑通全流程；CI 绿色。

---

## 版本总览

| 版本 | 主题 | 周期 | 核心产出 |
|---|---|---|---|
| v0.1 | 链路打通 | ✅ 已完成 | 端到端推理 + 基础指标 |
| v0.2 | 数据可信 | ~1 周 | 真实逐 token 指标、流式输出 |
| v0.3 | 真机基线 | ~2 周 | 真机性能/功耗/温度报告 |
| v0.4 | KV Cache 优化 | ~2–3 周 | TTFT ≥70%↓、内存 ≥40%↓ |
| v0.5 | 推测解码 | ~3 周 | decode 提速 ≥1.5× |
| v1.0 | 后端 + 工程化 | ~3–4 周 | GPU/NPU、CI、可交付发布 |
