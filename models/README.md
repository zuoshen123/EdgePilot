# Models

此目录存放推理模型文件。模型文件不纳入版本控制。

## 推荐模型

| 模型 | 大小 | 量化 | 用途 |
|------|------|------|------|
| TinyLlama 1.1B Chat | ~670MB | Q4_K_M | 初始验证 |
| Phi-2 2.7B | ~1.5GB | Q4_K_M | 效果对比 |
| Llama 3.2 3B | ~1.8GB | Q4_K_M | 主力模型 |

## 下载

```bash
# 自动下载 TinyLlama
bash tools/download_model.sh

# 手动下载 (HuggingFace)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## 推送到设备

```bash
# Android
adb push tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf /sdcard/models/

# iOS (模拟器)
xcrun simctl addmedia booted tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```
