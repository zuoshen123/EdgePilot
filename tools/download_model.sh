#!/bin/bash
# 下载 TinyLlama 1.1B 模型 (GGUF 格式, INT4 量化)
# 用于 EdgePilot 项目验证

set -e

MODEL_DIR="$(dirname "$0")/../models"
mkdir -p "$MODEL_DIR"

# TinyLlama 1.1B Chat v1.0 - Q4_K_M (约 670MB)
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "=== EdgePilot Model Downloader ==="
echo "Model: TinyLlama 1.1B Chat (Q4_K_M)"
echo "Size: ~670MB"
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo ""

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Model already exists. Skipping download."
    echo "Delete $MODEL_DIR/$MODEL_FILE to re-download."
    exit 0
fi

echo "Downloading..."
if command -v wget &> /dev/null; then
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
else
    echo "Error: wget or curl not found"
    exit 1
fi

echo ""
echo "Download complete!"
echo "Model path: $MODEL_DIR/$MODEL_FILE"
echo ""
echo "To push to Android device:"
echo "  adb push $MODEL_DIR/$MODEL_FILE /sdcard/models/"
echo ""
echo "To push to iOS (via Finder or xcrun):"
echo "  xcrun simctl addmedia booted $MODEL_DIR/$MODEL_FILE"
