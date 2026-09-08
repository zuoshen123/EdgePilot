package com.edgepilot.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.edgepilot.ui.theme.*
import com.edgepilot.viewmodel.BenchmarkUiState

@Composable
fun HomeScreen(
    uiState: BenchmarkUiState,
    onDetectHardware: () -> Unit,
    onLoadModel: (String) -> Unit,
    onRunBenchmark: (String) -> Unit
) {
    var promptText by remember { mutableStateOf("Explain quantum computing in simple terms") }
    var modelPath by remember { mutableStateOf("/data/data/com.edgepilot/files/models/tinyllama.gguf") }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 硬件信息卡片
        item {
            HardwareCard(uiState, onDetectHardware)
        }

        // 模型加载
        item {
            ModelLoadCard(
                modelPath = modelPath,
                onModelPathChange = { modelPath = it },
                onLoad = { onLoadModel(modelPath) },
                isLoaded = uiState.modelLoaded,
                isLoading = uiState.isRunning
            )
        }

        // Benchmark 输入
        item {
            BenchmarkInputCard(
                prompt = promptText,
                onPromptChange = { promptText = it },
                onRun = { onRunBenchmark(promptText) },
                isRunning = uiState.isRunning,
                enabled = uiState.modelLoaded
            )
        }

        // 结果
        uiState.result?.let { result ->
            item {
                ResultsCard(result)
            }
        }

        // 生成的文本
        if (uiState.generatedText.isNotEmpty()) {
            item {
                GeneratedTextCard(uiState.generatedText)
            }
        }

        // 错误
        uiState.error?.let { error ->
            item {
                ErrorCard(error)
            }
        }

        // 日志
        if (uiState.logs.isNotEmpty()) {
            item {
                LogsCard(uiState.logs)
            }
        }
    }
}

@Composable
private fun HardwareCard(uiState: BenchmarkUiState, onDetectHardware: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("硬件信息", style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold)
                FilledTonalButton(onClick = onDetectHardware) {
                    Icon(Icons.Default.Refresh, contentDescription = null,
                        modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("检测硬件")
                }
            }
            Spacer(modifier = Modifier.height(8.dp))

            val hw = uiState.hardwareInfo
            if (hw.isNotEmpty()) {
                Text(hw, style = MaterialTheme.typography.bodySmall)
            } else {
                Text("点击上方按钮检测硬件",
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            }
        }
    }
}

@Composable
private fun ModelLoadCard(
    modelPath: String,
    onModelPathChange: (String) -> Unit,
    onLoad: () -> Unit,
    isLoaded: Boolean,
    isLoading: Boolean
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("模型加载", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            OutlinedTextField(
                value = modelPath,
                onValueChange = onModelPathChange,
                label = { Text("模型路径 (.gguf)") },
                placeholder = { Text("/sdcard/models/tinyllama.gguf") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            Spacer(modifier = Modifier.height(8.dp))

            Button(
                onClick = onLoad,
                enabled = !isLoading && modelPath.isNotBlank(),
                modifier = Modifier.fillMaxWidth()
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Color.White
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(if (isLoaded) "✓ 模型已加载" else "加载模型")
            }
        }
    }
}

@Composable
private fun BenchmarkInputCard(
    prompt: String,
    onPromptChange: (String) -> Unit,
    onRun: () -> Unit,
    isRunning: Boolean,
    enabled: Boolean
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("推理测试", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            if (!enabled) {
                Text("请先加载模型",
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            }

            OutlinedTextField(
                value = prompt,
                onValueChange = onPromptChange,
                label = { Text("输入 Prompt") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 2,
                enabled = enabled
            )

            Spacer(modifier = Modifier.height(8.dp))

            Button(
                onClick = onRun,
                enabled = enabled && !isRunning && prompt.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Secondary)
            ) {
                if (isRunning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = Color.White
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Icon(Icons.Default.PlayArrow, contentDescription = null)
                Spacer(modifier = Modifier.width(4.dp))
                Text(if (isRunning) "推理中..." else "开始推理")
            }
        }
    }
}

@Composable
private fun ResultsCard(result: com.edgepilot.viewmodel.BenchmarkResult) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("结果", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            ResultRow("总耗时", "${result.totalTimeMs.toInt()} ms")
            ResultRow("Token 数", "${result.totalTokens}")
            ResultRow("吞吐量", "${result.tokensPerSec.toInt()} tokens/sec")
            if (result.acceptanceRate > 0) {
                ResultRow("接受率", "${(result.acceptanceRate * 100).toInt()}%")
            }
        }
    }
}

@Composable
private fun ResultRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun GeneratedTextCard(text: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("生成结果", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            Text(text, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun ErrorCard(error: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Row(modifier = Modifier.padding(16.dp)) {
            Icon(Icons.Default.Warning, contentDescription = null,
                tint = MaterialTheme.colorScheme.error)
            Spacer(modifier = Modifier.width(8.dp))
            Text(error, color = MaterialTheme.colorScheme.onErrorContainer)
        }
    }
}

@Composable
private fun LogsCard(logs: List<String>) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("日志", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))
            logs.takeLast(5).forEach { log ->
                Text(log, style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
            }
        }
    }
}
