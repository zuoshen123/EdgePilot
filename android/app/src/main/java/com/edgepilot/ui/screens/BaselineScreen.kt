package com.edgepilot.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.edgepilot.harness.BenchmarkSuiteRunner
import com.edgepilot.native.NativeEngine
import com.edgepilot.ui.theme.MetricBad
import com.edgepilot.ui.theme.MetricGood
import com.edgepilot.ui.theme.MetricWarn
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.util.Locale

@Composable
fun BaselineScreen() {
    val context = LocalContext.current
    val runner = remember { BenchmarkSuiteRunner(context.applicationContext) }
    val scope = rememberCoroutineScope()

    var modelPath by remember { mutableStateOf("/data/data/com.edgepilot/files/models/tinyllama.gguf") }
    var maxTokensText by remember { mutableStateOf("128") }
    var threadsSel by remember { mutableStateOf(setOf(2, 4, 8)) }
    var running by remember { mutableStateOf(false) }
    var logs by remember { mutableStateOf(listOf<String>()) }
    var cells by remember { mutableStateOf(listOf<BenchmarkSuiteRunner.CellResult>()) }
    var outDir by remember { mutableStateOf<String?>(null) }
    var badge by remember { mutableStateOf("能力自检中…") }

    LaunchedEffect(Unit) { badge = probeBadge(NativeEngine.samplerProbe()) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("基线矩阵", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Text(
            badge, style = MaterialTheme.typography.bodySmall,
            color = if (badge.contains("⚠")) MetricWarn else MetricGood
        )
        Text(
            "矩阵 = 线程集 × {短/中/长} prompt · 结果即时写入应用外部目录（免权限，adb pull 可达）",
            fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
        )

        OutlinedTextField(
            value = modelPath, onValueChange = { modelPath = it },
            label = { Text("模型路径 (.gguf)") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !running, singleLine = true
        )

        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedTextField(
                value = maxTokensText,
                onValueChange = { v -> maxTokensText = v.filter { it.isDigit() } },
                label = { Text("maxTokens") },
                modifier = Modifier.width(140.dp),
                enabled = !running, singleLine = true
            )
            Column {
                Text("线程集", style = MaterialTheme.typography.bodySmall)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(2, 4, 8).forEach { t ->
                        if (t in threadsSel) {
                            Button(onClick = { threadsSel = threadsSel - t }, enabled = !running) { Text("$t") }
                        } else {
                            OutlinedButton(onClick = { threadsSel = threadsSel + t }, enabled = !running) { Text("$t") }
                        }
                    }
                }
            }
        }

        Button(
            onClick = {
                if (running) return@Button
                val mt = maxTokensText.toIntOrNull() ?: 128
                val ths = threadsSel.sorted()
                running = true
                cells = emptyList()
                outDir = null
                logs = logs + "矩阵开始: threads=$ths maxTokens=$mt"
                scope.launch {
                    try {
                        val outcome = withContext(Dispatchers.Default) {
                            runner.runSuite(modelPath, ths, mt) { msg ->
                                scope.launch(Dispatchers.Main) { logs = logs + msg }
                            }
                        }
                        cells = outcome.cells
                        outDir = outcome.outDir.path
                        logs = logs + "完成，共导出 ${outcome.cells.size} 个 cell"
                    } catch (e: Exception) {
                        logs = logs + "矩阵异常中止: ${e.message}"
                    } finally {
                        running = false
                    }
                }
            },
            enabled = !running && threadsSel.isNotEmpty() && modelPath.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (!running) Icon(Icons.Default.PlayArrow, contentDescription = null)
            Text(if (running) " 矩阵运行中…" else " 跑完整矩阵（${threadsSel.size * 3} 个 cell）")
        }

        if (cells.isNotEmpty()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("单元结果", style = MaterialTheme.typography.titleSmall)
                    cells.forEach { c ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            Text("${c.bucket}·t${c.threads}", fontSize = 12.sp, modifier = Modifier.width(84.dp))
                            Text("${c.totalTokens}tok", fontSize = 12.sp, modifier = Modifier.width(60.dp))
                            Text(
                                if (c.tokensPerJoule.isNaN()) "tok/J —"
                                else String.format(Locale.US, "%.1f tok/J", c.tokensPerJoule),
                                fontSize = 12.sp, modifier = Modifier.width(96.dp)
                            )
                            Text(
                                if (c.tempPeakC.isNaN()) "温 —"
                                else String.format(Locale.US, "%.0f℃", c.tempPeakC),
                                fontSize = 12.sp, modifier = Modifier.width(52.dp)
                            )
                            Text(
                                if (c.pssPeakMB.isNaN()) "存 —"
                                else String.format(Locale.US, "%.0fMB", c.pssPeakMB),
                                fontSize = 12.sp, modifier = Modifier.width(66.dp)
                            )
                            Text(c.status, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                                color = if (c.status == "OK") MetricGood else MetricBad)
                        }
                    }
                }
            }
        }

        if (outDir != null) {
            Text("导出: $outDir", style = MaterialTheme.typography.bodySmall)
            Text("adb pull \"$outDir\" .",
                fontFamily = FontFamily.Monospace, fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
        }

        if (logs.isNotEmpty()) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text("运行日志", style = MaterialTheme.typography.titleSmall)
                    logs.takeLast(14).forEach {
                        Text(it, fontSize = 11.sp, fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f))
                    }
                }
            }
        }
    }
}

private fun probeBadge(json: String): String = try {
    val o = JSONObject(json)
    fun m(k: String, label: String): String = label + when (o.optString(k, "ABSENT")) {
        "OK" -> "✓"
        "PERM_DENIED" -> "⚠ 无权限"
        else -> "⚠ 不可读"
    }
    listOf(m("power", "功耗"), m("thermal", "温度"), m("mem", "内存")).joinToString(" · ")
} catch (e: Exception) {
    "能力自检失败（不伪造：跑后按不可用导出）"
}
