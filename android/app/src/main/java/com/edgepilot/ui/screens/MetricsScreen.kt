package com.edgepilot.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.edgepilot.ui.theme.*
import com.edgepilot.viewmodel.BenchmarkResult

@Composable
fun MetricsScreen(
    result: BenchmarkResult?,
    metricsJson: String
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Performance Metrics", style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold)

        if (result != null) {
            // TTFT 卡片
            MetricCard(
                title = "TTFT (Time to First Token)",
                value = "${String.format("%.1f", result.ttftMs)} ms",
                color = when {
                    result.ttftMs < 200 -> MetricGood
                    result.ttftMs < 500 -> MetricWarn
                    else -> MetricBad
                },
                description = "Target: < 400ms"
            )

            // ITL 卡片
            MetricCard(
                title = "ITL (Inter-Token Latency)",
                value = "${String.format("%.1f", result.itlAvgMs)} ms avg",
                color = when {
                    result.itlAvgMs < 50 -> MetricGood
                    result.itlAvgMs < 100 -> MetricWarn
                    else -> MetricBad
                },
                description = "P50: ${String.format("%.1f", result.itlP50Ms)} ms · " +
                    "P90: ${String.format("%.1f", result.itlP90Ms)} ms · " +
                    "P99: ${String.format("%.1f", result.itlP99Ms)} ms"
            )

            // Tokens/sec 卡片
            MetricCard(
                title = "Throughput",
                value = "${String.format("%.1f", result.tokensPerSec)} tok/s",
                color = when {
                    result.tokensPerSec > 30 -> MetricGood
                    result.tokensPerSec > 15 -> MetricWarn
                    else -> MetricBad
                },
                description = "Target: > 30 tok/s"
            )

            // 功耗曲线（v0.3 接入真实数据）
            PowerChart()

            // Token 延迟瀑布图（真实 ITL 序列）
            TokenTimeChart(result.ttftMs, result.itlSeries)
        } else {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    "Run a benchmark to see metrics",
                    modifier = Modifier.padding(24.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                )
            }
        }
    }
}

@Composable
private fun MetricCard(
    title: String,
    value: String,
    color: Color,
    description: String
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(title, style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f))
            Text(value, fontSize = 28.sp, fontWeight = FontWeight.Bold, color = color)
            Text(description, style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
        }
    }
}

@Composable
private fun PowerChart(powerSeries: List<Float>? = null) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Power Consumption", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            if (powerSeries == null || powerSeries.size < 2) {
                Spacer(modifier = Modifier.height(8.dp))
                Text("真机功耗数据 · v0.3 接入（需物理设备电量/电流采集）",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
                Spacer(modifier = Modifier.height(8.dp))
                // 空坐标框
                Canvas(modifier = Modifier.fillMaxWidth().height(120.dp)) {
                    val stroke = 1.dp.toPx()
                    drawLine(Color.Gray.copy(alpha = 0.3f),
                        Offset(0f, size.height), Offset(size.width, size.height), stroke)
                    drawLine(Color.Gray.copy(alpha = 0.3f),
                        Offset(0f, 0f), Offset(0f, size.height), stroke)
                }
            } else {
                Spacer(modifier = Modifier.height(8.dp))
                Canvas(modifier = Modifier.fillMaxWidth().height(120.dp)) {
                    val maxVal = powerSeries.max()
                    val stepX = size.width / (powerSeries.size - 1)
                    val path = Path()
                    powerSeries.forEachIndexed { index, value ->
                        val x = index * stepX
                        val y = size.height - (value / maxVal * size.height)
                        if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
                    }
                    drawPath(path, MetricGood, style = Stroke(width = 2.dp.toPx()))
                }
            }
        }
    }
}

@Composable
private fun TokenTimeChart(ttftMs: Float, itlSeries: List<Double>) {
    // 首柱 = TTFT，其余 = 每个 token 的 ITL（真实数据）
    val bars = remember(ttftMs, itlSeries) {
        buildList { add(ttftMs.toDouble()); addAll(itlSeries) }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Token Latency Waterfall", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Text("首柱 = TTFT · 其后每柱 = 单 token ITL（共 ${itlSeries.size} 个）",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            Spacer(modifier = Modifier.height(8.dp))

            if (bars.size < 2 || bars.max() <= 0.0) {
                Text("暂无生成数据",
                    modifier = Modifier.fillMaxWidth().height(100.dp),
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f))
            } else {
                Canvas(modifier = Modifier.fillMaxWidth().height(100.dp)) {
                    val maxVal = bars.max().toFloat()
                    val slot = size.width / bars.size
                    val barWidth = slot * 0.8f
                    val gap = slot * 0.2f

                    bars.forEachIndexed { index, value ->
                        val v = value.toFloat()
                        val x = index * slot
                        val height = (v / maxVal) * size.height
                        val color = if (index == 0) Accent else MetricGood

                        drawRect(
                            color = color,
                            topLeft = Offset(x, size.height - height),
                            size = androidx.compose.ui.geometry.Size(barWidth, height)
                        )
                    }
                }
            }
        }
    }
}
