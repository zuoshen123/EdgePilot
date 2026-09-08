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
                description = "P99: ${String.format("%.1f", result.itlP99Ms)} ms"
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

            // 模拟功耗曲线
            PowerChart()

            // 模拟 Token 时间分布
            TokenTimeChart()
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
private fun PowerChart() {
    // 模拟功耗数据
    val data = listOf(1200f, 1800f, 2400f, 2100f, 1900f, 2200f, 1600f, 1400f,
                      2000f, 2500f, 2300f, 1800f, 1500f, 1700f, 2100f, 1900f)

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Power Consumption (mW)", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            Canvas(modifier = Modifier.fillMaxWidth().height(120.dp)) {
                val maxVal = data.max()
                val stepX = size.width / (data.size - 1)

                val path = Path()
                data.forEachIndexed { index, value ->
                    val x = index * stepX
                    val y = size.height - (value / maxVal * size.height)
                    if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
                }

                drawPath(path, MetricGood, style = Stroke(width = 2.dp.toPx()))

                // 基准线
                val baselineY = size.height - (2000f / maxVal * size.height)
                drawLine(
                    MetricWarn,
                    Offset(0f, baselineY),
                    Offset(size.width, baselineY),
                    strokeWidth = 1.dp.toPx()
                )
            }
            Text("Baseline: 2000mW", fontSize = 11.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
        }
    }
}

@Composable
private fun TokenTimeChart() {
    // 模拟 token 时间分布 (瀑布图)
    val tokenTimes = listOf(350f, 45f, 52f, 48f, 55f, 42f, 60f, 47f, 51f, 44f,
                            53f, 49f, 46f, 58f, 43f, 50f)

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Token Latency Waterfall", style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold)
            Text("First bar = TTFT, rest = ITL", fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f))
            Spacer(modifier = Modifier.height(8.dp))

            Canvas(modifier = Modifier.fillMaxWidth().height(100.dp)) {
                val maxVal = tokenTimes.max()
                val barWidth = size.width / tokenTimes.size * 0.8f
                val gap = size.width / tokenTimes.size * 0.2f

                tokenTimes.forEachIndexed { index, value ->
                    val x = index * (barWidth + gap)
                    val height = (value / maxVal) * size.height
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
