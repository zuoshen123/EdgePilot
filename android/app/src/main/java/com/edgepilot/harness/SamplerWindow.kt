package com.edgepilot.harness

import org.json.JSONObject

/**
 * MetricsCollector 窗口 JSON（Task 2 形状）解析 + 聚合（spec §③-5）。
 * 能量积分：中点法，首末样本各半窗；energy_mJ = Σ mW · Δt_ms / 1000。
 * NaN = 该通道无样本（导出留空，绝不补 0 —— spec 红线）。
 */
data class WindowAgg(
    val powerMw: List<Float>,
    val avgMW: Double,
    val peakMW: Double,
    val energyMJ: Double,        // 样本 <2 → 0.0（不可积分，非伪造）
    val tempStartC: Double,
    val tempPeakC: Double,
    val pssPeakMB: Double,
    val nPower: Int,
    val nThermal: Int,
    val nMem: Int
)

object SamplerWindow {

    fun parse(json: String): WindowAgg {
        val ts = ArrayList<Double>()
        val mw = ArrayList<Double>()
        val temps = ArrayList<Double>()
        var pssPeak = 0L
        var nMem = 0
        try {
            val o = JSONObject(json)
            val pa = o.optJSONArray("power")
            if (pa != null) for (i in 0 until pa.length()) {
                val s = pa.optJSONArray(i) ?: continue
                if (s.length() < 2) continue
                ts.add(s.getDouble(0)); mw.add(s.getDouble(1))
            }
            val ta = o.optJSONArray("thermal")
            if (ta != null) for (i in 0 until ta.length()) {
                val s = ta.optJSONArray(i) ?: continue
                if (s.length() < 2) continue
                temps.add(s.getDouble(1))
            }
            val ma = o.optJSONArray("mem")
            if (ma != null) for (i in 0 until ma.length()) {
                val s = ma.optJSONArray(i) ?: continue
                if (s.length() < 3) continue
                val pss = s.getLong(1)
                if (pss > pssPeak) pssPeak = pss
                nMem++
            }
        } catch (e: Exception) { /* 解析失败 = 无数据 */ }

        var energyMj = 0.0
        for (i in ts.indices) {
            val dtMs = when {
                ts.size == 1 -> 0.0
                i == 0 -> (ts[1] - ts[0]) / 2.0
                i == ts.size - 1 -> (ts[i] - ts[i - 1]) / 2.0
                else -> (ts[i + 1] - ts[i - 1]) / 2.0
            }
            if (dtMs > 0) energyMj += mw[i] * dtMs / 1000.0
        }

        return WindowAgg(
            powerMw = mw.map { it.toFloat() },
            avgMW = if (mw.isNotEmpty()) mw.average() else Double.NaN,
            peakMW = if (mw.isNotEmpty()) mw.max() else Double.NaN,
            energyMJ = energyMj,
            tempStartC = if (temps.isNotEmpty()) temps.first() else Double.NaN,
            tempPeakC = if (temps.isNotEmpty()) temps.max() else Double.NaN,
            pssPeakMB = if (nMem > 0) pssPeak / 1024.0 else Double.NaN,
            nPower = ts.size, nThermal = temps.size, nMem = nMem
        )
    }
}
