package com.edgepilot.harness

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import com.edgepilot.native.NativeEngine
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** 三桶内置 prompt（spec §③ 定稿文本，逐字，勿改） */
object Prompts {
    const val SHORT = "Explain gravity in one sentence."

    const val MID = "You are a physics tutor. A student asks: \"Why does a feather fall slower " +
        "than a hammer on Earth, but both land together on the Moon?\" Explain the role of air " +
        "resistance and gravity step by step, then give one everyday example that shows the same " +
        "principle, and finish with a short check question for the student to answer."

    const val LONG = "Please summarize the following article in two sentences:\n\n" +
        "Modern smartphone processors are examples of carefully negotiated compromise between " +
        "performance, battery life, and heat. A laptop or desktop chip may draw tens or hundreds " +
        "of watts because a heatsink and fan carry the heat away. A phone has no fan. Its metal " +
        "frame and glass back are the cooling system, and they can only shed so much heat before " +
        "the surface becomes uncomfortable to hold. When the silicon gets too hot, the power " +
        "management firmware lowers the clock frequency of the CPU and GPU cores. This process, " +
        "called thermal throttling, protects the hardware but makes everything slower. Users " +
        "notice it most during long gaming sessions, video recording, and lately, during on-device " +
        "artificial intelligence workloads such as running a large language model locally. These " +
        "workloads stress many cores continuously, which is exactly the pattern that produces heat " +
        "fastest. Manufacturers handle the tension differently. Some allow brief bursts of very " +
        "high speed followed by a long cooldown, while others hold a lower but steadier frequency " +
        "for as long as the workload continues. Benchmark numbers published at launch usually " +
        "measure only the first burst, which is why real-world sustained performance is often " +
        "lower than advertised. Measuring a phone fairly therefore means watching not just speed, " +
        "but how speed changes as the device warms up, how much energy each useful computation " +
        "costs, and whether the machine can keep going without shutting down."

    val BUCKETS: List<Pair<String, String>> =
        listOf("short" to SHORT, "mid" to MID, "long" to LONG)
}

/**
 * 基线矩阵：线程集 × 三桶 prompt，串行逐 cell 跑完并即时导出（spec §③/§④）。
 * 阻塞式调用（generateStream 在 native 路径同步跑完），调用方置于 Dispatchers.Default。
 */
class BenchmarkSuiteRunner(private val context: Context) {

    data class CellResult(
        val idx: Int, val bucket: String, val threads: Int,
        val promptChars: Int, val maxTokens: Int,
        val totalTokens: Int, val ttftMs: Float,
        val itlP50: Float, val itlP90: Float, val itlP99: Float, val tokensPerSec: Float,
        val energyJ: Double, val avgMW: Double, val peakMW: Double, val tokensPerJoule: Double,
        val tempStartC: Double, val tempPeakC: Double, val pssPeakMB: Double,
        val powerSource: String, val thermalSource: String,
        val status: String, val note: String = ""
    )

    data class SuiteOutcome(val outDir: File, val cells: List<CellResult>)

    fun runSuite(
        modelPath: String,
        threadSets: List<Int>,
        maxTokens: Int,
        onLog: (String) -> Unit
    ): SuiteOutcome {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.getExternalFilesDir("bench"), stamp)
        File(dir, "cells").mkdirs()
        val csv = File(dir, "bench.csv")
        csv.writeText(CSV_HEADER + "\n")

        val probe = try { JSONObject(NativeEngine.samplerProbe()) } catch (e: Exception) { JSONObject() }
        val powerSrc = probe.optString("power", "ABSENT")
        val thermalSrc = probe.optString("thermal", "ABSENT")

        val runTs = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
        val device = Build.MODEL
        val soc = socString()
        val battStart = batteryPercent()
        val modelFile = File(modelPath)
        val sha16 = if (modelFile.canRead()) sha256_16(modelFile) else ""

        val cells = ArrayList<CellResult>()
        val totalCells = threadSets.size * Prompts.BUCKETS.size
        var idx = 0
        var aborted = false

        for (th in threadSets) {
            onLog("=== threads=$th reload model ===")
            val okInit = NativeEngine.init(modelPath, th)
            for ((bucket, prompt) in Prompts.BUCKETS) {
                idx++
                val tag = "[$idx/$totalCells] $bucket t$th"
                if (aborted || !okInit) {
                    val note = if (!okInit) "模型加载失败" else "矩阵已中止"
                    finishCell(dir, csv, runTs, device, soc, cells,
                        errorCell(idx, bucket, th, prompt.length, maxTokens, powerSrc, thermalSrc, note),
                        prompt, emptyList(), null, onLog, tag)
                    continue
                }
                NativeEngine.samplerBegin()
                var doneJson: String? = null
                val listener = object : NativeEngine.StreamListener {
                    override fun onToken(piece: String, tokenId: Int) {}
                    override fun onDone(resultJson: String) { doneJson = resultJson }
                }
                val started = NativeEngine.generateStream(prompt, maxTokens, listener)
                val windowJson = NativeEngine.samplerEnd()
                if (!started) {
                    aborted = true   // 引擎不可用：停跑保导出（spec §③）
                    finishCell(dir, csv, runTs, device, soc, cells,
                        errorCell(idx, bucket, th, prompt.length, maxTokens, powerSrc, thermalSrc,
                            "流式推理启动失败（矩阵中止）"),
                        prompt, emptyList(), null, onLog, tag)
                    continue
                }
                val out = NativeEngine.parseGenerateOutput(doneJson ?: "{}")
                val agg = SamplerWindow.parse(windowJson)
                val failed = out.totalTokens == 0 && out.text.startsWith("[错误]")  // N4 联动
                val energyJ = agg.energyMJ / 1000.0
                val tpj = if (out.totalTokens > 0 && agg.nPower >= 2 && energyJ > 0.001)
                    out.totalTokens / energyJ else Double.NaN
                val cell = CellResult(
                    idx, bucket, th, prompt.length, maxTokens,
                    out.totalTokens, out.ttftMs, out.itlP50Ms, out.itlP90Ms, out.itlP99Ms,
                    out.tokensPerSec,
                    if (agg.nPower >= 2) energyJ else Double.NaN,
                    agg.avgMW, agg.peakMW, tpj,
                    agg.tempStartC, agg.tempPeakC, agg.pssPeakMB,
                    powerSrc, thermalSrc,
                    if (failed) "ERROR" else "OK",
                    if (failed) out.text.take(80) else ""
                )
                finishCell(dir, csv, runTs, device, soc, cells, cell,
                    prompt, out.itlSeries, windowJson, onLog, tag)
            }
        }

        writeMeta(File(dir, "meta.json"), runTs, device, soc, modelFile, sha16,
            battStart, batteryPercent(), threadSets, maxTokens, probe, cells.size)
        return SuiteOutcome(dir, cells)
    }

    // ---- 私有辅助 ----

    private fun errorCell(idx: Int, bucket: String, th: Int, chars: Int, maxTok: Int,
                          ps: String, ts: String, note: String) = CellResult(
        idx, bucket, th, chars, maxTok, 0, 0f, 0f, 0f, 0f, 0f,
        Double.NaN, Double.NaN, Double.NaN, Double.NaN,
        Double.NaN, Double.NaN, Double.NaN, ps, ts, "ERROR", note)

    /** 即时落盘（CSV append + cell JSON），崩溃也只丢当前 cell */
    private fun finishCell(dir: File, csv: File, runTs: String, device: String, soc: String,
                           cells: ArrayList<CellResult>, c: CellResult,
                           prompt: String, itl: List<Double>, windowJson: String?,
                           onLog: (String) -> Unit, tag: String) {
        cells.add(c)
        csv.appendText(csvRow(runTs, device, soc, c) + "\n")
        if (windowJson != null) {
            File(dir, "cells/${c.idx}_${c.bucket}_t${c.threads}.json")
                .writeText(cellJson(c, prompt, itl, windowJson))
        }
        onLog("$tag: ${c.totalTokens} tok · ${log(c.energyJ, 2)}J · " +
            "${log(c.tokensPerJoule, 1)} tok/J · 峰温 ${log(c.tempPeakC, 0)}℃ · " +
            "PSS ${log(c.pssPeakMB, 0)}MB ${if (c.status == "OK") "" else "ERROR(${c.note})"}")
    }

    private fun csvRow(runTs: String, device: String, soc: String, c: CellResult): String {
        return listOf(
            csvEsc(runTs), csvEsc(device), csvEsc(soc),
            c.threads.toString(), c.bucket,
            c.promptChars.toString(), c.maxTokens.toString(), c.totalTokens.toString(),
            fmt(c.ttftMs.toDouble(), 1), fmt(c.itlP50.toDouble(), 1),
            fmt(c.itlP90.toDouble(), 1), fmt(c.itlP99.toDouble(), 1),
            fmt(c.tokensPerSec.toDouble(), 2),
            fmt(c.energyJ, 3), fmt(c.avgMW, 1), fmt(c.peakMW, 1), fmt(c.tokensPerJoule, 3),
            fmt(c.tempStartC, 1), fmt(c.tempPeakC, 1), fmt(c.pssPeakMB, 1),
            c.powerSource, c.thermalSource, c.status
        ).joinToString(",")
    }

    private fun cellJson(c: CellResult, prompt: String, itl: List<Double>, windowJson: String): String {
        val o = JSONObject()
        o.put("idx", c.idx).put("bucket", c.bucket).put("threads", c.threads)
        o.put("prompt", prompt).put("prompt_chars", c.promptChars).put("max_tokens", c.maxTokens)
        o.put("status", c.status).put("note", c.note)
        o.put("total_tokens", c.totalTokens)
        o.put("ttft_ms", c.ttftMs.toDouble())
        o.put("itl_p50_ms", c.itlP50.toDouble()).put("itl_p90_ms", c.itlP90.toDouble())
        o.put("itl_p99_ms", c.itlP99.toDouble())
        o.put("tokens_per_sec", c.tokensPerSec.toDouble())
        o.put("itl_series", JSONArray(itl))
        o.put("energy_J", jsonNum(c.energyJ)).put("avg_power_mw", jsonNum(c.avgMW))
        o.put("peak_power_mW", jsonNum(c.peakMW)).put("tokens_per_joule", jsonNum(c.tokensPerJoule))
        o.put("temp_start_c", jsonNum(c.tempStartC)).put("temp_peak_c", jsonNum(c.tempPeakC))
        o.put("pss_peak_mb", jsonNum(c.pssPeakMB))
        o.put("power_source", c.powerSource).put("thermal_source", c.thermalSource)
        o.put("window", try { JSONObject(windowJson) } catch (e: Exception) { windowJson })
        return o.toString(2)
    }

    private fun writeMeta(f: File, runTs: String, device: String, soc: String,
                          modelFile: File, sha16: String, battStart: Int, battEnd: Int,
                          threadSets: List<Int>, maxTokens: Int,
                          probe: JSONObject, cellCount: Int) {
        val appVer = try {
            context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "?"
        } catch (e: Exception) { "?" }
        val o = JSONObject()
        o.put("run_ts", runTs).put("device", device).put("android_soc", soc)
        o.put("app_package", context.packageName).put("app_version", appVer)
        o.put("model_file", modelFile.name)
        o.put("model_bytes", if (modelFile.exists()) modelFile.length() else -1L)
        o.put("model_sha256_16", sha16)
        o.put("battery_start_pct", battStart).put("battery_end_pct", battEnd)
        o.put("capability", probe)
        o.put("threads", JSONArray(threadSets)).put("max_tokens_per_cell", maxTokens)
        o.put("cells", cellCount)
        f.writeText(o.toString(2))
    }

    private fun batteryPercent(): Int = try {
        val i: Intent? = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = i?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = i?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        if (level >= 0 && scale > 0) level * 100 / scale else -1
    } catch (e: Exception) { -1 }

    private fun socString(): String =
        if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}"
        else Build.HARDWARE

    private fun sha256_16(f: File): String = try {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { ins ->
            val buf = ByteArray(65536)
            while (true) {
                val n = ins.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        md.digest().joinToString("") { "%02x".format(it) }.take(16)
    } catch (e: Exception) { "" }

    private fun fmt(v: Double, d: Int): String =
        if (v.isNaN()) "" else String.format(Locale.US, "%.${d}f", v)

    private fun log(v: Double, d: Int): String =
        if (v.isNaN()) "—" else String.format(Locale.US, "%.${d}f", v)

    private fun jsonNum(v: Double): Any = if (v.isNaN()) JSONObject.NULL else v

    private fun csvEsc(s: String): String =
        if (s.any { it == ',' || it == '"' }) "\"" + s.replace("\"", "\"\"") + "\"" else s

    companion object {
        const val CSV_HEADER = "run_ts,device,android_soc,threads,bucket,prompt_chars," +
            "max_tokens,total_tokens,ttft_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms,tokens_per_sec," +
            "energy_J,avg_power_mw,peak_power_mW,tokens_per_joule,temp_start_c,temp_peak_c," +
            "pss_peak_mb,power_source,thermal_source,status"
    }
}
