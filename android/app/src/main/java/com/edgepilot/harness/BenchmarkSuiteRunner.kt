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

    /** v0.4 会话桶单元（spec §5.1：cache off=全量重放基线，on=KV 跨轮复用） */
    data class SessionCell(val idx: Int, val bucket: String, val turn: Int, val cacheMode: String,
        val threads: Int, val submittedChars: Int, val maxTokens: Int, val totalTokens: Int,
        val ttftMs: Float, val itlP99: Float, val tokensPerSec: Float,
        val energyJ: Double, val pssPeakMB: Double, val cacheHit: Double,
        val status: String, val note: String = "")

    data class SuiteOutcome2(val outDir: File, val cells: List<SessionCell>)

    /** v0.4 量化对比单元（spec §5.2，V2/V5） */
    data class QuantCell(val bits: Int, val nCtx: Int, val prefillTarget: Int, val totalTokens: Int,
        val ttftMs: Float, val usedBytes: Long, val totalBytes: Long,
        val energyJ: Double, val pssPeakMB: Double, val status: String, val note: String = "")

    data class SuiteOutcome3(val outDir: File, val cells: List<QuantCell>)

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

    /**
     * v0.4 会话矩阵（spec §5.1，V1 测量）：{off,on} × 3 桶 × 3 轮 = 18 单元。
     * off 先跑=纯 v0.3 语义全量重放（hist+新轮做单 prompt、重装 KV），on 后跑=续写复用。
     * 阻塞式（同 runSuite），调用方置于 Dispatchers.Default。
     */
    fun runSessionSuite(modelPath: String, threads: Int, maxTokens: Int,
                        onLog: (String) -> Unit): SuiteOutcome2 {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.getExternalFilesDir("session"), stamp); File(dir, "cells").mkdirs()
        val csv = File(dir, "session.csv"); csv.writeText(SESSION_CSV_HEADER + "\n")
        val runTs = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
        val device = Build.MODEL; val soc = socString()
        val cells = ArrayList<SessionCell>(); var idx = 0; var aborted = false
        val okInit = NativeEngine.init(modelPath, threads)
        // 不伪造守卫（T6 评审 Important 收账）：Mock 的 generateStream 立即返 true 而 onDone ~560ms 后
        // 才达，本 runner 不等待 → doneJson 恒 null → 18 行"OK"零值假 CSV。kvInfo()=="{}" 即 Mock 特征。
        val mock = okInit && NativeEngine.kvInfo() == "{}"
        val totalCells = 2 * SessionScripts.SCRIPTS.size * 3
        for (mode in listOf("off", "on")) {
            for (s in SessionScripts.SCRIPTS) {
                val hist = StringBuilder()
                s.turns.forEachIndexed { i, text ->
                    idx++
                    val tag = "[$idx/$totalCells] ${s.bucket} ${mode} R${i + 1}"
                    val prompt = if (mode == "off") hist.toString() + text else text
                    val cont = mode == "on" && i > 0
                    if (aborted || !okInit || mock) {
                        val note = if (mock) "Mock 环境（不模拟会话）" else if (!okInit) "模型加载失败" else "套件已中止"
                        val c = SessionCell(idx, s.bucket, i + 1, mode, threads, prompt.length, maxTokens,
                            0, 0f, 0f, 0f, Double.NaN, Double.NaN, if (cont) 1.0 else 0.0, "ERROR", note)
                        cells.add(c); csv.appendText(sessionRow(runTs, device, soc, c) + "\n"); onLog("$tag: ERROR($note)"); return@forEachIndexed
                    }
                    NativeEngine.samplerBegin()
                    var doneJson: String? = null
                    val started = NativeEngine.generateStream(prompt, maxTokens,
                        object : NativeEngine.StreamListener {
                            override fun onToken(piece: String, tokenId: Int) {}
                            override fun onDone(resultJson: String) { doneJson = resultJson }
                        }, cont)
                    val agg = SamplerWindow.parse(NativeEngine.samplerEnd())
                    if (!started) {
                        aborted = true   // v0.3 同款：停跑保导出
                        onLog("$tag: ERROR(启动失败·中止)"); cells.add(SessionCell(idx, s.bucket, i + 1, mode, threads,
                            prompt.length, maxTokens, 0, 0f, 0f, 0f, Double.NaN, Double.NaN, 0.0, "ERROR", "启动失败(中止)"))
                        csv.appendText(sessionRow(runTs, device, soc, cells.last()) + "\n")
                        hist.append(text).append("\n\n"); return@forEachIndexed
                    }
                    val out = NativeEngine.parseGenerateOutput(doneJson ?: "{}")
                    val failed = out.totalTokens == 0 && out.text.startsWith("[错误]")
                    val energyJ = agg.energyMJ / 1000.0
                    val c = SessionCell(idx, s.bucket, i + 1, mode, threads, prompt.length, maxTokens,
                        out.totalTokens, out.ttftMs, out.itlP99Ms, out.tokensPerSec,
                        if (agg.nPower >= 2) energyJ else Double.NaN, agg.pssPeakMB,
                        if (mode == "on") (if (i > 0) 1.0 else 0.0) else 0.0,   // off=全量重放，无命中（诚实）
                        if (failed) "ERROR" else "OK", if (failed) out.text.take(80) else "")
                    cells.add(c); csv.appendText(sessionRow(runTs, device, soc, c) + "\n")
                    File(dir, "cells/session_${s.bucket}_${mode}_R${i + 1}.json")
                        .writeText(sessionCellJson(c, prompt, out.itlSeries))
                    onLog("$tag: TTFT ${c.ttftMs.toInt()}ms · ${c.totalTokens}tok ${if (c.status == "OK") "" else "ERROR"}")
                    hist.append(text).append("\n").append(out.text).append("\n\n")
                }
            }
        }
        writeMeta(File(dir, "meta.json"), runTs, device, soc, File(modelPath),
            if (File(modelPath).canRead()) sha256_16(File(modelPath)) else "",
            batteryPercent(), batteryPercent(), listOf(threads), maxTokens,
            try { JSONObject(NativeEngine.samplerProbe()) } catch (e: Exception) { JSONObject() }, cells.size)
        return SuiteOutcome2(dir, cells)
    }

    /** spec §5.2：同一长 prompt 三档重建 ctx → KV 字节/PSS/贪心 token。串行、每档全新 init。 */
    fun runQuantCompare(modelPath: String, onLog: (String) -> Unit): SuiteOutcome3 {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = File(context.getExternalFilesDir("quant"), stamp)
        val csv = File(dir, "quant.csv"); csv.writeText(QUANT_CSV_HEADER + "\n")
        val runTs = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
        val device = Build.MODEL; val soc = socString()
        val cells = ArrayList<QuantCell>()
        // 不伪造守卫（T6 同款，评审裁带入）：Mock 下贪心文本恒空 → 三档假 OK + 空 V5 文件
        val mock = NativeEngine.kvInfo() == "{}"
        val long6 = (Prompts.LONG + "\n\n").repeat(6)   // ~3.6K token（4096 档内；降档设备由重试缩量）
        for (bits in listOf(16, 8, 4)) {
            if (mock) {
                cells.add(QuantCell(bits, 0, 0, 0, 0f, 0, 0, Double.NaN, Double.NaN, "ERROR", "Mock 环境（无真实推理）"))
                csv.appendText(quantRow(runTs, device, soc, cells.last()) + "\n"); continue
            }
            val okInit = NativeEngine.init(modelPath, 0, bits)
            if (!okInit) {
                cells.add(QuantCell(bits, 0, 0, 0, 0f, 0, 0, Double.NaN, Double.NaN, "ERROR", "加载失败"))
                csv.appendText(quantRow(runTs, device, soc, cells.last()) + "\n"); continue
            }
            val long3 = (Prompts.LONG + "\n\n").repeat(3)
            var text = long6; var retried = false; var failed = true
            var doneJson: String? = null; var started = false
            var agg = SamplerWindow.parse("{}")
            var out = NativeEngine.parseGenerateOutput("{}")
            while (true) {
                NativeEngine.samplerBegin()
                doneJson = null
                started = NativeEngine.generateStream(text, 32,
                    object : NativeEngine.StreamListener {
                        override fun onToken(piece: String, tokenId: Int) {}
                        override fun onDone(resultJson: String) { doneJson = resultJson }
                    }, false, 0f)   // 贪心 temp=0：V5 跨档可比
                agg = SamplerWindow.parse(NativeEngine.samplerEnd())
                out = NativeEngine.parseGenerateOutput(doneJson ?: "{}")
                failed = !started || (out.totalTokens == 0 && out.text.startsWith("[错误]"))
                if (!failed || retried) break
                text = long3; retried = true   // 如实缩：预算降档设备半量重试一次（note 记录），再败即 ERROR
            }
            val kv = try { JSONObject(NativeEngine.kvInfo()) } catch (e: Exception) { JSONObject() }
            val c = QuantCell(bits, kv.optInt("n_ctx"), text.length / 4 /*目标tok近似*/, out.totalTokens,
                out.ttftMs, kv.optLong("used_bytes"), kv.optLong("total_bytes"),
                if (agg.nPower >= 2) agg.energyMJ / 1000.0 else Double.NaN, agg.pssPeakMB,
                if (failed) "ERROR" else "OK",
                if (failed) out.text.take(80) else if (retried) "半量重试(3×LONG)" else "6×LONG")
            cells.add(c); csv.appendText(quantRow(runTs, device, soc, c) + "\n")
            if (!failed) File(dir, "quant_tokens_$bits.txt").writeText(out.text)   // V5 源：成功才有文件（不伪造）
            onLog("bits=$bits: KV ${(c.usedBytes shr 20)}MB/${(c.totalBytes shr 20)}MB · PSS ${if (c.pssPeakMB.isNaN()) "—" else c.pssPeakMB.toInt()}MB ${if (failed) "ERROR" else ""}")
        }
        NativeEngine.release()
        return SuiteOutcome3(dir, cells)
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

    /** 会话行：列序与 SESSION_CSV_HEADER 逐列对应；NaN→空（不伪造） */
    private fun sessionRow(runTs: String, device: String, soc: String, c: SessionCell): String {
        return listOf(
            csvEsc(runTs), csvEsc(device), csvEsc(soc),
            c.threads.toString(), csvEsc(c.bucket), c.turn.toString(), c.cacheMode,
            c.submittedChars.toString(), c.maxTokens.toString(), c.totalTokens.toString(),
            fmt(c.ttftMs.toDouble(), 1), fmt(c.itlP99.toDouble(), 1), fmt(c.tokensPerSec.toDouble(), 2),
            fmt(c.energyJ, 3), fmt(c.pssPeakMB, 1), fmt(c.cacheHit, 1), c.status
        ).joinToString(",")
    }

    /** 量化行：列序与 QUANT_CSV_HEADER 逐列对应；threads 恒 0=推荐（runbook 注记）；retried 由 note 互证 */
    private fun quantRow(runTs: String, device: String, soc: String, c: QuantCell): String {
        return listOf(
            csvEsc(runTs), csvEsc(device), csvEsc(soc),
            c.bits.toString(), "0", c.nCtx.toString(),
            c.prefillTarget.toString(), c.totalTokens.toString(),
            fmt(c.ttftMs.toDouble(), 1), c.usedBytes.toString(), c.totalBytes.toString(),
            fmt(c.energyJ, 3), fmt(c.pssPeakMB, 1),
            if (c.note.startsWith("半量重试")) "1" else "0",
            c.status, csvEsc(c.note)
        ).joinToString(",")
    }

    private fun sessionCellJson(c: SessionCell, prompt: String, itl: List<Double>): String {
        val o = JSONObject()
        o.put("idx", c.idx).put("bucket", c.bucket).put("turn", c.turn)
        o.put("cache_mode", c.cacheMode).put("threads", c.threads)
        o.put("prompt", prompt).put("submitted_chars", c.submittedChars).put("max_tokens", c.maxTokens)
        o.put("status", c.status).put("note", c.note)
        o.put("total_tokens", c.totalTokens)
        o.put("ttft_ms", c.ttftMs.toDouble()).put("itl_p99_ms", c.itlP99.toDouble())
        o.put("tokens_per_sec", c.tokensPerSec.toDouble())
        o.put("itl_series", JSONArray(itl))
        o.put("energy_J", jsonNum(c.energyJ)).put("pss_peak_mb", jsonNum(c.pssPeakMB))
        o.put("cache_hit", jsonNum(c.cacheHit))
        return o.toString(2)
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

        /** v0.4 会话桶 17 列（v0.3 23 列 bench.csv 冻结不动，spec §5.1） */
        const val SESSION_CSV_HEADER = "run_ts,device,android_soc,threads,bucket,turn,cache_mode," +
            "submitted_chars,max_tokens,total_tokens,ttft_ms,itl_p99_ms,tokens_per_sec," +
            "energy_J,pss_peak_mb,cache_hit,status"

        /** v0.4 量化对比 16 列（spec §5.2） */
        const val QUANT_CSV_HEADER = "run_ts,device,android_soc,kv_bits,threads,n_ctx," +
            "prefill_target_chars,total_tokens,ttft_ms,used_bytes,total_bytes," +
            "energy_J,pss_peak_mb,retried,status,note"
    }
}
