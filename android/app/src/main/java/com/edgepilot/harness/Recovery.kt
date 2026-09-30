package com.edgepilot.harness

import android.content.Context
import com.edgepilot.native.GenerateOutput
import com.edgepilot.native.NativeEngine
import kotlinx.coroutines.CompletableDeferred
import java.io.File

/**
 * v0.4 §2.3 跨进程恢复通道（无 UI，adb `am start --es ep_session_save/-load <name>` 驱动）。
 * 不伪造：任何校验段失败 = REJECTED+原因；比对不符 = MISMATCH 双文本留档，不重试不掩盖。
 */
object Recovery {
    const val MODEL = "/data/data/com.edgepilot/files/models/tinyllama.gguf"  // 与 UI 默认路径同值（runbook 注记两处同改）

    suspend fun run(ctx: Context, baseName: String, export: Boolean, log: (String) -> Unit): String {
        val dir = ctx.getExternalFilesDir("recovery")!!.also { it.mkdirs() }
        val base = File(dir, baseName).absolutePath
        val res = try { if (export) exportFlow(base, dir, log) else loadFlow(base, dir, log) }
                  catch (e: Exception) { "REJECTED: 异常 ${e.message}" }
        finally { NativeEngine.release() }
        File(dir, if (export) "session_export_result.txt" else "session_recover_result.txt").writeText(res + "\n")
        log("Recovery[$export] $res")
        return res
    }

    private suspend fun turn(prompt: String, cont: Boolean): GenerateOutput {
        val done = CompletableDeferred<String?>()
        val ok = NativeEngine.generateStream(prompt, 32,
            object : NativeEngine.StreamListener {
                override fun onToken(piece: String, tokenId: Int) {}
                override fun onDone(resultJson: String) { if (!done.isCompleted) done.complete(resultJson) }
            }, cont, 0f)   // 贪心（§2.3 贪心前置，T1 生效）
        if (!ok) return NativeEngine.parseGenerateOutput("{}")
        return NativeEngine.parseGenerateOutput(done.await() ?: "{}")
    }
    private fun bad(o: GenerateOutput) = o.totalTokens == 0 && o.text.startsWith("[错误]")

    private suspend fun exportFlow(base: String, dir: File, log: (String) -> Unit): String {
        if (!NativeEngine.init(MODEL)) return "REJECTED: 模型加载失败"
        val t1 = turn(SessionScripts.RECOVERY[0], false); if (bad(t1)) return "REJECTED: 首轮失败"
        val t2 = turn(SessionScripts.RECOVERY[1], true);  if (bad(t2)) return "REJECTED: 次轮失败"
        if (!NativeEngine.sessionSave(base)) return "FAILED: sessionSave（原因见 logcat EdgePilot）"   // 先存档（t2 末态，裁定 P-save）
        val probe = turn(SessionScripts.PROBE, true);     if (bad(probe)) return "FAILED: probe 轮失败"
        File(dir, "replay_baseline.txt").writeText(probe.text)
        log("probe ${probe.totalTokens} tok 已留基线")
        return "EXPORTED: $base"
    }

    private suspend fun loadFlow(base: String, dir: File, log: (String) -> Unit): String {
        if (!NativeEngine.init(MODEL)) return "REJECTED: 模型加载失败"
        if (!NativeEngine.sessionLoad(base)) return "REJECTED: 存档校验失败（原因见 logcat EdgePilot）"
        val bl = File(dir, "replay_baseline.txt")
        if (!bl.canRead()) return "REJECTED: replay_baseline.txt 缺失"
        val probe = turn(SessionScripts.PROBE, true)      // 导入态 n_past>0 → 续写
        if (bad(probe)) return "FAILED: probe 轮失败"
        val expected = bl.readText()
        return if (probe.text == expected) { log("恢复一致：${probe.totalTokens} tok 逐字相同"); "PASS: 贪心 ${probe.totalTokens} tokens 逐字一致" }
        else { File(dir, "mismatch_actual.txt").writeText(probe.text); File(dir, "mismatch_expected.txt").writeText(expected)
               "MISMATCH: 实际≠期望（全长留档 mismatch_*.txt，不重试）" }
    }
}
