package com.edgepilot.harness

/** v0.4 会话桶脚本（spec §5.1）与恢复校验脚本（spec §2.3/V3）。文本定稿，勿改。 */
object SessionScripts {
    data class Script(val bucket: String, val turns: List<String>)

    val SCRIPTS = listOf(
        Script("short", listOf(
            Prompts.SHORT,
            "Now make it one word only.",
            "And what would an astronaut say about it?")),
        Script("mid", listOf(
            Prompts.MID,
            "I still don't get why air resistance disappears on the Moon.",
            "Give me one more everyday example, shorter this time.")),
        Script("long", listOf(
            Prompts.LONG,
            "Now compress your summary to exactly one sentence.",
            "Which problem in the article matters most for phone AI benchmarks?"))
    )

    val RECOVERY = listOf(
        "Remember this session code: ECHO-7. Reply with the code only.",
        "Now state the code again in one word.")
    const val PROBE = "Repeat the code once more, then stop."
}
