#!/usr/bin/env bash
# v0.3 校验脚本 — adb 侧电流×电压积分，对照 app 导出的 avg_power_mw / energy_J（spec §⑥）
# 用法: bash scripts/xcheck_power.sh <秒数> [adb路径]      例: ... 120 "$LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe"
set -u
SECS="${1:-60}"
ADB="${2:-adb}"

case "$SECS" in (''|0|0[0-9]*|*[!0-9]*) echo "秒数须为正整数，得到: '$SECS'"; exit 2;; esac
command -v "$ADB" >/dev/null 2>&1 || { echo "找不到 adb: $ADB（SDK platform-tools 入 PATH 或传第 2 参数）"; exit 2; }
"$ADB" get-state >/dev/null 2>&1 || { echo "无设备连接。连真机并允许 USB 调试后重试。"; exit 2; }

# 枚举 Battery supply（与 app 侧 PlatformSampler 同法；多 Battery supply 设备的首个命中/tie-break 可与 app 不同）
SUPPLY=$("$ADB" shell 'for d in /sys/class/power_supply/*; do [ "$(cat $d/type 2>/dev/null)" = Battery ] && basename $d && break; done' | tr -d '\r' | head -1)
[ -z "$SUPPLY" ] && SUPPLY=battery   # 兜底常见名（读不到会在下面归入 PERM）
CUR="/sys/class/power_supply/$SUPPLY/current_now"
VOL="/sys/class/power_supply/$SUPPLY/voltage_now"
echo "== supply=$SUPPLY 采样 ${SECS}s @5Hz（设备端单次循环，主机中点积分）=="

"$ADB" shell "for i in \$(seq 1 $((SECS * 5))); do c=\$(cat $CUR 2>/dev/null); v=\$(cat $VOL 2>/dev/null); echo \"\$c \$v\"; sleep 0.2; done" \
  | tr -d '\r' \
  | awk -v dt=0.2 '
      $1 ~ /^-?[0-9]+$/ && $2 ~ /^[0-9]+$/ {
        cur = ($1 + 0 < 0 ? -($1 + 0) : $1 + 0) * ($2 + 0) / 1e9   # |µA|×µV/1e9 = mW
        if (cur <= 0 || cur > 30000) next                          # 越界样本作废。与 app 近似非全等：app 钳电压域且保留 cur=0 样本（本脚本丢弃）
        n++; sum += cur
        if (n > 1) e += (prev + cur) / 2 * dt                      # 梯形 → 中点等价；mW × s = mJ
        prev = cur
      }
      END {
        if (n < 2) { print "PERM_OR_EMPTY: 取不到有效 current/voltage 读数"; exit 3 }
        printf "samples=%d  avg_mW=%.1f  energy_mJ=%.0f  energy_mWh=%.4f\n", n, sum/n, e, e/3600
        printf "对照: bench.csv 同窗 cell 的 avg_power_mw / energy_J（energy_J 应 ≈ energy_mJ/1000，验收 ≤±30%%）\n"
      }'
RC=$?
if [ "$RC" -eq 3 ]; then
  echo "→ SELinux 拒绝 adb shell 读 sysfs。app 侧 power_source=PERM_DENIED 属预期降级（验收标准 2 改 batterystats 口径）："
  echo "   adb shell dumpsys batterystats --reset   # 跑矩阵前"
  echo "   adb shell dumpsys batterystats | grep -A3 'Battery run'   # 结束后取 estimate 毫焦/时长"
fi
echo "== dumpsys battery 摘录（电压/电流人工对照）=="
"$ADB" shell dumpsys battery | tr -d '\r' | sed -n '1,14p'
