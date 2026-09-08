#!/usr/bin/env python3
"""
EdgePilot 离线性能分析工具

从 SQLite 数据库读取性能指标，生成分析报告。
"""

import json
import sys
import argparse
from pathlib import Path

def analyze_json_report(json_path: str):
    """分析 JSON 格式的性能报告"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    records = data.get('records', [])
    if not records:
        print("No records found")
        return

    # 计算统计
    ttfts = [r['ttft_ms'] for r in records if r.get('ttft_ms')]
    itls = [r['itl_avg_ms'] for r in records if r.get('itl_avg_ms')]
    tps = [r['tokens_per_sec'] for r in records if r.get('tokens_per_sec')]

    print("=" * 60)
    print("EdgePilot Performance Report")
    print("=" * 60)
    print(f"Total inferences: {len(records)}")
    print()

    if ttfts:
        print("TTFT (Time to First Token):")
        print(f"  Avg: {sum(ttfts)/len(ttfts):.1f} ms")
        print(f"  Min: {min(ttfts):.1f} ms")
        print(f"  Max: {max(ttfts):.1f} ms")
        print()

    if itls:
        sorted_itls = sorted(itls)
        print("ITL (Inter-Token Latency):")
        print(f"  Avg: {sum(itls)/len(itls):.1f} ms")
        print(f"  P50: {sorted_itls[len(sorted_itls)//2]:.1f} ms")
        print(f"  P95: {sorted_itls[int(len(sorted_itls)*0.95)]:.1f} ms")
        print(f"  P99: {sorted_itls[int(len(sorted_itls)*0.99)]:.1f} ms")
        print()

    if tps:
        print("Throughput:")
        print(f"  Avg: {sum(tps)/len(tps):.1f} tok/s")
        print(f"  Max: {max(tps):.1f} tok/s")
        print()

    # Speculative stats
    acceptance_rates = [r['acceptance_rate'] for r in records if r.get('acceptance_rate')]
    if acceptance_rates:
        print("Speculative Decoding:")
        print(f"  Avg acceptance rate: {sum(acceptance_rates)/len(acceptance_rates)*100:.1f}%")
        print()

    # Grade
    avg_tps = sum(tps)/len(tps) if tps else 0
    avg_ttft = sum(ttfts)/len(ttfts) if ttfts else 0
    grade = "A" if avg_tps > 30 and avg_ttft < 400 else \
            "B" if avg_tps > 15 and avg_ttft < 800 else "C"
    print(f"Overall Grade: {grade}")
    print("=" * 60)


def generate_chrome_trace(json_path: str, output_path: str):
    """生成 Chrome Tracing 格式文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    events = []
    for i, record in enumerate(data.get('records', [])):
        base_ts = i * 1000000  # 每条记录间隔 1s

        # TTFT event
        events.append({
            "name": "prefill",
            "cat": "inference",
            "ph": "X",
            "ts": base_ts,
            "dur": int(record.get('ttft_ms', 0) * 1000),
            "pid": 1,
            "tid": record.get('agent_id', 0)
        })

        # Decode events
        decode_start = base_ts + int(record.get('ttft_ms', 0) * 1000)
        for j in range(record.get('total_tokens', 0)):
            itl = int(record.get('itl_avg_ms', 50) * 1000)
            events.append({
                "name": f"token_{j}",
                "cat": "inference",
                "ph": "X",
                "ts": decode_start + j * itl,
                "dur": itl,
                "pid": 1,
                "tid": record.get('agent_id', 0)
            })

    trace = {"traceEvents": events}
    with open(output_path, 'w') as f:
        json.dump(trace, f)

    print(f"Chrome trace written to: {output_path}")
    print("Open chrome://tracing and load this file to visualize")


def main():
    parser = argparse.ArgumentParser(description="EdgePilot Performance Analyzer")
    parser.add_argument("input", help="Input JSON file from MetricsCollector")
    parser.add_argument("--trace", help="Export Chrome Tracing format", metavar="OUTPUT")
    args = parser.parse_args()

    analyze_json_report(args.input)

    if args.trace:
        generate_chrome_trace(args.input, args.trace)


if __name__ == "__main__":
    main()
