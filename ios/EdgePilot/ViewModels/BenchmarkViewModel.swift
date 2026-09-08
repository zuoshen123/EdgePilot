import Foundation
import SwiftUI

struct BenchmarkResult {
    var ttftMs: Float = 0
    var itlAvgMs: Float = 0
    var itlP99Ms: Float = 0
    var tokensPerSec: Float = 0
    var totalTokens: Int = 0
    var totalTimeMs: Float = 0
    var acceptanceRate: Float = 0
}

@MainActor
class BenchmarkViewModel: ObservableObject {
    @Published var isRunning = false
    @Published var generatedText = ""
    @Published var result: BenchmarkResult?
    @Published var logs: [String] = []
    @Published var error: String?
    @Published var hardwareInfo: String?
    @Published var modelLoaded = false

    func detectHardware() {
        let raw = EdgePilotBridge.detectHardware()
        hardwareInfo = raw
        addLog("Hardware: \(raw)")
    }

    func loadModel(path: String, contextLength: Int = 2048) {
        isRunning = true
        error = nil
        addLog("Loading model: \(path)")

        Task.detached { [weak self] in
            let status = EdgePilotBridge.loadModel(path, contextLength: contextLength)
            await MainActor.run {
                self?.isRunning = false
                if status == 0 {
                    self?.modelLoaded = true
                    self?.addLog("Model loaded successfully")
                } else {
                    self?.error = "Model load failed (status=\(status))"
                }
            }
        }
    }

    func runBenchmark(prompt: String, maxTokens: Int = 128, temperature: Float = 0.7) {
        guard modelLoaded else {
            error = "Model not loaded"
            return
        }

        isRunning = true
        generatedText = ""
        error = nil
        addLog("Running benchmark...")

        Task.detached { [weak self] in
            EdgePilotBridge.startMetrics()

            let startTime = Date()
            let text = EdgePilotBridge.generate(prompt, maxTokens: maxTokens, temperature: temperature)
            let elapsed = Float(Date().timeIntervalSince(startTime) * 1000)

            EdgePilotBridge.stopMetrics()

            let tokenCount = text.split(separator: " ").count
            let tps = elapsed > 0 ? Float(tokenCount) * 1000 / elapsed : 0

            await MainActor.run {
                self?.isRunning = false
                self?.generatedText = text
                self?.result = BenchmarkResult(
                    totalTimeMs: elapsed,
                    totalTokens: tokenCount,
                    tokensPerSec: tps
                )
                self?.addLog("Done: \(tokenCount) tokens in \(Int(elapsed))ms")
            }
        }
    }

    func runSpeculativeBenchmark(draftPath: String, targetPath: String,
                                  prompt: String, maxTokens: Int = 128) {
        isRunning = true
        error = nil
        addLog("Initializing speculative decoding...")

        Task.detached { [weak self] in
            let status = EdgePilotBridge.initSpeculative(withDraft: draftPath,
                                                          target: targetPath,
                                                          window: 4)
            guard status == 0 else {
                await MainActor.run {
                    self?.isRunning = false
                    self?.error = "Speculative init failed"
                }
                return
            }

            let startTime = Date()
            let text = EdgePilotBridge.generate(prompt, maxTokens: maxTokens, temperature: 0.7)
            let elapsed = Float(Date().timeIntervalSince(startTime) * 1000)
            let rate = EdgePilotBridge.getAcceptanceRate()

            let tokenCount = text.split(separator: " ").count
            let tps = elapsed > 0 ? Float(tokenCount) * 1000 / elapsed : 0

            await MainActor.run {
                self?.isRunning = false
                self?.generatedText = text
                self?.result = BenchmarkResult(
                    totalTimeMs: elapsed,
                    totalTokens: tokenCount,
                    tokensPerSec: tps,
                    acceptanceRate: rate
                )
                self?.addLog("Acceptance rate: \(Int(rate * 100))%")
            }
        }
    }

    private func addLog(_ message: String) {
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        logs.append("[\(ts)] \(message)")
    }
}
