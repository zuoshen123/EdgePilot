import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = BenchmarkViewModel()
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            BenchmarkView(viewModel: viewModel)
                .tabItem {
                    Label("Benchmark", systemImage: "play.circle")
                }
                .tag(0)

            MetricsView(viewModel: viewModel)
                .tabItem {
                    Label("Metrics", systemImage: "chart.bar")
                }
                .tag(1)
        }
        .accentColor(Color(hex: "6366F1"))
        .onAppear {
            viewModel.detectHardware()
        }
    }
}

struct BenchmarkView: View {
    @ObservedObject var viewModel: BenchmarkViewModel
    @State private var prompt = "Explain quantum computing in simple terms"
    @State private var modelPath = ""

    var body: some View {
        NavigationView {
            List {
                // Hardware
                Section("Hardware") {
                    if let hw = viewModel.hardwareInfo {
                        Text(hw)
                    } else {
                        Text("Detecting...")
                            .foregroundColor(.secondary)
                    }
                }

                // Model
                Section("Model") {
                    TextField("Model path (.gguf)", text: $modelPath)
                        .textFieldStyle(.roundedBorder)

                    Button(action: { viewModel.loadModel(path: modelPath) }) {
                        HStack {
                            if viewModel.isRunning {
                                ProgressView()
                                    .scaleEffect(0.8)
                            }
                            Text(viewModel.modelLoaded ? "✓ Model Loaded" : "Load Model")
                        }
                    }
                    .disabled(viewModel.isRunning || modelPath.isEmpty)
                }

                // Benchmark
                Section("Benchmark") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 80)

                    Button(action: { viewModel.runBenchmark(prompt: prompt) }) {
                        HStack {
                            Image(systemName: "play.fill")
                            if viewModel.isRunning {
                                ProgressView()
                                    .scaleEffect(0.8)
                            }
                            Text(viewModel.isRunning ? "Running..." : "Run Benchmark")
                        }
                    }
                    .disabled(!viewModel.modelLoaded || viewModel.isRunning)
                }

                // Results
                if let result = viewModel.result {
                    Section("Results") {
                        HStack {
                            VStack {
                                Text(String(format: "%.1f", result.tokensPerSec))
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.green)
                                Text("tok/s")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            VStack {
                                Text("\(result.totalTokens)")
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.blue)
                                Text("tokens")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            VStack {
                                Text(String(format: "%.0f ms", result.totalTimeMs))
                                    .font(.title2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.orange)
                                Text("time")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }

                        if result.acceptanceRate > 0 {
                            VStack(alignment: .leading) {
                                ProgressView(value: result.acceptanceRate)
                                    .tint(.green)
                                Text("Speculative: \(Int(result.acceptanceRate * 100))%")
                                    .font(.caption)
                            }
                        }
                    }
                }

                // Generated text
                if !viewModel.generatedText.isEmpty {
                    Section("Generated Text") {
                        Text(viewModel.generatedText)
                            .font(.system(.body, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }

                // Error
                if let error = viewModel.error {
                    Section {
                        HStack {
                            Image(systemName: "exclamationmark.triangle")
                                .foregroundColor(.red)
                            Text(error)
                                .foregroundColor(.red)
                        }
                    }
                }

                // Logs
                if !viewModel.logs.isEmpty {
                    Section("Logs") {
                        ForEach(viewModel.logs.suffix(10), id: \.self) { log in
                            Text(log)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("EdgePilot")
        }
    }
}

struct MetricsView: View {
    @ObservedObject var viewModel: BenchmarkViewModel

    var body: some View {
        NavigationView {
            List {
                if let result = viewModel.result {
                    Section("Latency") {
                        MetricRow(label: "TTFT", value: "\(Int(result.ttftMs)) ms",
                                  color: result.ttftMs < 400 ? .green : .red)
                        MetricRow(label: "ITL Avg", value: "\(Int(result.itlAvgMs)) ms",
                                  color: result.itlAvgMs < 80 ? .green : .orange)
                    }

                    Section("Throughput") {
                        MetricRow(label: "Tokens/sec",
                                  value: String(format: "%.1f", result.tokensPerSec),
                                  color: result.tokensPerSec > 30 ? .green : .orange)
                    }
                } else {
                    Text("Run a benchmark to see metrics")
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Metrics")
        }
    }
}

struct MetricRow: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
    }
}

// Color hex extension
extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex)
        var rgbValue: UInt64 = 0
        scanner.scanHexInt64(&rgbValue)
        self.init(
            red: Double((rgbValue & 0xFF0000) >> 16) / 255,
            green: Double((rgbValue & 0x00FF00) >> 8) / 255,
            blue: Double(rgbValue & 0x0000FF) / 255
        )
    }
}

#Preview {
    ContentView()
}
