// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "EdgePilot",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    targets: [
        // C++ 核心库 (作为 system module 引入)
        .systemLibrary(
            name: "CEdgePilot",
            path: "EdgePilot/Core",
            pkgConfig: nil,
            providers: nil
        ),
        // Swift 应用
        .executableTarget(
            name: "EdgePilot",
            dependencies: ["CEdgePilot"],
            path: "EdgePilot",
            exclude: ["Core", "Bridge/EdgePilotBridge.mm"],
            sources: ["Views", "ViewModels", "EdgePilotApp.swift"],
            swiftSettings: [
                .unsafeFlags(["-I", "../core/include"])
            ]
        )
    ]
)
