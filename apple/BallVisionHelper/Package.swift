// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BallVisionHelper",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "BallVisionHelper", targets: ["BallVisionHelper"]),
    ],
    targets: [
        .executableTarget(
            name: "BallVisionHelper",
            path: "Sources"
        ),
    ]
)
