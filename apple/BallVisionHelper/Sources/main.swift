import AVFoundation
import CoreImage
import CoreMedia
import CoreML
import Darwin
import Foundation
import ImageIO
import Vision

struct Options {
    enum ComputeUnitsOption: String {
        case all
        case cpuOnly
        case cpuAndGPU
        case cpuAndNeuralEngine

        var coreMLValue: MLComputeUnits {
            switch self {
            case .all:
                return .all
            case .cpuOnly:
                return .cpuOnly
            case .cpuAndGPU:
                return .cpuAndGPU
            case .cpuAndNeuralEngine:
                return .cpuAndNeuralEngine
            }
        }
    }

    var modelPath: String?
    var targetLabel: String?
    var cameraIndex: Int = 0
    var width: Int = 1920
    var height: Int = 1080
    var fps: Double = 60.0
    var detectEvery: Int = 3
    var confidence: Float = 0.20
    var computeUnits: ComputeUnitsOption = .all
    var localSearchScale: Double = 2.6
    var fullRecoverEvery: Int = 4
    var emitFrames = false
    var frameJPEGQuality: Float = 0.70
    var listCameras = false

    static func parse(_ args: [String]) throws -> Options {
        var options = Options()
        var index = 1
        while index < args.count {
            let arg = args[index]
            switch arg {
            case "--model":
                index += 1
                options.modelPath = try requireValue(args, index, flag: arg)
            case "--label":
                index += 1
                options.targetLabel = try requireValue(args, index, flag: arg)
            case "--camera":
                index += 1
                options.cameraIndex = Int(try requireValue(args, index, flag: arg)) ?? 0
            case "--width":
                index += 1
                options.width = Int(try requireValue(args, index, flag: arg)) ?? 1920
            case "--height":
                index += 1
                options.height = Int(try requireValue(args, index, flag: arg)) ?? 1080
            case "--fps":
                index += 1
                options.fps = Double(try requireValue(args, index, flag: arg)) ?? 60.0
            case "--detect-every":
                index += 1
                options.detectEvery = max(1, Int(try requireValue(args, index, flag: arg)) ?? 3)
            case "--confidence":
                index += 1
                options.confidence = Float(try requireValue(args, index, flag: arg)) ?? 0.20
            case "--local-search-scale":
                index += 1
                options.localSearchScale = max(1.2, Double(try requireValue(args, index, flag: arg)) ?? 2.6)
            case "--full-recover-every":
                index += 1
                options.fullRecoverEvery = max(1, Int(try requireValue(args, index, flag: arg)) ?? 4)
            case "--compute-units":
                index += 1
                let raw = try requireValue(args, index, flag: arg)
                guard let value = ComputeUnitsOption(rawValue: raw) else {
                    throw NSError(domain: "BallVisionHelper", code: 2, userInfo: [
                        NSLocalizedDescriptionKey: "Invalid --compute-units value '\(raw)'. Use all, cpuOnly, cpuAndGPU, or cpuAndNeuralEngine."
                    ])
                }
                options.computeUnits = value
            case "--emit-frames":
                options.emitFrames = true
            case "--frame-jpeg-quality":
                index += 1
                let raw = try requireValue(args, index, flag: arg)
                options.frameJPEGQuality = min(1.0, max(0.20, Float(raw) ?? 0.70))
            case "--list-cameras":
                options.listCameras = true
            case "--help", "-h":
                printUsageAndExit()
            default:
                throw NSError(domain: "BallVisionHelper", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Unknown argument: \(arg)"
                ])
            }
            index += 1
        }
        return options
    }

    private static func requireValue(_ args: [String], _ index: Int, flag: String) throws -> String {
        guard index < args.count else {
            throw NSError(domain: "BallVisionHelper", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Missing value for \(flag)"
            ])
        }
        return args[index]
    }
}

struct DetectionMessage: Encodable {
    let type = "detection"
    let timestamp: Double
    let frameIndex: Int
    let source: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    let radius: Double
}

struct FrameMessage: Encodable {
    let type = "frame"
    let timestamp: Double
    let frameIndex: Int
    let width: Int
    let height: Int
    let jpeg: String
}

func stderrPrint(_ line: String) {
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

func printUsageAndExit() -> Never {
    let usage = """
    Usage:
      swift run BallVisionHelper --list-cameras
      swift run BallVisionHelper --model /path/to/model.mlmodelc [options]

    Options:
      --model PATH         Core ML model (.mlmodel, .mlpackage, or .mlmodelc)
      --label NAME         Optional target label filter (example: ball)
      --camera INDEX       Camera index from --list-cameras (default: 0)
      --width N            Preferred width (default: 1920)
      --height N           Preferred height (default: 1080)
      --fps N              Preferred fps (default: 60)
      --detect-every N     Run full detection every N frames, track between them (default: 3)
      --confidence N       Minimum label confidence (default: 0.20)
      --local-search-scale N Expand local detect ROI around the tracked ball by this factor (default: 2.6)
      --full-recover-every N Force a full-frame detector recovery every N detect passes while tracking (default: 4)
      --compute-units MODE Core ML compute units: all, cpuOnly, cpuAndGPU, cpuAndNeuralEngine (default: all)
      --emit-frames        Emit JPEG-compressed preview frames on stdout in addition to detections
      --frame-jpeg-quality JPEG quality for --emit-frames, 0.20..1.00 (default: 0.70)
    """
    print(usage)
    exit(0)
}

func discoverVideoDevices() -> [AVCaptureDevice] {
    var deviceTypes: [AVCaptureDevice.DeviceType] = [.externalUnknown, .builtInWideAngleCamera]
    if #available(macOS 14.0, *) {
        deviceTypes.append(.continuityCamera)
    }
    let discovery = AVCaptureDevice.DiscoverySession(
        deviceTypes: deviceTypes,
        mediaType: .video,
        position: .unspecified
    )
    return discovery.devices
}

func listCameras() {
    let devices = discoverVideoDevices()
    if devices.isEmpty {
        stderrPrint("[VisionHelper] No video devices found.")
        return
    }
    for (idx, device) in devices.enumerated() {
        stderrPrint("[VisionHelper] camera[\(idx)] \(device.localizedName) id=\(device.uniqueID)")
    }
}

func compileOrLoadModel(at path: String, computeUnits: MLComputeUnits) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    let ext = url.pathExtension.lowercased()
    if ext == "mlmodelc" {
        return try MLModel(contentsOf: url, configuration: configuration)
    }
    let compiledURL = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiledURL, configuration: configuration)
}

func normalizedTopLeftCenter(from bbox: CGRect) -> (Double, Double) {
    let centerX = Double(bbox.origin.x + 0.5 * bbox.size.width)
    let centerYVision = Double(bbox.origin.y + 0.5 * bbox.size.height)
    return (centerX, 1.0 - centerYVision)
}

struct LetterboxInfo {
    let fullWidth: Double
    let fullHeight: Double
    let cropX: Double
    let cropYTop: Double
    let sourceWidth: Double
    let sourceHeight: Double
    let targetWidth: Double
    let targetHeight: Double
    let scale: Double
    let padX: Double
    let padY: Double
}

final class VisionBallDetector {
    private let rawModel: MLModel
    private let modelInputName: String
    private let modelOutputName: String
    private let modelInputWidth: Double
    private let modelInputHeight: Double
    private let ciContext = CIContext(options: nil)
    private let visionModel: VNCoreMLModel
    private let targetLabel: String?
    private let confidenceThreshold: Float
    private let detectEvery: Int
    private let localSearchScale: Double
    private let fullRecoverEvery: Int
    private var trackedObservation: VNDetectedObjectObservation?
    private var frameIndex = 0
    private var detectPassIndex = 0

    var currentFrameIndex: Int {
        frameIndex
    }

    init(
        modelPath: String,
        targetLabel: String?,
        confidenceThreshold: Float,
        detectEvery: Int,
        localSearchScale: Double,
        fullRecoverEvery: Int,
        computeUnits: Options.ComputeUnitsOption
    ) throws {
        let model = try compileOrLoadModel(at: modelPath, computeUnits: computeUnits.coreMLValue)
        self.rawModel = model
        self.modelInputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        self.modelOutputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "var_1440"
        if let imageConstraint = model.modelDescription.inputDescriptionsByName[self.modelInputName]?.imageConstraint {
            self.modelInputWidth = Double(imageConstraint.pixelsWide)
            self.modelInputHeight = Double(imageConstraint.pixelsHigh)
        } else {
            self.modelInputWidth = 640.0
            self.modelInputHeight = 640.0
        }
        self.visionModel = try VNCoreMLModel(for: model)
        self.targetLabel = targetLabel
        self.confidenceThreshold = confidenceThreshold
        self.detectEvery = max(1, detectEvery)
        self.localSearchScale = max(1.2, localSearchScale)
        self.fullRecoverEvery = max(1, fullRecoverEvery)
        stderrPrint("[VisionHelper] compute_units=\(computeUnits.rawValue)")
    }

    func process(_ sampleBuffer: CMSampleBuffer) -> DetectionMessage? {
        frameIndex += 1
        let performDetect = trackedObservation == nil || frameIndex % detectEvery == 0
        do {
            if let local = try detectLocal(sampleBuffer) {
                return local
            }
            if performDetect {
                detectPassIndex += 1
                return try detect(sampleBuffer)
            }
            return nil
        } catch {
            if error.localizedDescription.contains("Exceeded maximum allowed number of Trackers") {
                clearTrackingState()
            }
            stderrPrint("[VisionHelper] Vision error: \(error.localizedDescription)")
            return nil
        }
    }

    private func detectLocal(_ sampleBuffer: CMSampleBuffer) throws -> DetectionMessage? {
        guard let trackedObservation else {
            return nil
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            clearTrackingState()
            return nil
        }
        guard let searchRect = expandedSearchRect(
            from: trackedObservation,
            fullWidth: CVPixelBufferGetWidth(pixelBuffer),
            fullHeight: CVPixelBufferGetHeight(pixelBuffer),
            scale: localSearchScale
        ) else {
            return nil
        }
        guard let raw = try detectRaw(pixelBuffer: pixelBuffer, cropRectTopLeft: searchRect, sourceLabel: "detect_local") else {
            return nil
        }
        updateTrackingObservation(raw.observation, resetRequest: false)
        return raw.message
    }

    private func detect(_ sampleBuffer: CMSampleBuffer) throws -> DetectionMessage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            trackedObservation = nil
            return nil
        }
        let shouldRecoverFullFrame = trackedObservation == nil || detectPassIndex % max(1, fullRecoverEvery) == 0
        guard shouldRecoverFullFrame else {
            return nil
        }
        guard let raw = try detectRaw(pixelBuffer: pixelBuffer, cropRectTopLeft: nil, sourceLabel: "detect_full") else {
            return nil
        }
        updateTrackingObservation(raw.observation, resetRequest: false)
        return raw.message
    }

    private func detectRaw(
        pixelBuffer: CVPixelBuffer,
        cropRectTopLeft: CGRect?,
        sourceLabel: String
    ) throws -> (observation: VNDetectedObjectObservation, message: DetectionMessage)? {
        guard let (resizedBuffer, letterbox) = resizedPixelBuffer(from: pixelBuffer, cropRectTopLeft: cropRectTopLeft) else {
            return nil
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            modelInputName: MLFeatureValue(pixelBuffer: resizedBuffer)
        ])
        let prediction = try rawModel.prediction(from: provider)
        guard let multiArray = prediction.featureValue(for: modelOutputName)?.multiArrayValue else {
            return nil
        }
        return decodeRawDetections(from: multiArray, letterbox: letterbox, sourceLabel: sourceLabel)
    }

    private func resizedPixelBuffer(from source: CVPixelBuffer, cropRectTopLeft: CGRect?) -> (CVPixelBuffer, LetterboxInfo)? {
        let targetWidth = max(1, Int(round(modelInputWidth)))
        let targetHeight = max(1, Int(round(modelInputHeight)))

        var output: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: targetWidth,
            kCVPixelBufferHeightKey: targetHeight,
            kCVPixelBufferIOSurfacePropertiesKey: [:],
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &output
        )
        guard status == kCVReturnSuccess, let output else {
            return nil
        }

        let fullWidth = CGFloat(CVPixelBufferGetWidth(source))
        let fullHeight = CGFloat(CVPixelBufferGetHeight(source))
        let cropTopLeft = (cropRectTopLeft ?? CGRect(x: 0, y: 0, width: fullWidth, height: fullHeight))
            .intersection(CGRect(x: 0, y: 0, width: fullWidth, height: fullHeight))
        if cropTopLeft.width < 8.0 || cropTopLeft.height < 8.0 {
            return nil
        }
        let dstWidth = CGFloat(targetWidth)
        let dstHeight = CGFloat(targetHeight)
        let scale = min(dstWidth / max(1.0, cropTopLeft.width), dstHeight / max(1.0, cropTopLeft.height))
        let renderWidth = cropTopLeft.width * scale
        let renderHeight = cropTopLeft.height * scale
        let padX = (dstWidth - renderWidth) * 0.5
        let padY = (dstHeight - renderHeight) * 0.5

        let cropBottomY = fullHeight - cropTopLeft.origin.y - cropTopLeft.height
        let image = CIImage(cvPixelBuffer: source)
            .cropped(to: CGRect(x: cropTopLeft.origin.x, y: cropBottomY, width: cropTopLeft.width, height: cropTopLeft.height))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: padX, y: padY))
        let background = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 1))
            .cropped(to: CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight))
        let composed = image.composited(over: background)
        ciContext.render(
            composed,
            to: output,
            bounds: CGRect(x: 0, y: 0, width: dstWidth, height: dstHeight),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        let info = LetterboxInfo(
            fullWidth: Double(fullWidth),
            fullHeight: Double(fullHeight),
            cropX: Double(cropTopLeft.origin.x),
            cropYTop: Double(cropTopLeft.origin.y),
            sourceWidth: Double(cropTopLeft.width),
            sourceHeight: Double(cropTopLeft.height),
            targetWidth: Double(dstWidth),
            targetHeight: Double(dstHeight),
            scale: Double(scale),
            padX: Double(padX),
            padY: Double(padY)
        )
        return (output, info)
    }

    private func decodeRawDetections(
        from multiArray: MLMultiArray,
        letterbox: LetterboxInfo,
        sourceLabel: String
    ) -> (observation: VNDetectedObjectObservation, message: DetectionMessage)? {
        let shape = multiArray.shape.map { Int(truncating: $0) }
        let rows: Int
        let cols: Int
        if shape.count == 3, shape[0] == 1, shape[2] == 6 {
            rows = shape[1]
            cols = shape[2]
        } else if shape.count == 2, shape[1] == 6 {
            rows = shape[0]
            cols = shape[1]
        } else {
            return nil
        }
        guard cols == 6 else {
            return nil
        }

        var bestRow: Int? = nil
        var bestConfidence = confidenceThreshold
        for row in 0..<rows {
            let conf = multiArrayValue(multiArray, row: row, col: 4, rank: shape.count)
            if conf < bestConfidence {
                continue
            }
            let cls = Int(round(multiArrayValue(multiArray, row: row, col: 5, rank: shape.count)))
            if let targetLabel, !targetLabel.isEmpty {
                if targetLabel.lowercased() != "ball" || cls != 0 {
                    continue
                }
            }
            let x1 = multiArrayValue(multiArray, row: row, col: 0, rank: shape.count)
            let y1 = multiArrayValue(multiArray, row: row, col: 1, rank: shape.count)
            let x2 = multiArrayValue(multiArray, row: row, col: 2, rank: shape.count)
            let y2 = multiArrayValue(multiArray, row: row, col: 3, rank: shape.count)
            if x2 <= x1 || y2 <= y1 {
                continue
            }
            bestRow = row
            bestConfidence = conf
        }

        guard let row = bestRow else {
            return nil
        }

        let x1Model = Swift.max(0.0, Swift.min(modelInputWidth, Double(multiArrayValue(multiArray, row: row, col: 0, rank: shape.count))))
        let y1Model = Swift.max(0.0, Swift.min(modelInputHeight, Double(multiArrayValue(multiArray, row: row, col: 1, rank: shape.count))))
        let x2Model = Swift.max(0.0, Swift.min(modelInputWidth, Double(multiArrayValue(multiArray, row: row, col: 2, rank: shape.count))))
        let y2Model = Swift.max(0.0, Swift.min(modelInputHeight, Double(multiArrayValue(multiArray, row: row, col: 3, rank: shape.count))))
        let x1Crop = Swift.max(0.0, Swift.min(letterbox.sourceWidth, (x1Model - letterbox.padX) / Swift.max(1e-6, letterbox.scale)))
        let y1Crop = Swift.max(0.0, Swift.min(letterbox.sourceHeight, (y1Model - letterbox.padY) / Swift.max(1e-6, letterbox.scale)))
        let x2Crop = Swift.max(0.0, Swift.min(letterbox.sourceWidth, (x2Model - letterbox.padX) / Swift.max(1e-6, letterbox.scale)))
        let y2Crop = Swift.max(0.0, Swift.min(letterbox.sourceHeight, (y2Model - letterbox.padY) / Swift.max(1e-6, letterbox.scale)))
        let x1 = letterbox.cropX + x1Crop
        let y1 = letterbox.cropYTop + y1Crop
        let x2 = letterbox.cropX + x2Crop
        let y2 = letterbox.cropYTop + y2Crop
        let normX = x1 / Swift.max(1.0, letterbox.fullWidth)
        let normYTop = y1 / Swift.max(1.0, letterbox.fullHeight)
        let normW = Swift.max(0.0, (x2 - x1) / Swift.max(1.0, letterbox.fullWidth))
        let normH = Swift.max(0.0, (y2 - y1) / Swift.max(1.0, letterbox.fullHeight))
        if normW <= 0.0 || normH <= 0.0 {
            return nil
        }

        let bbox = CGRect(x: normX, y: 1.0 - (normYTop + normH), width: normW, height: normH)
        let observation = VNDetectedObjectObservation(boundingBox: bbox)
        return (observation, buildMessage(observation: observation, confidence: bestConfidence, source: sourceLabel))
    }

    private func multiArrayValue(_ array: MLMultiArray, row: Int, col: Int, rank: Int) -> Float {
        let index: [NSNumber]
        if rank == 3 {
            index = [0, NSNumber(value: row), NSNumber(value: col)]
        } else {
            index = [NSNumber(value: row), NSNumber(value: col)]
        }
        return array[index].floatValue
    }

    private func expandedSearchRect(
        from observation: VNDetectedObjectObservation,
        fullWidth: Int,
        fullHeight: Int,
        scale: Double
    ) -> CGRect? {
        let bbox = observation.boundingBox
        if bbox.width <= 0.0 || bbox.height <= 0.0 {
            return nil
        }
        let centerXNorm = bbox.origin.x + 0.5 * bbox.width
        let centerYTopNorm = 1.0 - bbox.origin.y - 0.5 * bbox.height
        let sideNorm = max(bbox.width, bbox.height) * scale
        let xNorm = max(0.0, centerXNorm - 0.5 * sideNorm)
        let yTopNorm = max(0.0, centerYTopNorm - 0.5 * sideNorm)
        let clampedSideX = min(1.0 - xNorm, sideNorm)
        let clampedSideY = min(1.0 - yTopNorm, sideNorm)
        let side = max(0.04, min(clampedSideX, clampedSideY))
        return CGRect(
            x: xNorm * Double(fullWidth),
            y: yTopNorm * Double(fullHeight),
            width: side * Double(fullWidth),
            height: side * Double(fullHeight)
        )
    }

    private func updateTrackingObservation(_ observation: VNDetectedObjectObservation, resetRequest: Bool) {
        trackedObservation = observation
        _ = resetRequest
    }

    private func clearTrackingState() {
        trackedObservation = nil
    }

    private func buildMessage(observation: VNDetectedObjectObservation, confidence: VNConfidence, source: String) -> DetectionMessage {
        let bbox = observation.boundingBox
        let (centerX, centerY) = normalizedTopLeftCenter(from: bbox)
        let radius = Double(max(bbox.width, bbox.height) * 0.5)
        return DetectionMessage(
            timestamp: Date().timeIntervalSince1970,
            frameIndex: frameIndex,
            source: source,
            confidence: Float(confidence),
            x: centerX,
            y: centerY,
            width: Double(bbox.width),
            height: Double(bbox.height),
            radius: radius
        )
    }
}

final class CameraRunner: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let options: Options
    private let detector: VisionBallDetector
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "BallVisionHelper.VideoQueue")
    private let encoder = JSONEncoder()
    private let framePermit = DispatchSemaphore(value: 1)
    private let ciContext = CIContext(options: nil)
    private var lastStatsTime = Date()
    private var deliveredFrames = 0

    init(options: Options, detector: VisionBallDetector) {
        self.options = options
        self.detector = detector
    }

    func start() throws {
        let devices = discoverVideoDevices()
        guard options.cameraIndex >= 0, options.cameraIndex < devices.count else {
            throw NSError(domain: "BallVisionHelper", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Camera index \(options.cameraIndex) is out of range. Run with --list-cameras."
            ])
        }
        let device = devices[options.cameraIndex]
        stderrPrint("[VisionHelper] using camera[\(options.cameraIndex)] \(device.localizedName)")

        try configure(device: device)
        session.startRunning()
        stderrPrint("[VisionHelper] session started. Press Ctrl+C to stop.")
        RunLoop.main.run()
    }

    private func configure(device: AVCaptureDevice) throws {
        session.beginConfiguration()
        session.sessionPreset = .high

        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
        }

        try configureActiveFormat(device)

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.commitConfiguration()
    }

    private func configureActiveFormat(_ device: AVCaptureDevice) throws {
        let requestedFPS = options.fps
        let targetWidth = options.width
        let targetHeight = options.height

        let bestFormat = device.formats
            .compactMap { format -> (AVCaptureDevice.Format, Int32, Int32, Double)? in
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                let supportsFPS = format.videoSupportedFrameRateRanges.contains { range in
                    range.minFrameRate <= requestedFPS && requestedFPS <= range.maxFrameRate
                }
                guard supportsFPS else { return nil }
                return (format, dims.width, dims.height, requestedFPS)
            }
            .min { lhs, rhs in
                let lhsScore = abs(Int(lhs.1) - targetWidth) + abs(Int(lhs.2) - targetHeight)
                let rhsScore = abs(Int(rhs.1) - targetWidth) + abs(Int(rhs.2) - targetHeight)
                return lhsScore < rhsScore
            }

        guard let (format, width, height, fps) = bestFormat else {
            stderrPrint("[VisionHelper] no exact camera format found for \(targetWidth)x\(targetHeight)@\(requestedFPS). Using device defaults.")
            return
        }

        try device.lockForConfiguration()
        device.activeFormat = format
        let timescale = Int32(max(1, Int(round(fps))))
        let duration = CMTime(value: 1, timescale: timescale)
        device.activeVideoMinFrameDuration = duration
        device.activeVideoMaxFrameDuration = duration
        device.unlockForConfiguration()
        stderrPrint("[VisionHelper] requested \(targetWidth)x\(targetHeight)@\(requestedFPS) -> using \(width)x\(height)@\(fps)")
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard framePermit.wait(timeout: .now()) == .success else {
            return
        }
        defer { framePermit.signal() }

        deliveredFrames += 1
        let message = detector.process(sampleBuffer)
        let currentFrameIndex = detector.currentFrameIndex
        if let message, let data = try? encoder.encode(message) {
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
        if options.emitFrames,
           let frameMessage = buildFrameMessage(from: sampleBuffer, frameIndex: currentFrameIndex),
           let data = try? encoder.encode(frameMessage)
        {
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        }

        let now = Date()
        let elapsed = now.timeIntervalSince(lastStatsTime)
        if elapsed >= 1.0 {
            let fps = Double(deliveredFrames) / elapsed
            stderrPrint(String(format: "[VisionHelper] input_fps=%.1f", fps))
            lastStatsTime = now
            deliveredFrames = 0
        }
    }

    private func buildFrameMessage(from sampleBuffer: CMSampleBuffer, frameIndex: Int) -> FrameMessage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            return nil
        }
        let options: [CIImageRepresentationOption: Any] = [
            kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: self.options.frameJPEGQuality
        ]
        guard let jpegData = ciContext.jpegRepresentation(of: image, colorSpace: colorSpace, options: options) else {
            return nil
        }
        return FrameMessage(
            timestamp: Date().timeIntervalSince1970,
            frameIndex: frameIndex,
            width: width,
            height: height,
            jpeg: jpegData.base64EncodedString()
        )
    }
}

do {
    let options = try Options.parse(CommandLine.arguments)
    if options.listCameras {
        listCameras()
        exit(0)
    }
    guard let modelPath = options.modelPath else {
        printUsageAndExit()
    }

    let detector = try VisionBallDetector(
        modelPath: modelPath,
        targetLabel: options.targetLabel,
        confidenceThreshold: options.confidence,
        detectEvery: options.detectEvery,
        localSearchScale: options.localSearchScale,
        fullRecoverEvery: options.fullRecoverEvery,
        computeUnits: options.computeUnits
    )
    let runner = CameraRunner(options: options, detector: detector)
    try runner.start()
} catch {
    stderrPrint("[VisionHelper] fatal: \(error.localizedDescription)")
    exit(1)
}
