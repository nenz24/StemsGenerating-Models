// Decoder 0.2 s and 1 s parity passed on device. Encoder device results pending.
import CoreAI
import CryptoKit
import Darwin
import Dispatch
import Foundation

enum CoreAIDecoderProbe {
    private enum Profile {
        case decoderShort, decoderOneSecond, encoderOneSecond
        var isEncoder: Bool { self == .encoderOneSecond }
        var component: String { isEncoder ? "encoder" : "decoder" }
        var frames: Int { self == .decoderShort ? 5 : 25 }
        var folderName: String {
            self == .decoderShort ? "decoder-pretrained" : "\(component)-pretrained-1s"
        }
        var inputName: String { isEncoder ? "audio" : "latent" }
        var outputName: String { isEncoder ? "latent" : "audio" }
        var inputShape: [Int] { isEncoder ? [1, 2, frames * 1920] : [1, 64, frames] }
        var outputShape: [Int] { isEncoder ? [1, 64, frames] : [1, 2, frames * 1920] }
    }

    private struct Contract: Decodable {
        struct Checkpoint: Decodable { let digest: String }
        let schemaVersion: Int
        let component: String?
        let posteriorMode: String?
        let assetFile: String
        let functionName: String
        let inputName: String
        let outputName: String
        let inputShape: [Int]
        let outputShape: [Int]
        let inputFile: String
        let expectedFile: String
        let scalarType: String
        let byteOrder: String
        let sampleRate: Int
        let checkpoint: Checkpoint
        let checkpointFullSha256Verified: Bool
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    private enum ProbeError: Error {
        case missingPreparedBundle
        case invalidContract
        case referenceIntegrityFailure(String)
        case invalidFloatData
        case invalidOutputTensor
        case numericalParityFailed
    }

    static func run(oneSecond: Bool = false) async -> String {
        await run(profile: oneSecond ? .decoderOneSecond : .decoderShort)
    }

    static func runEncoder() async -> String {
        await run(profile: .encoderOneSecond)
    }

    private static func run(profile: Profile) async -> String {
        let component = profile.component
        var lines = [
            "Runtime: CoreAI",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "CoreAI architecture: \(AIModel.deviceArchitectureName)",
            "Test: pretrained ACE VAE \(component) / \(profile.isEncoder ? "synthetic audio / posterior mean" : "seeded latent") / NOT instrument generation",
            "Requested compute policy: cpuOnly",
            "Accelerator used by a model: NOT MEASURED"
        ]
        lines.append("Window profile: \(profile.frames == 25 ? "1.0 s" : "0.2 s")")
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        let clock = ContinuousClock()
        let totalStart = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var loaded = false
        var executed = false
        do {
            guard let folder = Bundle.main.resourceURL?
                .appendingPathComponent("ModelCandidates/\(profile.folderName)"),
                FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append("Prepare on Mac: bash scripts/prepare_coreai_decoder.sh --component \(component) --frames \(profile.frames); then rebuild the app.")
                throw ProbeError.missingPreparedBundle
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let contract = try decoder.decode(Contract.self, from:
                Data(contentsOf: folder.appendingPathComponent("reference.json")))
            guard contract.schemaVersion == 1, contract.assetFile == "\(component).aimodel",
                  (contract.component ?? "decoder") == component,
                  (!profile.isEncoder || contract.posteriorMode == "mean"),
                  contract.functionName == "main", contract.inputName == profile.inputName, contract.outputName == profile.outputName,
                  contract.inputShape == profile.inputShape, contract.outputShape == profile.outputShape,
                  contract.sampleRate == 48000, contract.scalarType == "float32", contract.byteOrder == "little",
                  contract.inputFile == "\(profile.inputName).f32le", contract.expectedFile == "\(profile.outputName)-reference.f32le",
                  contract.checkpointFullSha256Verified,
                  contract.checkpoint.digest == "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0",
                  contract.absoluteTolerance > 0, contract.absoluteTolerance <= 0.001,
                  contract.relativeTolerance > 0, contract.relativeTolerance <= 0.01,
                  contract.relativeRmseTolerance > 0, contract.relativeRmseTolerance <= 0.01 else {
                throw ProbeError.invalidContract
            }
            let input = try readFloats(folder: folder, name: contract.inputFile,
                                       count: contract.inputShape.reduce(1, *), hashes: contract.artifactSha256)
            let expected = try readFloats(folder: folder, name: contract.expectedFile,
                                          count: contract.outputShape.reduce(1, *), hashes: contract.artifactSha256)
            let duration = Double(profile.frames) / 25
            lines.append("Audio duration / sample rate: \(duration) s / \(contract.sampleRate) Hz stereo")
            lines.append("Input/output shapes: \(contract.inputShape) -> \(contract.outputShape)")
            lines.append("Reference binary SHA-256: VERIFIED; pretrained checkpoint verified by Mac preparation")
            lines.append("Tolerances: abs \(contract.absoluteTolerance), rel \(contract.relativeTolerance), normalized RMSE \(contract.relativeRmseTolerance)")
            let url = folder.appendingPathComponent(contract.assetFile)
            let validationStart = clock.now
            lines.append("Source asset validation: \(AIModelAsset.isValid(at: url))")
            lines.append("Source asset validation ms: \(ms(validationStart.duration(to: clock.now)))")
            let loadStart = clock.now
            let probe = try await CoreAIModelProbe(contentsOf: url, options: .cpuOnly)
            loaded = true
            lines.append("Validated load/specialization ms: \(ms(loadStart.duration(to: clock.now)))")
            lines.append("Function names: \(probe.model.functionNames)")
            if let descriptor = probe.model.functionDescriptor(for: contract.functionName) {
                if let value = descriptor.inputDescriptor(of: contract.inputName), case .ndArray(let array) = value {
                    lines.append("Runtime input descriptor: \(array.shape), \(array.scalarType), interleave \(String(describing: array.interleaveLayout))")
                }
                if let value = descriptor.outputDescriptor(of: contract.outputName), case .ndArray(let array) = value {
                    lines.append("Runtime output descriptor: \(array.shape), \(array.scalarType), interleave \(String(describing: array.interleaveLayout))")
                }
            }
            let tensorInput = NDArray(scalars: input, shape: contract.inputShape)
            for iteration in 0..<3 {
                let start = clock.now
                var outputs = try await probe.runStateless(functionNamed: contract.functionName,
                                                           inputs: [contract.inputName: tensorInput])
                executed = true
                guard let value = outputs.remove(contract.outputName), let tensor = value.ndArray,
                      tensor.scalarType == .float32, tensor.shape == contract.outputShape,
                      tensor.interleaveLayout == nil else { throw ProbeError.invalidOutputTensor }
                let view = tensor.view(as: Float.self)
                guard view.isContiguous else { throw ProbeError.invalidOutputTensor }
                let actual = view.withUnsafePointer { pointer, _, _ in
                    Array(UnsafeBufferPointer(start: pointer, count: expected.count))
                }
                let elapsed = ms(start.duration(to: clock.now))
                var maxError = 0.0
                var squaredError = 0.0
                var squaredReference = 0.0
                var elementwisePass = true
                for (result, reference) in zip(actual, expected) {
                    guard result.isFinite else { throw ProbeError.invalidOutputTensor }
                    let error = abs(Double(result) - Double(reference))
                    maxError = max(maxError, error)
                    squaredError += error * error
                    squaredReference += Double(reference) * Double(reference)
                    if error > contract.absoluteTolerance + contract.relativeTolerance * abs(Double(reference)) {
                        elementwisePass = false
                    }
                }
                let rmse = sqrt(squaredError / Double(expected.count))
                let referenceRms = sqrt(squaredReference / Double(expected.count))
                let normalizedRmse = rmse / max(referenceRms, 1e-12)
                let label = iteration == 0 ? "First inference" : "Warm inference \(iteration)"
                lines.append("\(label) run+read-back ms: \(elapsed)")
                lines.append("Max abs error: \(maxError); RMSE: \(rmse); normalized RMSE: \(normalizedRmse)")
                guard elementwisePass, normalizedRmse <= contract.relativeRmseTolerance else {
                    throw ProbeError.numericalParityFailed
                }
            }
            lines.append("Pretrained \(component) parity: PASS (three runs)")
        } catch {
            let detail = error as NSError
            lines.append("Pretrained \(component) parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
            lines.append("Error domain/code: \(detail.domain) / \(detail.code)")
            lines.append("Description: \(detail.localizedDescription)")
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process footprint baseline MiB: \(Double(baseline) / 1048576)") }
        if let peak = usage.peak { lines.append("Sampled process footprint peak MiB (10 ms sampling): \(Double(peak) / 1048576)") }
        if usage.peak == nil { lines.append("Process footprint: unavailable (task_info failed)") }
        lines.append("Memory scope: whole process; sampled maximum, NOT exact peak or isolated model allocation")
        lines.append("Total \(component) probe ms: \(ms(totalStart.duration(to: clock.now)))")
        lines.append("Model loaded: \(loaded ? "YES" : "NO")")
        lines.append("Model inference executed: \(executed ? "YES" : "NO")")
        return lines.joined(separator: "\n")
    }

    private static func readFloats(folder: URL, name: String, count: Int, hashes: [String: String]) throws -> [Float] {
        let url = folder.appendingPathComponent(name)
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard (attributes[.size] as? NSNumber)?.intValue == count * 4 else {
            throw ProbeError.referenceIntegrityFailure(name)
        }
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard digest == hashes[name], data.count == count * 4 else {
            throw ProbeError.referenceIntegrityFailure(name)
        }
        let values: [Float] = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            (0..<count).map { offset in
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset * 4, as: UInt32.self)))
            }
        }
        guard values.allSatisfy({ $0.isFinite }) else { throw ProbeError.invalidFloatData }
        return values
    }

    private static func ms(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
    }
}

// Timer state lives on one serial queue. The owning task starts/stops the timer;
// it returns only numeric snapshots, never the timer or mutable state.
private final class CoreAIProcessFootprintSampler: @unchecked Sendable {
    private let queue = DispatchQueue(label: "nenz24.coreai.memory-sampling")
    private let timer: any DispatchSourceTimer
    private let baseline: UInt64?
    private var peak: UInt64?

    init() {
        let initial = Self.currentFootprint()
        baseline = initial
        peak = initial
        timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in self?.observe() }
        timer.resume()
    }

    func finish() -> (baseline: UInt64?, peak: UInt64?) {
        timer.cancel()
        return queue.sync {
            observe()
            return (baseline, peak)
        }
    }

    private func observe() {
        if let value = Self.currentFootprint() { peak = max(peak ?? value, value) }
    }

    private static func currentFootprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let capacity = Int(count)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) { words in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), words, &count)
            }
        }
        return status == KERN_SUCCESS ? info.phys_footprint : nil
    }

    deinit { timer.cancel() }
}
