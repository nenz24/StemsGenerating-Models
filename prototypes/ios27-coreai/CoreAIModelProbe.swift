// A caller for a valid, locally exported CoreAI asset. This is not a converter.
// API declarations checked against uploaded interfaces; model execution pending.
import CoreAI
import Foundation

@available(iOS 27.0, macOS 27.0, *)
struct CoreAIModelProbe: Sendable {
    enum ProbeError: Error {
        case requiresValidLocalAsset
        case missingFunction(String)
        case statefulFunctionRequiresExplicitState
        case inputNamesMismatch
        case unsupportedInput(String)
        case inputTypeOrShapeMismatch(String)
    }

    let model: AIModel

    init(contentsOf url: URL, options: SpecializationOptions = .default) async throws {
        guard url.isFileURL, AIModelAsset.isValid(at: url) else {
            throw ProbeError.requiresValidLocalAsset
        }
        model = try await AIModel(contentsOf: url, options: options)
    }

    func runStateless(
        functionNamed name: String,
        inputs: [String: NDArray]
    ) async throws -> InferenceFunction.Outputs {
        guard let function = try model.loadFunction(named: name) else {
            throw ProbeError.missingFunction(name)
        }
        let descriptor = function.descriptor
        guard descriptor.stateNames.isEmpty else {
            throw ProbeError.statefulFunctionRequiresExplicitState
        }
        guard Set(inputs.keys) == Set(descriptor.inputNames) else {
            throw ProbeError.inputNamesMismatch
        }
        for (inputName, array) in inputs {
            guard let valueDescriptor = descriptor.inputDescriptor(of: inputName),
                  case .ndArray(let expected) = valueDescriptor else {
                throw ProbeError.unsupportedInput(inputName)
            }
            // Dynamic shapes need an explicit export contract; reject them here
            // rather than guessing how an audio model encodes dynamic dimensions.
            guard !expected.hasDynamicShape,
                  expected.interleaveLayout == nil,
                  array.interleaveLayout == nil,
                  array.scalarType == expected.scalarType,
                  array.shape == expected.shape else {
                throw ProbeError.inputTypeOrShapeMismatch(inputName)
            }
        }
        return try await function.run(inputs: inputs)
    }
}

// Runs outside the UI actor so noncopyable inference outputs are read in the
// same isolation domain as the model call; only report text returns to the UI.
@available(iOS 27.0, macOS 27.0, *)
enum CoreAINativeNumericalProbe {
    private struct Contract: Decodable {
        let assetFile: String
        let functionName: String
        let inputName: String
        let outputName: String
        let inputShape: [Int]
        let outputShape: [Int]
        let inputValues: [Float]
        let expectedValues: [Float]
        let absoluteTolerance: Float
    }

    private enum NumericalError: Error {
        case missingBundledCandidate
        case invalidReferenceContract
        case outputNotFloat32Tensor
        case unexpectedOutputLayout
        case parityFailed
    }

    static func run() async -> String {
        var lines = [
            "Runtime: CoreAI",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "CoreAI architecture: \(AIModel.deviceArchitectureName)",
            "Test: y = 2*x + 0.25 (native .aimodel)",
            "Requested compute policy: cpuOnly",
            "Accelerator used by a model: NOT MEASURED"
        ]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        var loaded = false
        var executed = false
        let clock = ContinuousClock()
        do {
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates"),
                  FileManager.default.fileExists(atPath: folder.path) else {
                throw NumericalError.missingBundledCandidate
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let contract = try decoder.decode(Contract.self, from: Data(contentsOf:
                folder.appendingPathComponent("affine-native-reference.json")))
            guard contract.assetFile == "affine.aimodel",
                  contract.inputShape == [1, 4], contract.outputShape == [1, 4],
                  contract.inputValues.count == 4, contract.expectedValues.count == 4,
                  contract.inputValues.allSatisfy({ $0.isFinite }),
                  contract.expectedValues.allSatisfy({ $0.isFinite }),
                  contract.absoluteTolerance.isFinite,
                  contract.absoluteTolerance > 0, contract.absoluteTolerance <= 0.001 else {
                throw NumericalError.invalidReferenceContract
            }
            let url = folder.appendingPathComponent(contract.assetFile)
            lines.append("Source asset validation: \(AIModelAsset.isValid(at: url))")
            // Validate and load the native asset produced by Apple's exporter.
            let start = clock.now
            let probe = try await CoreAIModelProbe(contentsOf: url, options: .cpuOnly)
            loaded = true
            lines.append("Load/specialization ms: \(milliseconds(start.duration(to: clock.now)))")
            lines.append("Function names: \(probe.model.functionNames)")
            for iteration in 0..<3 {
                let input = NDArray(scalars: contract.inputValues, shape: contract.inputShape)
                let runStart = clock.now
                var outputs = try await probe.runStateless(functionNamed: contract.functionName,
                                                           inputs: [contract.inputName: input])
                executed = true
                // Includes waiting for run() and extracting/reading the output.
                guard let value = outputs.remove(contract.outputName),
                      let tensor = value.ndArray, tensor.scalarType == .float32 else {
                    throw NumericalError.outputNotFloat32Tensor
                }
                guard tensor.shape == contract.outputShape,
                      tensor.interleaveLayout == nil else {
                    throw NumericalError.unexpectedOutputLayout
                }
                let view = tensor.view(as: Float.self)
                guard view.isContiguous else { throw NumericalError.unexpectedOutputLayout }
                let actual = view.withUnsafePointer { pointer, _, _ in
                    Array(UnsafeBufferPointer(start: pointer, count: contract.expectedValues.count))
                }
                let errors = zip(actual, contract.expectedValues).map { abs($0.0 - $0.1) }
                guard actual.allSatisfy({ $0.isFinite }),
                      errors.allSatisfy({ $0 <= contract.absoluteTolerance }) else {
                    lines.append("Actual values: \(actual)")
                    throw NumericalError.parityFailed
                }
                let label = iteration == 0 ? "First inference" : "Warm inference \(iteration)"
                lines.append("\(label) ms: \(milliseconds(runStart.duration(to: clock.now)))")
                lines.append("Values: \(actual); max absolute error: \(errors.max() ?? 0)")
            }
            lines.append("Numerical parity: PASS (three runs)")
        } catch {
            let detail = error as NSError
            lines.append("Numerical parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
            lines.append("Error domain/code: \(detail.domain) / \(detail.code)")
            lines.append("Description: \(detail.localizedDescription)")
        }
        lines.append("Model loaded: \(loaded ? "YES" : "NO")")
        lines.append("Model inference executed: \(executed ? "YES" : "NO")")
        return lines.joined(separator: "\n")
    }

    private static func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
    }
}
