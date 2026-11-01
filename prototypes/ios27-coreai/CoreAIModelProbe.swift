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
