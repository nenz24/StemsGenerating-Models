// Component gates and the four-step synthetic denoiser loop passed on device.
import CoreAI
import CryptoKit
import Darwin
import Dispatch
import Foundation

enum CoreAIDecoderProbe {
    private enum Profile {
        case decoderShort, decoderOneSecond, encoderOneSecond, ditBlock, ditContext, ditStorage16, denoiser
        var isEncoder: Bool { self == .encoderOneSecond }
        var isStorage16: Bool { self == .ditStorage16 }
        var isDenoiser: Bool { self == .denoiser }
        var hasStoredHalfWeights: Bool { isStorage16 || isDenoiser }
        var isDiTContext: Bool { self == .ditContext || isStorage16 }
        var isDiT: Bool { self == .ditBlock || isDiTContext }
        var component: String { isDenoiser ? "denoiser" : (isStorage16 ? "dit-storage16" : (isDiT ? (isDiTContext ? "dit-context" : "dit-block") : (isEncoder ? "encoder" : "decoder"))) }
        var frames: Int { self == .decoderShort ? 5 : 25 }
        var folderName: String {
            (isDiT || isDenoiser) ? "\(component)-pretrained" : (self == .decoderShort ? "decoder-pretrained" : "\(component)-pretrained-1s")
        }
        var inputName: String { isEncoder ? "audio" : "latent" }
        var outputName: String { isDenoiser ? "velocity" : (isDiT ? "hidden_out" : (isEncoder ? "latent" : "audio")) }
        var inputShape: [Int] { isEncoder ? [1, 2, frames * 1920] : [1, 64, frames] }
        var hiddenTokens: Int { isDiTContext ? 160 : 16 }
        var conditionTokens: Int { isDiTContext ? 64 : 8 }
        var outputShape: [Int] { isDenoiser ? [1, 25, 64] : (isDiT ? [1, hiddenTokens, 2048] : (isEncoder ? [1, 64, frames] : [1, 2, frames * 1920])) }
        var inputShapes: [String: [Int]] {
            if isDenoiser {
                return ["noise": [1, 25, 64], "condition": [1, 64, 2048], "context": [1, 25, 128],
                        "timestep": [1], "timestep_r": [1]]
            }
            guard isDiT else { return [inputName: inputShape] }
            var shapes = ["hidden": [1, hiddenTokens, 2048], "condition": [1, conditionTokens, 2048], "temb": [1, 6, 2048],
                          "rope_cos": [1, hiddenTokens, 128], "rope_sin": [1, hiddenTokens, 128],
                          "self_mask": [1, 1, hiddenTokens, hiddenTokens], "cross_mask": [1, 1, hiddenTokens, conditionTokens]]
            if isDiTContext { shapes["local_mask"] = [1, 1, hiddenTokens, hiddenTokens] }
            return shapes
        }
        var expectedFile: String { isDenoiser ? "velocity-reference.f32le" : (isDiT ? "hidden-reference.f32le" : "\(outputName)-reference.f32le") }
        var originalReferenceFile: String { isDenoiser ? "velocity-fp32-reference.f32le" : "hidden-fp32-reference.f32le" }
        var checkpointDigest: String {
            (isDiT || isDenoiser) ? "4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b" :
                "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0"
        }
    }

    private struct Contract: Decodable {
        struct Stage: Decodable {
            let assetFile: String
            let layerIndices: [Int]
            let inputShapes: [String: [Int]]
            let outputShapes: [String: [Int]]
            let outputBindings: [String: String]?
        }
        struct Checkpoint: Decodable { let digest: String }
        let schemaVersion: Int
        let component: String?
        let posteriorMode: String?
        let assetFile: String
        let functionName: String
        let inputName: String?
        let outputName: String
        let inputShape: [Int]?
        let inputShapes: [String: [Int]]?
        let layerPrefix: String?
        let layerPrefixes: [String]?
        let profile: String?
        let slidingWindow: Int?
        let maskGeometryVerified: Bool?
        let layerCount: Int?
        let checkpointPrefix: String?
        let latentFrames: Int?
        let stages: [Stage]?
        let weightStorageDtype: String?
        let computeDtype: String?
        let originalFp32ReferenceFile: String?
        let outputShape: [Int]
        let inputFile: String?
        let expectedFile: String?
        let scalarType: String
        let byteOrder: String
        let sampleRate: Int?
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
        case originalFP32ParityFailed
    }

    static func run(oneSecond: Bool = false) async -> String {
        await run(profile: oneSecond ? .decoderOneSecond : .decoderShort)
    }

    static func runEncoder() async -> String {
        await run(profile: .encoderOneSecond)
    }

    static func runDiTBlock() async -> String {
        await run(profile: .ditBlock)
    }

    static func runDiTContext() async -> String {
        await run(profile: .ditContext)
    }

    static func runDiTStorage16() async -> String {
        await run(profile: .ditStorage16)
    }

    static func runDenoiser() async -> String {
        "Monolithic denoiser disabled after EXC_RESOURCE at loading/specializing (reported limit 3376 MB). Prepare and run Uji denoiser bertahap."
    }

    private enum ShardError: Error {
        case invalidManifest, missingPreparedBundle, memoryBudgetReached, footprintUnavailable
        case manifestMismatch(String)
        case memoryBudgetReachedAfterBind(asset: String, mib: Double)
    }

    private static func checkManifest<Value: Equatable>(_ actual: Value, _ expected: Value, _ field: String) throws {
        guard actual == expected else {
            throw ShardError.manifestMismatch("\(field): expected \(expected), got \(actual)")
        }
    }

    private struct LoopContract: Decodable {
        struct Step: Decodable {
            let index: Int
            let timestep: Float
            let nextTimestep: Float
            let timestepR: Float
            let velocityFile: String
            let latentFile: String
            let originalVelocityFile: String
            let originalLatentFile: String
        }
        let schemaVersion: Int
        let profile: String
        let method: String
        let timesteps: [Float]
        let steps: [Step]
        let shift: Float
        let guidanceScale: Float
        let kvCacheEnabled: Bool
        let inputShapes: [String: [Int]]
        let outputShape: [Int]
        let weightStorageDtype: String
        let computeDtype: String
        let scalarType: String
        let byteOrder: String
        let checkpointDigest: String
        let checkpointFullSha256Verified: Bool
        let sharedInputSha256: [String: String]
        let inputFiles: [String: String]
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    private struct StagePhases: Sendable {
        var count = 0
        var modelOpenMs = 0.0
        var functionBindMs = 0.0
        var inputValidationMs = 0.0
        var backendAwaitMs = 0.0
        var outputCopyMs = 0.0
        var runOtherMs = 0.0

        var measuredMs: Double {
            modelOpenMs + functionBindMs + inputValidationMs + backendAwaitMs + outputCopyMs + runOtherMs
        }

        mutating func add(_ other: StagePhases) {
            count += other.count
            modelOpenMs += other.modelOpenMs
            functionBindMs += other.functionBindMs
            inputValidationMs += other.inputValidationMs
            backendAwaitMs += other.backendAwaitMs
            outputCopyMs += other.outputCopyMs
            runOtherMs += other.runOtherMs
        }

        var report: String {
            "model-open ms \(modelOpenMs); function-bind ms \(functionBindMs); input-validation ms \(inputValidationMs); backend-await ms \(backendAwaitMs); output-copy ms \(outputCopyMs); run-other ms \(runOtherMs)"
        }

        var dominantPhase: String {
            let phases = [("model-open", modelOpenMs), ("function-bind", functionBindMs),
                          ("input-validation", inputValidationMs), ("backend-await", backendAwaitMs),
                          ("output-copy", outputCopyMs), ("run-other", runOtherMs)]
            guard count > 0, measuredMs > 0, let largest = phases.max(by: { $0.1 < $1.1 }) else {
                return "NOT MEASURED (no completed stages)"
            }
            return "\(largest.0); \(largest.1) ms; \(100 * largest.1 / measuredMs)% of completed-stage measured time"
        }
    }

    private struct RetainedStage: Sendable {
        let probe: CoreAIModelProbe
        let function: InferenceFunction
    }

    private struct RetentionStats: Sendable {
        var modelOpens = 0
        var functionBinds = 0
        var cacheHits = 0
        var completedCalls = 0
        var phaseTotals = StagePhases()
        var firstCallPhaseTotals = StagePhases()
        var repeatedCallPhaseTotals = StagePhases()
        var phaseTotalsByAsset: [String: StagePhases] = [:]
    }

    private struct CacheRepairStats: Sendable {
        // One repair for the entire generation, not one per asset or step.
        var attempts = 0
        var successfulRetries = 0
        var events: [String] = []
    }

    private typealias StageResult = (values: [String: NDArray], loadMs: Double, runMs: Double, phases: StagePhases)

    private struct SourceContextContract: Decodable {
        struct Recording: Decodable {
            let name: String
            let sha256: String
            let startSeconds: Double
        }
        let schemaVersion: Int
        let profile: String
        let assetFile: String
        let functionName: String
        let inputShapes: [String: [Int]]
        let outputShapes: [String: [Int]]
        let scalarType: String
        let byteOrder: String
        let sampleRate: Int
        let audioSeconds: Double
        let taskType: String
        let isCovers: Bool
        let posteriorMode: String
        let checkpointDigest: String
        let checkpointFullSha256Verified: Bool
        let hostDistributionSampleParityExact: Bool
        let hostWeightNormFoldingParityExact: Bool
        let recording: Recording
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    static func runSourceContext(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let result = await executeSourceContext(repetitions: 3, progress: progress)
        return result.report
    }

    private static func executeSourceContext(repetitions: Int, window: CoreAIPreviewWindow = .bundled, importedPiano: CoreAIImportedPiano? = nil, progress: @escaping @MainActor @Sendable (String) -> Void) async
        -> (report: String, context: [Float]?) {
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: actual recording -> VAE posterior sample -> full-range lego audio context",
                     "Scope: audio side only; NO text embedding, ConditionEncoder, denoiser or bass generation",
                     "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
                     "Audio: \(window.seconds) s / 48000 Hz stereo; \(window.latentFrames) latent frames",
                     "Posterior: mean + (softplus(scale) + 1e-4) * explicit epsilon; fixed host epsilon reused",
                     importedPiano == nil
                        ? "PCM decode/resampling happens during Mac preparation; fixed-reference input"
                        : "Input: Files import decoded/resampled on iPhone; numerical parity NOT EVALUATED for new recording",
                     "Footprint guard before model and each call: 2000 MiB; NOT an OS peak-memory guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var completed = 0
        var capturedContext: [Float]?
        do {
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates/" + window.sourceBundle),
                  FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append(window == .short512
                    ? "Prepare on Mac: bash scripts/prepare_coreai_source_context.sh; choose your piano recording, then rebuild app."
                    : "Prepare on Mac: bash scripts/prepare_coreai_bass_preview_1024.sh; wait for Ready then rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(SourceContextContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            let shapes = ["audio":[1,2,window.samples],"epsilon":[1,64,window.latentFrames],"chunk_mask":[1,window.latentFrames,64]]
            let outputShapes = ["posterior_mean":[1,64,window.latentFrames],"posterior_std":[1,64,window.latentFrames],
                                "source_latents":[1,window.latentFrames,64],"context_latents":[1,window.latentFrames,128]]
            try checkManifest(ref.schemaVersion, 1, "source_context.schema_version")
            try checkManifest(ref.profile, window.sourceBundle, "source_context.profile")
            try checkManifest(ref.assetFile, "source-context.aimodel", "source_context.asset_file")
            try checkManifest(ref.functionName, "main", "source_context.function_name")
            try checkManifest(ref.inputShapes, shapes, "source_context.input_shapes")
            try checkManifest(ref.outputShapes, outputShapes, "source_context.output_shapes")
            try checkManifest(ref.scalarType, "float32", "source_context.scalar_type")
            try checkManifest(ref.byteOrder, "little", "source_context.byte_order")
            try checkManifest(ref.sampleRate, 48000, "source_context.sample_rate")
            try checkManifest(ref.audioSeconds, window.seconds, "source_context.audio_seconds")
            try checkManifest(ref.taskType, "lego", "source_context.task_type")
            try checkManifest(ref.isCovers, false, "source_context.is_covers")
            try checkManifest(ref.posteriorMode, "sample_with_explicit_epsilon", "source_context.posterior_mode")
            try checkManifest(ref.checkpointDigest, Profile.encoderOneSecond.checkpointDigest, "source_context.checkpoint_digest")
            try checkManifest(ref.checkpointFullSha256Verified, true, "source_context.checkpoint_verified")
            try checkManifest(ref.hostDistributionSampleParityExact, true, "source_context.host_distribution_exact")
            try checkManifest(ref.hostWeightNormFoldingParityExact, true, "source_context.host_weight_norm_exact")
            try checkManifest(ref.absoluteTolerance, 1e-4, "source_context.absolute_tolerance")
            try checkManifest(ref.relativeTolerance, 1e-3, "source_context.relative_tolerance")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "source_context.relative_rmse_tolerance")
            var inputs: [String: NDArray] = [:]
            for (name, shape) in shapes {
                let values: [Float]
                if name == "audio", let importedPiano {
                    try importedPiano.validatePCM(window: window)
                    values = importedPiano.planarPCM
                } else {
                    values = try readFloats(folder: folder, name: "\(name).f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                }
                if name == "chunk_mask" { try checkManifest(values.allSatisfy({ $0 == 1 }), true, "source_context.full_range_mask") }
                inputs[name] = NDArray(scalars: values, shape: shape)
            }
            var expected: [String: [Float]] = [:]
            if let importedPiano {
                lines.append("Recording imported on iPhone: \(importedPiano.metadata.name); segment start seconds: 0")
                lines.append("Original recording SHA-256: \(importedPiano.metadata.originalSHA256)")
                lines.append("Imported planar PCM SHA-256: \(importedPiano.metadata.planarPCMSHA256)")
                lines.append("SHA-256: VERIFIED imported PCM, fixture epsilon and full-range mask; pretrained checkpoint verified by preparation")
                lines.append("No old-recording output references loaded; output shape and finite-value checks remain required")
            } else {
                for (name, shape) in outputShapes {
                    expected[name] = try readFloats(folder: folder, name: "\(name)-reference.f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                }
                lines.append("Recording: \(ref.recording.name); segment start seconds: \(ref.recording.startSeconds)")
                lines.append("Original recording SHA-256 (recorded by Mac preparation): \(ref.recording.sha256)")
                lines.append("SHA-256: VERIFIED all three input and four output-reference binaries; pretrained checkpoint verified by preparation")
            }
            lines.append("Phase: opening source-context model and binding one function")
            saveProgress(lines, profile: .denoiser)
            await progress(lines.joined(separator: "\n"))
            guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
            guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
            let openStart = clock.now
            let probe = try await CoreAIModelProbe(contentsOf: folder.appendingPathComponent(ref.assetFile), options: .cpuOnly)
            let openMs = ms(openStart.duration(to: clock.now))
            var setup = CoreAIModelProbe.StatelessTimings()
            let function = try probe.loadStatelessFunction(functionNamed: ref.functionName, inputs: inputs, timings: &setup)
            defer {
                withExtendedLifetime(function) {}
                withExtendedLifetime(probe) {}
            }
            lines.append("Setup: model-open ms \(openMs); function-bind ms \(setup.functionBindMs); input-validation ms \(setup.inputValidationMs)")
            for iteration in 1...repetitions {
                guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
                lines.append("Before audio-context run \(iteration)/\(repetitions) footprint MiB: \(Double(footprint)/1048576)")
                guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
                let phase = "Phase: audio-context run \(iteration)/\(repetitions) on same actual PCM and epsilon"
                saveProgress(lines + [phase], profile: .denoiser)
                await progress((lines + [phase]).joined(separator: "\n"))
                let result = try await runSourceContextFunction(function: function, inputs: inputs, shapes: outputShapes)
                lines.append("Audio-context run \(iteration): backend-await ms \(result.backendMs); readback+checks ms \(result.copyMs)")
                for name in ["posterior_mean","posterior_std","source_latents","context_latents"] {
                    guard let actual = result.values[name] else { throw ProbeError.invalidOutputTensor }
                    if importedPiano != nil {
                        // runSourceContextFunction has already checked actual shape/type/finiteness.
                        if name == "posterior_std", !actual.allSatisfy({ $0 >= 0 }) { throw ProbeError.invalidOutputTensor }
                        lines.append("Imported audio-context \(name): shape and finite values verified; numerical parity NOT EVALUATED")
                        continue
                    }
                    guard let reference = expected[name] else { throw ProbeError.invalidOutputTensor }
                    let stats = try compare(actual: actual, expected: reference,
                                            absoluteTolerance: ref.absoluteTolerance, relativeTolerance: ref.relativeTolerance,
                                            relativeRmseTolerance: ref.relativeRmseTolerance)
                    lines.append("Audio-context run \(iteration) \(name): max abs error \(stats.maxError); RMSE \(stats.rmse); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                    guard stats.passed else { throw ProbeError.numericalParityFailed }
                }
                capturedContext = result.values["context_latents"]
                completed += 1
                saveProgress(lines, profile: .denoiser)
                await progress(lines.joined(separator: "\n"))
            }
            if importedPiano != nil {
                lines.append("Imported audio-context execution: COMPLETED; numerical parity NOT EVALUATED (no matching host reference)")
            } else {
                lines.append(repetitions == 3 ? "Real-recording audio-context parity: PASS (three runs, all four outputs)" : "Real-recording audio-context parity: PASS (one integrated call, all four outputs)")
            }
        } catch {
            lines.append(importedPiano == nil ? "Real-recording audio-context parity: NOT PASSED" : "Imported audio-context execution: NOT COMPLETED")
            lines.append("Error: \(String(reflecting: error))")
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed audio-context runs: \(completed)/\(repetitions); reference comparisons enabled: \(importedPiano == nil); total probe ms: \(ms(start.duration(to: clock.now)))")
        lines.append("Scope: sampled process footprint only, not total model RAM; no bass/audio output generated")
        saveProgress(lines, profile: .denoiser)
        return (lines.joined(separator: "\n"), completed == repetitions ? capturedContext : nil)
    }

    private static func runSourceContextFunction(function: InferenceFunction, inputs: [String: NDArray], shapes: [String: [Int]]) async throws
        -> (values: [String: [Float]], backendMs: Double, copyMs: Double) {
        let clock = ContinuousClock(); let start = clock.now
        var outputs = try await function.run(inputs: inputs)
        let copyStart = clock.now
        var values: [String: [Float]] = [:]
        for (name, shape) in shapes {
            guard let value = outputs.remove(name), let tensor = value.ndArray else { throw ProbeError.invalidOutputTensor }
            values[name] = try copiedValues(tensor, shape: shape)
        }
        return (values, ms(start.duration(to: copyStart)), ms(copyStart.duration(to: clock.now)))
    }

    private struct BassConditioningStage: Decodable, Equatable {
        let assetFile: String
        let inputShapes: [String: [Int]]
        let outputShapes: [String: [Int]]
        let outputBindings: [String: String]
    }

    private struct BassConditioningContract: Decodable {
        struct HostParity: Decodable { let parityPassed: Bool }
        struct Prompts: Decodable { let instruction: String; let presetSha256: String; let audioSeconds: Double }
        let schemaVersion: Int
        let profile: String
        let functionName: String
        let batchSize: Int
        let textTokens: Int
        let lyricTokens: Int
        let qwenLayerCount: Int
        let lyricLayerCount: Int
        let timbreLayerCount: Int
        let stages: [BassConditioningStage]
        let qwenCheckpointDigest: String
        let baseCheckpointDigest: String
        let silenceDigest: String
        let allCheckpointHashesVerified: Bool
        let scalarType: String
        let byteOrder: String
        let weightStorageDtype: String
        let computeDtype: String
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
        let tokenizationOnDevice: Bool
        let referenceAudioMode: String
        let allValidSingleReferenceSpecialization: Bool
        let hostPartitionVsOriginalFp32: HostParity
        let hostStorage16VsOriginalFp32: HostParity
        let artifactSha256: [String: String]
        let prompts: Prompts
    }

    // Mirrors the fixed host contract independently. Every transformer pair is
    // required once, in order; arbitrary stage omission/reordering is rejected.
    private static func bassConditioningStages(window: CoreAIPreviewWindow = .bundled) -> [BassConditioningStage] {
        let nt = window.textTokens, nl = 11
        var plan: [BassConditioningStage] = []
        func add(_ asset: String, _ inputs: [String: [Int]], _ outputs: [String: [Int]], _ bindings: [String: String]? = nil) {
            let outputBindings = bindings ?? Dictionary(uniqueKeysWithValues: outputs.keys.map { ($0, $0) })
            plan.append(BassConditioningStage(assetFile: asset + ".aimodel", inputShapes: inputs,
                                             outputShapes: outputs, outputBindings: outputBindings))
        }
        add("token-embedding", ["text_ids":[1,nt],"lyric_ids":[1,nl]],
            ["text_hidden":[1,nt,1024],"lyric_hidden":[1,nl,1024]])
        for i in stride(from: 0, to: 28, by: 2) {
            add(String(format: "qwen-%02d-%02d", i, i+1), ["text_hidden":[1,nt,1024]],
                ["hidden_out":[1,nt,1024]], ["hidden_out":"text_hidden"])
        }
        add("qwen-norm", ["text_hidden":[1,nt,1024]], ["text_embeddings":[1,nt,1024]])
        add("lyric-front", ["lyric_hidden":[1,nl,1024]], ["hidden_out":[1,nl,2048]], ["hidden_out":"lyric_hidden"])
        for i in stride(from: 0, to: 8, by: 2) {
            add(String(format: "lyric-%02d-%02d", i, i+1), ["lyric_hidden":[1,nl,2048]],
                ["hidden_out":[1,nl,2048]], ["hidden_out":"lyric_hidden"])
        }
        add("lyric-norm", ["lyric_hidden":[1,nl,2048]], ["lyric_encoded":[1,nl,2048]])
        add("timbre-front", ["reference_audio":[1,750,64]], ["timbre_hidden":[1,750,2048]])
        for i in stride(from: 0, to: 4, by: 2) {
            add(String(format: "timbre-%02d-%02d", i, i+1), ["timbre_hidden":[1,750,2048]],
                ["hidden_out":[1,750,2048]], ["hidden_out":"timbre_hidden"])
        }
        add("timbre-norm", ["timbre_hidden":[1,750,2048]], ["timbre_encoded":[1,1,2048]])
        add("condition-pack", ["text_embeddings":[1,nt,1024],"lyric_encoded":[1,nl,2048],"timbre_encoded":[1,1,2048]],
            ["condition":[1,nt+nl+1,2048],"condition_mask":[1,nt+nl+1]])
        return plan
    }

    static func runBassConditioning(progress: @MainActor @Sendable (String) -> Void) async -> String {
        let result = await executeBassConditioning(progress: progress)
        return result.report
    }

    private static func executeBassConditioning(window: CoreAIPreviewWindow = .bundled, progress: @MainActor @Sendable (String) -> Void) async
        -> (report: String, condition: [Float]?, mask: [Float]?) {
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var lines = [
            "Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "CoreAI architecture: \(AIModel.deviceArchitectureName)",
            "Test: \(window.stemID) prompt -> all 28 Qwen blocks -> complete ACE ConditionEncoder (8 lyric + 4 timbre blocks)",
            "Scope: text/target conditioning; fixed host-tokenized prompt; NO denoiser or bass waveform",
            "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
            "27 sequential native assets; \(window.weightStorageDtype) stored weights, FP32 calculations; detached CPU copies between assets",
            "Prompt tokenization on Mac; arbitrary prompt entry/tokenization on iPhone NOT tested",
            "No timbre reference recording: publisher silence latent, first 750 frames; single batch with all tokens valid",
            "Output: condition [1,\(window.conditionTokens),2048], mask [1,\(window.conditionTokens)]; duration prompt \(window.seconds) s; source audio context tested separately",
            "Footprint guard before each stage: 2000 MiB; NOT an OS peak-memory guarantee"
        ]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        var completed = 0, checks = 0
        var capturedCondition: [Float]?, capturedMask: [Float]?
        do {
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates/" + window.conditionBundle),
                  FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append(window == .short512
                    ? "Prepare on Mac: bash scripts/prepare_coreai_bass_conditioning.sh; wait for Ready, then rebuild app."
                    : "Prepare on Mac: bash scripts/prepare_coreai_bass_preview_1024.sh; wait for Ready then rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(BassConditioningContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion, 1, "conditioning.schema_version")
            try checkManifest(ref.profile, window.conditionBundle, "conditioning.profile")
            try checkManifest(ref.functionName, "main", "conditioning.function")
            try checkManifest(ref.batchSize, 1, "conditioning.batch")
            try checkManifest(ref.textTokens, window.textTokens, "conditioning.text_tokens")
            try checkManifest(ref.lyricTokens, 11, "conditioning.lyric_tokens")
            try checkManifest(ref.qwenLayerCount, 28, "conditioning.qwen_layers")
            try checkManifest(ref.lyricLayerCount, 8, "conditioning.lyric_layers")
            try checkManifest(ref.timbreLayerCount, 4, "conditioning.timbre_layers")
            try checkManifest(ref.stages, bassConditioningStages(window: window), "conditioning.ordered_stages")
            try checkManifest(ref.qwenCheckpointDigest, "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd", "conditioning.qwen_digest")
            try checkManifest(ref.baseCheckpointDigest, "4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b", "conditioning.base_digest")
            try checkManifest(ref.silenceDigest, "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b", "conditioning.silence_digest")
            try checkManifest(ref.allCheckpointHashesVerified, true, "conditioning.checkpoints_verified")
            try checkManifest(ref.scalarType, "float32", "conditioning.scalar")
            try checkManifest(ref.byteOrder, "little", "conditioning.byte_order")
            try checkManifest(ref.weightStorageDtype, window.weightStorageDtype, "conditioning.storage")
            try checkManifest(ref.computeDtype, "float32", "conditioning.compute")
            try checkManifest(ref.absoluteTolerance, 1e-4, "conditioning.abs_tolerance")
            try checkManifest(ref.relativeTolerance, 1e-3, "conditioning.rel_tolerance")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "conditioning.rmse_tolerance")
            try checkManifest(ref.tokenizationOnDevice, false, "conditioning.host_tokenization")
            try checkManifest(ref.referenceAudioMode, "publisher_silence_750", "conditioning.reference_mode")
            try checkManifest(ref.allValidSingleReferenceSpecialization, true, "conditioning.single_reference")
            try checkManifest(ref.hostPartitionVsOriginalFp32.parityPassed, true, "conditioning.partition_gold")
            try checkManifest(ref.hostStorage16VsOriginalFp32.parityPassed, true, "conditioning.storage_gold")
            try checkManifest(ref.prompts.instruction, window.instruction, "conditioning.instruction")
            try checkManifest(ref.prompts.audioSeconds, window.seconds, "conditioning.clip_seconds")
            try checkManifest(ref.prompts.presetSha256, window.presetSHA256, "conditioning.preset")
            var state: [String: NDArray] = [:]
            let inputs = ["text_ids":[1,window.textTokens],"lyric_ids":[1,11],"reference_audio":[1,750,64]]
            for (name, shape) in inputs {
                let values = try readFloats(folder: folder, name: "\(name).f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                if name.hasSuffix("_ids") {
                    guard values.allSatisfy({ $0 >= 0 && $0 < 151669 && $0.rounded(.towardZero) == $0 }) else { throw ProbeError.invalidFloatData }
                }
                state[name] = NDArray(scalars: values, shape: shape)
            }
            let milestones = ["text_embeddings":[1,window.textTokens,1024],"condition":[1,window.conditionTokens,2048],"condition_mask":[1,window.conditionTokens]]
            var references: [String: [Float]] = [:]
            for (name, shape) in milestones {
                for kind in ["reference","fp32-reference"] {
                    references["\(name)-\(kind)"] = try readFloats(folder: folder, name: "\(name)-\(kind).f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                }
            }
            lines.append("SHA-256: VERIFIED three inputs and six references; all checkpoint hashes verified during preparation")
            for (index, stage) in ref.stages.enumerated() {
                guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
                guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
                let phase = "Phase: conditioning stage \(index+1)/27 \(stage.assetFile); footprint MiB \(Double(footprint)/1048576)"
                saveProgress(lines + [phase], profile: .denoiser); await progress((lines + [phase]).joined(separator: "\n"))
                var stageInputs: [String: NDArray] = [:]
                for (name, shape) in stage.inputShapes {
                    guard let array = state[name], array.shape == shape else { throw ShardError.invalidManifest }
                    stageInputs[name] = array
                }
                let result = try await runBassConditioningStage(folder: folder, stage: stage, inputs: stageInputs)
                lines.append("Stage \(index+1)/27 \(stage.assetFile): load+bind ms \(result.loadMs); backend-await ms \(result.backendMs); readback ms \(result.copyMs)")
                for (name, values) in result.values {
                    guard let shape = stage.outputShapes[name], let binding = stage.outputBindings[name] else { throw ShardError.invalidManifest }
                    state[binding] = NDArray(scalars: values, shape: shape)
                    if name == "condition" { capturedCondition = values }
                    if name == "condition_mask" { capturedMask = values }
                    if milestones[name] != nil {
                        if name == "condition_mask" { try checkManifest(values.allSatisfy({ $0 == 1 }), true, "conditioning.full_valid_mask") }
                        for kind in ["reference","fp32-reference"] {
                            guard let expected = references["\(name)-\(kind)"] else { throw ShardError.invalidManifest }
                            let stats = try compare(actual: values, expected: expected, absoluteTolerance: ref.absoluteTolerance,
                                                    relativeTolerance: ref.relativeTolerance, relativeRmseTolerance: ref.relativeRmseTolerance)
                            lines.append("\(name) vs \(kind): max abs error \(stats.maxError); RMSE \(stats.rmse); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                            guard stats.passed else { throw ProbeError.numericalParityFailed }
                            checks += 1
                        }
                    }
                }
                completed += 1
            }
            guard completed == 27, checks == 6 else { throw ShardError.invalidManifest }
            lines.append("\(window.stemID == "bass" ? "Bass" : window.stemID) text/ConditionEncoder dual parity: PASS (all 40 layers, six checks)")
        } catch {
            lines.append("\(window.stemID == "bass" ? "Bass" : window.stemID) text/ConditionEncoder dual parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed conditioning stages: \(completed)/27; parity checks: \(checks)/6; total probe ms: \(ms(start.duration(to: clock.now)))")
        lines.append("Scope: sampled app process only; total model/device RAM, native cache eviction and raw recording import unmeasured; no bass generated")
        saveProgress(lines, profile: .denoiser)
        return (lines.joined(separator: "\n"), completed == 27 && checks == 6 ? capturedCondition : nil, completed == 27 && checks == 6 ? capturedMask : nil)
    }

    private static func runBassConditioningStage(folder: URL, stage: BassConditioningStage, inputs: [String: NDArray]) async throws
        -> (values: [String: [Float]], loadMs: Double, backendMs: Double, copyMs: Double) {
        let clock = ContinuousClock(); let start = clock.now
        let probe = try await CoreAIModelProbe(contentsOf: folder.appendingPathComponent(stage.assetFile), options: .cpuOnly)
        var timings = CoreAIModelProbe.StatelessTimings()
        let function = try probe.loadStatelessFunction(functionNamed: "main", inputs: inputs, timings: &timings)
        let loadMs = ms(start.duration(to: clock.now))
        defer { withExtendedLifetime(function) {}; withExtendedLifetime(probe) {} }
        try requirePreviewMemoryBudget()
        let result = try await runSourceContextFunction(function: function, inputs: inputs, shapes: stage.outputShapes)
        // Return detached scalar arrays; no native output/model/function escapes.
        return (result.values, loadMs, result.backendMs, result.copyMs)
    }

    struct BassPreviewResult: Sendable {
        let report: String
        let bassURL: URL?
        let mixURL: URL?
    }

    private enum BassPreviewError: Error {
        case componentNotCompleted(String)
    }

    private struct BassPreviewContract: Decodable {
        struct HostComparison: Decodable {
            let finite: Bool
            let maxAbsoluteError: Double
            let rmse: Double
            let normalizedRmse: Double
            let elementwiseFractionOverBound: Double
            let parityPassed: Bool
        }
        struct HostStep: Decodable { let step: Int; let latent: HostComparison }
        struct DenoiserSameInput: Decodable {
            let step: Int
            let phase: String
            let timestep: Double
            let inputLatentFile: String
            let inputLatentSha256: String
            let conditioningInput: String
            let comparison: HostComparison
        }
        let schemaVersion: Int
        let profile: String
        let functionName: String
        let stages: [Contract.Stage]
        let inferenceSteps: Int
        let guidanceScale: Double
        let shift: Double
        let method: String
        let apgMomentum: Double
        let apgNormThreshold: Double
        let apgProjectionDims: [Int]
        let apgProjectionPrecision: String
        let referenceConditioningScope: String?
        let weightStorageDtype: String?
        let checkpoints: [Int]
        let latentFrames: Int
        let conditionTokens: Int
        let sampleRate: Int
        let audioSeconds: Double
        let scalarType: String
        let byteOrder: String
        let checkpointDigest: String
        let vaeCheckpointDigest: String
        let allCheckpointsVerified: Bool
        let hostDualTrajectoryParityPassed: Bool
        let hostExportStatus: String?
        let hostNumericalGatePassed: Bool?
        let hostComparisons: [HostStep]
        let hostTrajectoryDiagnosticRequested: Bool?
        let hostDenoiserSameInput: [DenoiserSameInput]?
        let apgSameInputVsPublisher: [HostComparison]
        let hostAudioVsFp32: HostComparison
        let hostDecoderFoldingExact: Bool
        let sourceReferenceSha256: String
        let conditionReferenceSha256: String
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    private static func bassPreviewStages(window: CoreAIPreviewWindow = .short512) -> [Contract.Stage] {
        let shared = ["hidden":[2,window.hiddenTokens,2048],"condition":[2,window.conditionTokens,2048],"temb":[2,6,2048],
                      "rope_cos":[1,window.hiddenTokens,128],"rope_sin":[1,window.hiddenTokens,128],"self_mask":[1,1,window.hiddenTokens,window.hiddenTokens],
                      "local_mask":[1,1,window.hiddenTokens,window.hiddenTokens],"cross_mask":[1,1,window.hiddenTokens,window.conditionTokens]]
        var frontOutputs = shared
        frontOutputs.removeValue(forKey: "condition")
        frontOutputs["condition_projected"] = [2,window.conditionTokens,2048]
        frontOutputs["final_temb"] = [2,2048]
        var frontBindings = Dictionary(uniqueKeysWithValues: frontOutputs.keys.map { ($0, $0) })
        frontBindings["condition_projected"] = "condition"
        var result = [Contract.Stage(assetFile: "front.aimodel", layerIndices: [],
            inputShapes: ["noise":[1,window.latentFrames,64],"condition":[1,window.conditionTokens,2048],"context":[1,window.latentFrames,128],"timestep":[1]],
            outputShapes: frontOutputs, outputBindings: frontBindings)]
        for i in stride(from: 0, to: 24, by: 2) {
            result.append(Contract.Stage(assetFile: String(format: "blocks-%02d-%02d.aimodel", i, i+1), layerIndices: [i,i+1],
                inputShapes: shared, outputShapes: ["hidden_out":[2,window.hiddenTokens,2048]], outputBindings: ["hidden_out":"hidden"]))
        }
        result.append(Contract.Stage(assetFile: "tail.aimodel", layerIndices: [],
            inputShapes: ["hidden":[2,window.hiddenTokens,2048],"final_temb":[2,2048]],
            outputShapes: ["velocity":[2,window.latentFrames,64]], outputBindings: ["velocity":"velocity"]))
        return result
    }

    private static func requirePreviewMemoryBudget() throws {
        guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
        guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
    }

    private static func verifyReferenceIdentity(_ url: URL, expected: String) throws {
        let data = try Data(contentsOf: url)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try checkManifest(digest, expected, "preview.component_reference_sha256")
    }

    private struct BassPreviewAOTContract: Decodable {
        let schemaVersion: Int
        let profile: String
        let architecture: String
        let platform: String
        let minDeploymentVersion: String
        let preferredCompute: String
        let runtimePolicy: String
        let sourceReferenceSha256: String
        let compiledAssetCount: Int
        let assets: [String: String]
        let artifactSha256: [String: String]
    }

    private struct PreviewAOTSelection: Sendable {
        let folder: URL
        let assets: [String: String]
    }

    static func streamingSHA256(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while true {
            // Bound Foundation temporary objects to one chunk, even when a
            // multi-gigabyte compiled payload is hashed in a detached task.
            let hasData = try autoreleasepool { () throws -> Bool in
                guard let chunk = try file.read(upToCount: 4*1024*1024), !chunk.isEmpty else { return false }
                digest.update(data: chunk)
                return true
            }
            if !hasData { break }
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // Foundation DirectoryEnumerator's Sequence iterator is unavailable in
    // async contexts. Finish enumeration synchronously and return owned values
    // before the AOT verifier suspends to report progress.
    private static func aotPayloadFiles(in folder: URL) throws -> Set<String> {
        let root = folder.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        var actualFiles = Set<String>()
        guard let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey]) else {
            throw ShardError.invalidManifest
        }
        for case let url as URL in files {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true, url.lastPathComponent != ".DS_Store", url != folder.appendingPathComponent("reference.json") else { continue }
            let path = url.standardizedFileURL.resolvingSymlinksInPath().path
            guard path.hasPrefix(root) else { throw ShardError.invalidManifest }
            actualFiles.insert(String(path.dropFirst(root.count)))
        }
        return actualFiles
    }

    private static func preparedBassPreviewAOT(source: URL, reportLines: [String],
                                               progress: @escaping @MainActor @Sendable (String) -> Void) async throws -> PreviewAOTSelection? {
        let folder = source.deletingLastPathComponent().appendingPathComponent("bass-preview-512-aot")
        guard FileManager.default.fileExists(atPath: folder.path) else { return nil }
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let ref = try decoder.decode(BassPreviewAOTContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
        let architecture = AIModel.deviceArchitectureName
        try checkManifest(ref.schemaVersion, 1, "aot.schema")
        try checkManifest(ref.profile, "bass-preview-512-aot", "aot.profile")
        try checkManifest(ref.architecture, architecture, "aot.device_architecture")
        try checkManifest(ref.platform, "iOS", "aot.platform")
        try checkManifest(ref.minDeploymentVersion, "27.0", "aot.deployment")
        try checkManifest(ref.preferredCompute, "none", "aot.compiler_preference")
        try checkManifest(ref.runtimePolicy, "default", "aot.runtime_policy")
        try checkManifest(ref.compiledAssetCount, 16, "aot.asset_count")
        let names = bassPreviewStages().map { $0.assetFile } + ["apg-euler.aimodel","decoder.aimodel"]
        let expected = Dictionary(uniqueKeysWithValues: names.map { name in
            (name, (name as NSString).deletingPathExtension + "." + architecture + ".aimodelc")
        })
        try checkManifest(ref.assets, expected, "aot.variant_names")
        try verifyReferenceIdentity(source.appendingPathComponent("reference.json"), expected: ref.sourceReferenceSha256)
        guard !ref.artifactSha256.isEmpty else { throw ShardError.invalidManifest }
        let root = folder.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        let actualFiles = try aotPayloadFiles(in: folder)
        try checkManifest(actualFiles, Set(ref.artifactSha256.keys), "aot.covered_files")
        guard ref.artifactSha256.keys.allSatisfy({ name in
            ref.assets.values.contains(where: { name == $0 || name.hasPrefix($0 + "/") })
        }) else { throw ShardError.invalidManifest }
        let clock = ContinuousClock(); let verificationStart = clock.now
        let orderedAssets = names.compactMap { ref.assets[$0] }
        for (index, asset) in orderedAssets.enumerated() {
            let payloads = ref.artifactSha256.keys.filter { $0 == asset || $0.hasPrefix(asset + "/") }.sorted()
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent(asset).path), !payloads.isEmpty else {
                throw ShardError.invalidManifest
            }
            let status = "AOT SHA-256 \(index + 1)/16: \(asset); reading payloads; elapsed ms \(ms(verificationStart.duration(to: clock.now)))"
            saveProgress(reportLines + [status], profile: .denoiser)
            await progress((reportLines + [status]).joined(separator: "\n"))
            for name in payloads {
                let url = folder.appendingPathComponent(name).standardizedFileURL.resolvingSymlinksInPath()
                guard url.path.hasPrefix(root), let expectedDigest = ref.artifactSha256[name] else { throw ShardError.invalidManifest }
                try checkManifest(try streamingSHA256(url), expectedDigest, "aot.artifact_sha256: \(name)")
            }
        }
        let status = "AOT SHA-256 VERIFIED 16/16; verification ms \(ms(verificationStart.duration(to: clock.now)))"
        saveProgress(reportLines + [status], profile: .denoiser)
        await progress((reportLines + [status]).joined(separator: "\n"))
        return PreviewAOTSelection(folder: folder, assets: ref.assets)
    }

    // Persist immediately before each native API interval: EXC_BAD_ACCESS
    // terminates the process and cannot be recovered by a Swift catch block.
    private static func previewPhaseProgress(prefix: String) -> @Sendable (String) -> Void {
        return { phase in
            let report = prefix + "\n" + phase
            saveProgress([report], profile: .denoiser)
            print(phase)
        }
    }

    static func generateBassPreview(window: CoreAIPreviewWindow = .bundled, importedPiano: CoreAIImportedPiano? = nil,
        diagnosticThreeSteps: Bool = false, progress: @escaping @MainActor @Sendable (String) -> Void) async -> BassPreviewResult {
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        let priorReport = savedDenoiserReport()
        let referenceMode = importedPiano == nil
        let expectedOutputChecks = referenceMode ? 10 : 0
        let reuseOverride = ProcessInfo.processInfo.environment["COREAI_PREVIEW_REUSE"]
        let aotFolderPresent = Bundle.main.resourceURL.map {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("ModelCandidates/bass-preview-512-aot").path)
        } ?? false
        // The 5.12-second source CPU case is device-validated with retained handles.
        // AOT remains one-scope unless explicitly overridden (then rejected below).
        let reuseSamplingStages = reuseOverride == "1" || (reuseOverride != "0" && !aotFolderPresent)
        // The long profile failed in BNNS compilation at the third asset while
        // earlier handles were retained. Complete its first actual Euler step
        // with one Swift model/function scope at a time, then retain from step 2.
        // This is not a hidden warmup or a guarantee of native cache eviction.
        let sequentialFirstStep = reuseSamplingStages && window.isLong
        var samplingPool: [String: RetainedStage] = [:]
        defer { samplingPool.removeAll() }
        var retention = RetentionStats()
        var cacheRepair = CacheRepairStats()
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "Probe run: \(UUID().uuidString); started UTC: \(ISO8601DateFormatter().string(from: Date()))",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "App build configuration: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeBuildConfiguration") as? String ?? "unrecorded")",
                     "Swift app optimization: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeSwiftOptimizationLevel") as? String ?? "unrecorded"); native backend optimization is separate",
                     diagnosticThreeSteps
                        ? "Test: actual piano -> VAE posterior -> full Qwen/ConditionEncoder -> first 3 actual APG-guided Euler steps; NO decoder/WAV"
                        : "Test: actual piano -> VAE posterior -> full Qwen/ConditionEncoder -> 50 APG-guided Euler steps -> \(window.stemID) WAV",
                     "Audio: \(window.seconds) s / 48000 Hz stereo; \(window.latentFrames) latent frames / \(window.conditionTokens) condition tokens",
                     "Guidance 7; APG momentum -0.75, threshold 2.5, projection over latent time (dim 1); shift 1",
                     "APG projection FP32, compared on host against original FP64 projection; not plain CFG",
                     "Requested compute policy: source/Qwen cpuOnly; sampling/decoder policy below; accelerator placement NOT MEASURED",
                     referenceMode
                        ? "Input mode: fixed host PCM regression; host tokenization/epsilon/noise fixtures; arbitrary prompts and on-device RNG NOT tested"
                        : "Input mode: iPhone Files import; on-device PCM decode/resampling; fixed \(window.stemID) prompt and host epsilon/noise fixtures; arbitrary prompts and on-device RNG NOT tested",
                     "Actual component outputs feed sampling; host references are comparisons only",
                     reuseSamplingStages
                        ? (sequentialFirstStep
                            ? "Execution: first actual sampling step uses one Swift model/function scope per asset; retain at most 15 source CPU handles from step 2; release before decoder; native residency/eviction NOT GUARANTEED"
                            : "Execution: retain at most 15 source CPU sampling model/function handles across steps; release before decoder; native residency/eviction NOT GUARANTEED")
                        : "Execution: single live Swift model/function scope per asset; detached CPU copies; native cache eviction NOT GUARANTEED",
                     window.isLong
                        ? "Raw-source cache repair: at most ONE targeted CPU sampling cache deletion and bind retry per generation, only for BNNS graphLoadFailed before inference; no step replay or cache-wide deletion"
                        : "Raw-source cache repair: only blocks-18-19 after its saved BNNS graphLoadFailed; skipped for AOT; no automatic retry",
                     "Native API phase markers are persisted before calls; observer overhead is included in total pipeline time",
                     "Footprint guard 2000 MiB before each stage and after bind; NOT an OS peak-memory guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        #if DEBUG
        lines.append("Swift DEBUG compilation condition: ON")
        #else
        lines.append("Swift DEBUG compilation condition: OFF")
        #endif
        lines.append("Preview profile: \(window.previewBundle); source \(window.sourceBundle); conditioning \(window.conditionBundle)")
        lines.append("Sampling handle reuse requested: \(reuseSamplingStages)")
        lines.append("Sampling reuse selection: \(reuseOverride == nil ? "app default for selected resources" : "launch override")")
        var lastPhase = "preview manifest/reference validation"
        var completedSteps = 0, checks = 0, allParity = true
        var bassURL: URL?, mixURL: URL?
        let plannedSteps = diagnosticThreeSteps ? 3 : 50
        let runHeader = lines[2]
        lines.append("Selected instrument: \(window.stemID); preset SHA-256 \(window.presetSHA256); text/condition tokens \(window.textTokens)/\(window.conditionTokens)")
        let boundedSamplingProgress = window.isLong
        let samplingBreadcrumb = diagnosticThreeSteps ? "last-sampling-diagnostic-phase.txt" : "last-\(window.stemID)-sampling-phase.txt"
        if boundedSamplingProgress && !diagnosticThreeSteps {
            lines.append("Instrumentation revision: generation-bounded-v1; short sampling/decoder live progress; separate native breadcrumb; complete report checkpointed after each step")
        }
        if diagnosticThreeSteps {
            lines.append("Actual-input sampling diagnostic: THREE-STEP PREFIX ONLY; no decoder or WAV; production remains 50 steps")
            lines.append("Instrumentation revision: actual-prefix-bounded-v1; short live progress, separate native breadcrumb, full checkpoint after each completed step")
            lines.append("Scope: actual imported piano, source/conditioning, front + 12 pairs + tail + APG, actual latent/momentum feedback; no matching host output reference, numerical parity NOT EVALUATED")
            lines.append("Diagnostic thermal before: \(ProcessInfo.processInfo.thermalState.rawValue); low-power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)")
        }
        do {
            if diagnosticThreeSteps {
                guard window.isLong, importedPiano != nil, reuseSamplingStages else { throw ShardError.invalidManifest }
            }
            if let importedPiano {
                try importedPiano.validatePCM(window: window)
                lines.append("Imported recording: \(importedPiano.metadata.name); first \(window.seconds) s, no input normalization")
                lines.append("Original recording SHA-256: \(importedPiano.metadata.originalSHA256)")
                lines.append("Imported planar PCM SHA-256: \(importedPiano.metadata.planarPCMSHA256)")
                lines.append("Generation numerical parity: NOT EVALUATED; no matching host trajectory/waveform for this input")
                lines.append("Fixed text/timbre preset conditioning still uses its own six matching reference checks")
            }
            guard let resources = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let folder = resources.appendingPathComponent("ModelCandidates/" + window.previewBundle)
            let sourceFolder = resources.appendingPathComponent("ModelCandidates/" + window.sourceBundle)
            let conditionFolder = resources.appendingPathComponent("ModelCandidates/" + window.conditionBundle)
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append(window == .short512
                    ? "Prepare on Mac: bash scripts/prepare_coreai_bass_preview.sh; wait for Ready then rebuild app."
                    : "Prepare on Mac: bash scripts/prepare_coreai_bass_preview_1024.sh; wait for Ready then rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(BassPreviewContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion, 1, "preview.schema")
            try checkManifest(ref.profile, window.previewBundle, "preview.profile")
            try checkManifest(ref.functionName, "main", "preview.function")
            try checkManifest(ref.inferenceSteps, 50, "preview.steps")
            try checkManifest(ref.guidanceScale, 7, "preview.guidance")
            try checkManifest(ref.shift, 1, "preview.shift")
            try checkManifest(ref.method, "euler_ode_apg", "preview.method")
            try checkManifest(ref.apgMomentum, -0.75, "preview.momentum")
            try checkManifest(ref.apgNormThreshold, 2.5, "preview.threshold")
            try checkManifest(ref.apgProjectionDims, [1], "preview.projection_dims")
            try checkManifest(ref.apgProjectionPrecision, window.apgProjectionPrecision, "preview.projection_precision")
            if window.isLong {
                try checkManifest(ref.referenceConditioningScope, Optional("original_fp32_vs_storage16_conditioning"), "preview.reference_conditioning_scope")
                try checkManifest(ref.weightStorageDtype, Optional(window.weightStorageDtype), "preview.weight_storage_dtype")
                lines.append("Long-window storage: BF16 constants preserve original checkpoint values; calculations remain FP32; full original FP32 conditioning/denoiser reference retained")
            }
            try checkManifest(ref.checkpoints, [1,10,25,50], "preview.checkpoints")
            try checkManifest(ref.latentFrames, window.latentFrames, "preview.frames")
            try checkManifest(ref.conditionTokens, window.conditionTokens, "preview.condition_tokens")
            try checkManifest(ref.sampleRate, 48000, "preview.sample_rate")
            try checkManifest(ref.audioSeconds, window.seconds, "preview.seconds")
            try checkManifest(ref.scalarType, "float32", "preview.dtype")
            try checkManifest(ref.byteOrder, "little", "preview.byte_order")
            try checkManifest(ref.checkpointDigest, "4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b", "preview.base_digest")
            try checkManifest(ref.vaeCheckpointDigest, Profile.encoderOneSecond.checkpointDigest, "preview.vae_digest")
            try checkManifest(ref.allCheckpointsVerified, true, "preview.checkpoints_verified")
            try checkManifest(ref.hostComparisons.map { $0.step }, Array(1...50), "preview.host_steps")
            try checkManifest(ref.apgSameInputVsPublisher.count, 50, "preview.host_apg_count")
            let trajectoryDiagnostic = ref.hostTrajectoryDiagnosticRequested ?? false
            try checkManifest(!trajectoryDiagnostic || (window.isLong && CoreAIPreviewWindow.trajectoryDiagnosticEnabled), true, "preview.explicit_trajectory_diagnostic_selection")
            func boundedHostComparison(_ stats: BassPreviewContract.HostComparison, allowTrajectoryDrift: Bool = false) throws -> Bool {
                guard stats.finite, stats.maxAbsoluteError.isFinite, stats.maxAbsoluteError >= 0,
                      stats.rmse.isFinite, stats.rmse >= 0, stats.normalizedRmse.isFinite,
                      stats.normalizedRmse >= 0,
                      stats.elementwiseFractionOverBound.isFinite, stats.elementwiseFractionOverBound >= 0,
                      stats.elementwiseFractionOverBound <= 1 else { throw ShardError.invalidManifest }
                if !allowTrajectoryDrift && stats.normalizedRmse > 1e-3 { throw ShardError.invalidManifest }
                let strict = stats.elementwiseFractionOverBound == 0 && stats.normalizedRmse <= 1e-3
                try checkManifest(stats.parityPassed, strict, "preview.host_comparison_flag")
                return strict
            }
            var hostTrajectoryPassed = true
            for step in ref.hostComparisons {
                let passed = try boundedHostComparison(step.latent, allowTrajectoryDrift: trajectoryDiagnostic)
                hostTrajectoryPassed = hostTrajectoryPassed && passed
            }
            for stats in ref.apgSameInputVsPublisher {
                try checkManifest(try boundedHostComparison(stats), true, "preview.host_apg_strict")
            }
            let hostAudioPassed = try boundedHostComparison(ref.hostAudioVsFp32, allowTrajectoryDrift: trajectoryDiagnostic)
            if trajectoryDiagnostic {
                guard let cases = ref.hostDenoiserSameInput, cases.map({ $0.step }) == [1,10,25,50] else { throw ShardError.invalidManifest }
                for row in cases {
                    try checkManifest(row.phase, "after_gold_step", "preview.denoiser_check_phase")
                    try checkManifest(row.inputLatentFile, "step-\(row.step)-latent-fp32-reference.f32le", "preview.denoiser_check_input")
                    try checkManifest(ref.artifactSha256[row.inputLatentFile], Optional(row.inputLatentSha256), "preview.denoiser_check_input_hash")
                    try checkManifest(row.conditioningInput, "original_fp32_condition_reference", "preview.denoiser_check_condition")
                    guard row.timestep.isFinite && row.timestep >= 0 && row.timestep <= 1 else { throw ShardError.invalidManifest }
                    try checkManifest(try boundedHostComparison(row.comparison), true, "preview.same_input_denoiser_strict")
                }
                lines.append("Explicit trajectory diagnostic: four identical-input full-denoiser checks and 50 identical-input APG checks must pass strict original bounds; free-trajectory failures remain failures")
            }
            let hostStrict = hostTrajectoryPassed && hostAudioPassed
            try checkManifest(ref.hostDualTrajectoryParityPassed, hostTrajectoryPassed, "preview.host_trajectory")
            if let claimed = ref.hostNumericalGatePassed {
                try checkManifest(claimed, hostStrict, "preview.host_numerical_gate")
            } else if !hostStrict { throw ShardError.invalidManifest }
            let exceedsTrajectoryBound = ref.hostComparisons.contains { $0.latent.normalizedRmse > 1e-3 } || ref.hostAudioVsFp32.normalizedRmse > 1e-3
            let hostStatus = trajectoryDiagnostic && exceedsTrajectoryBound ? "trajectory_drift_diagnostic" : (hostStrict ? "strict_pass" : "bounded_diagnostic")
            if let claimed = ref.hostExportStatus {
                try checkManifest(claimed, hostStatus, "preview.host_export_status")
            } else if !hostStrict { throw ShardError.invalidManifest }
            allParity = referenceMode && hostStrict
            lines.append("Baseline export provenance: \(hostStatus); original tolerances unchanged; applies to fixed prepared input only")
            if referenceMode { lines.append("Host preparation: \(hostStatus); strict gate \(hostStrict ? "PASS" : "NOT PASSED"); original tolerances unchanged") }
            if !hostStrict && hostStatus == "trajectory_drift_diagnostic" {
                lines.append("Host free-trajectory error exceeds original normalized-RMSE limit 0.001; full numerical gate NOT PASSED. Explicit component-validated experiment only; no automatic fallback or PASS relabeling.")
            } else if referenceMode && !hostStrict {
                lines.append("Bounded diagnostic bundle: finite, normalized RMSE <= 0.001; pointwise gate failed. Result cannot be labeled overall numerical PASS.")
            }
            try checkManifest(ref.hostDecoderFoldingExact, true, "preview.decoder_fold")
            try checkManifest(ref.absoluteTolerance, 1e-4, "preview.abs")
            try checkManifest(ref.relativeTolerance, 1e-3, "preview.rel")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "preview.nrmse")
            let expectedStages = bassPreviewStages(window: window)
            try checkManifest(ref.stages.count, expectedStages.count, "preview.stage_count")
            for (actual, expected) in zip(ref.stages, expectedStages) {
                try checkManifest(actual.assetFile, expected.assetFile, "preview.asset_order")
                try checkManifest(actual.layerIndices, expected.layerIndices, "preview.layer_order")
                try checkManifest(actual.inputShapes, expected.inputShapes, "preview.input_shapes")
                try checkManifest(actual.outputShapes, expected.outputShapes, "preview.output_shapes")
                try checkManifest(actual.outputBindings, expected.outputBindings, "preview.output_bindings")
            }
            try verifyReferenceIdentity(sourceFolder.appendingPathComponent("reference.json"), expected: ref.sourceReferenceSha256)
            try verifyReferenceIdentity(conditionFolder.appendingPathComponent("reference.json"), expected: ref.conditionReferenceSha256)
            lastPhase = "verifying optional architecture-matched AOT bundle"
            saveProgress(lines + [lastPhase], profile: .denoiser)
            await progress((lines + [lastPhase]).joined(separator: "\n"))
            let preparedAOT: PreviewAOTSelection?
            if window == .short512 {
                preparedAOT = try await preparedBassPreviewAOT(source: folder, reportLines: lines + [lastPhase], progress: progress)
            } else {
                preparedAOT = nil
                lines.append("10.24 s profile: source CPU only; AOT is not selected")
            }
            try checkManifest(!reuseSamplingStages || preparedAOT == nil, true, "retained_preview_requires_source_cpuOnly")
            if let preparedAOT {
                lines.append("Sampling/decoder assets: AOT .aimodelc; architecture \(AIModel.deviceArchitectureName); 16 compiled assets SHA-256 VERIFIED")
                lines.append("Sampling/decoder requested policy: default, matching compiler --preferred-compute none; physical accelerator NOT MEASURED")
                lines.append("AOT can still specialize on device; reduced compiler memory/time NOT YET MEASURED")
                lines.append("AOT source identity matched: \(preparedAOT.folder.lastPathComponent)")
            } else {
                lines.append("Sampling/decoder assets: source .aimodel; requested policy cpuOnly; device compilation required")
            }
            let noise = try readFloats(folder: folder, name: "initial-noise.f32le", count: window.latentFrames*64, hashes: ref.artifactSha256)
            let times = try readFloats(folder: folder, name: "timesteps.f32le", count: 51, hashes: ref.artifactSha256)
            try checkManifest(times[0], Float(1), "preview.time_start")
            try checkManifest(times[50], Float(0), "preview.time_end")
            if trajectoryDiagnostic, let cases = ref.hostDenoiserSameInput {
                for row in cases { try checkManifest(row.timestep, Double(times[row.step]), "preview.same_input_denoiser_time") }
            }
            guard zip(times, times.dropFirst()).allSatisfy({ $0.0 > $0.1 }) else { throw ShardError.invalidManifest }
            // Only the regression input has matching host output references.
            var references: [String: [Float]] = [:]
            if referenceMode {
                for step in ref.checkpoints {
                    for name in ["latent"] {
                        for kind in ["reference","fp32-reference"] {
                            let file = "step-\(step)-\(name)-\(kind).f32le"
                            references[file] = try readFloats(folder: folder, name: file, count: window.latentFrames*64, hashes: ref.artifactSha256)
                        }
                    }
                }
                for kind in ["reference","fp32-reference"] {
                    let file = "audio-\(kind).f32le"
                    references[file] = try readFloats(folder: folder, name: file, count: 2*window.samples, hashes: ref.artifactSha256)
                }
            }
            lines.append(referenceMode
                ? "SHA-256: VERIFIED component identities, initial noise, schedule and 10 output references"
                : "SHA-256: VERIFIED component identities, initial noise and schedule; no old-recording output references loaded")
            // Repair only the observed failing derived CPU cache entry, before
            // any model is opened. Bundled assets/weights/references are untouched.
            // The saved report is captured before component progress overwrites it.
            if window == .short512, case .none = preparedAOT, let priorReport,
               priorReport.contains("Failing asset: blocks-18-19.aimodel"),
               priorReport.contains("BNNSCoreAIDelegate.BNNSDelegate.BNNSError.graphLoadFailed") {
                lastPhase = "targeted compiled-cache recovery for blocks-18-19.aimodel"
                lines.append("Cache recovery requested by prior failure: blocks-18-19.aimodel / cpuOnly; derived cache only")
                saveProgress(lines, profile: .denoiser); await progress(lines.joined(separator: "\n"))
                do {
                    try AIModelCache.default.deleteEntry(for: folder.appendingPathComponent("blocks-18-19.aimodel"), options: .cpuOnly)
                    lines.append("Targeted compiled-cache removal: completed; asset will specialize again when needed")
                } catch {
                    lines.append("Targeted compiled-cache removal: not completed; continuing with single-stage residency")
                    lines.append(contentsOf: modelErrorDetails(error))
                }
            }
            let sourcePrefix = lines.joined(separator: "\n") + "\n=== Integrated source encoding ===\n"
            lastPhase = "source-context.aimodel: source encoding"
            let source = await executeSourceContext(repetitions: 1, window: window, importedPiano: importedPiano) { text in progress(sourcePrefix + text) }
            lines.append("=== Source component before sampling ===\n" + source.report)
            guard let actualContext = source.context else { throw BassPreviewError.componentNotCompleted("source-context; see component report") }
            let conditionPrefix = lines.joined(separator: "\n") + "\n=== Integrated bass conditioning ===\n"
            lastPhase = "bass conditioning pipeline"
            let conditioning = await executeBassConditioning(window: window) { text in progress(conditionPrefix + text) }
            lines.append("=== Bass conditioning before sampling ===\n" + conditioning.report)
            guard let actualCondition = conditioning.condition, let mask = conditioning.mask,
                  mask.count == window.conditionTokens, mask.allSatisfy({ $0 == 1 }) else { throw BassPreviewError.componentNotCompleted("bass conditioning; see component report") }
            let contextInput = NDArray(scalars: actualContext, shape: [1,window.latentFrames,128])
            let conditionInput = NDArray(scalars: actualCondition, shape: [1,window.conditionTokens,2048])
            lines.append("=== Sampling and waveform generation ===")
            var latent = NDArray(scalars: noise, shape: [1,window.latentFrames,64])
            var momentum = NDArray(scalars: [Float](repeating: 0, count: window.latentFrames*64), shape: [1,window.latentFrames,64])
            let apgStage = Contract.Stage(assetFile: "apg-euler.aimodel", layerIndices: [],
                inputShapes: ["velocity":[2,window.latentFrames,64],"latent":[1,window.latentFrames,64],"momentum":[1,window.latentFrames,64],"dt":[1]],
                outputShapes: ["latent_next":[1,window.latentFrames,64],"momentum_next":[1,window.latentFrames,64],"guided_velocity":[1,window.latentFrames,64]], outputBindings: nil)
            for index in 0..<plannedSteps {
                let stepStart = clock.now
                let backendBefore = retention.phaseTotals.backendAwaitMs
                let copyBefore = retention.phaseTotals.outputCopyMs
                let reuseThisStep = reuseSamplingStages && (!sequentialFirstStep || index > 0)
                if sequentialFirstStep && index == 1 {
                    try checkManifest(samplingPool.count, 0, "preview.first_step_retained_handles")
                    try checkManifest(retention.modelOpens, 15 + cacheRepair.attempts, "preview.first_step_model_opens")
                    try checkManifest(retention.functionBinds, 15, "preview.first_step_function_binds")
                    try checkManifest(retention.completedCalls, 15, "preview.first_step_calls")
                    try checkManifest(retention.cacheHits, 0, "preview.first_step_hits")
                    lines.append("First actual step completed with zero retained Swift sampling handles; opening retained handles from step 2; compiled-cache reuse is backend-managed and unverified")
                }
                var state = ["noise":latent,"condition":conditionInput,"context":contextInput,
                             "timestep":NDArray(scalars: [times[index]], shape: [1])]
                for (stageIndex, stage) in ref.stages.enumerated() {
                    let phase = "Phase: sampling step \(index+1)/50, stage \(stageIndex+1)/14 \(stage.assetFile); completed steps \(completedSteps)/50; handles \(reuseThisStep ? "retained" : "single-stage scope")"
                    lastPhase = phase
                    try requirePreviewMemoryBudget()
                    let stageLog = boundedSamplingProgress ? [runHeader,phase] : lines + cacheRepair.events + [phase]
                    if boundedSamplingProgress {
                        interleavingPhaseProgress(prefix: runHeader, fileName: samplingBreadcrumb)(phase)
                        await progress("\(diagnosticThreeSteps ? "Uji langkah aktual" : "Sampling \(window.instrumentLabel)") \(index+1)/\(plannedSteps); tahap \(stageIndex+1)/14: \(stage.assetFile)")
                    } else {
                        saveProgress(stageLog, profile: .denoiser)
                        await progress(stageLog.joined(separator: "\n"))
                    }
                    var inputs: [String: NDArray] = [:]
                    for (name, shape) in stage.inputShapes {
                        guard let array = state[name], array.shape == shape else { throw ShardError.invalidManifest }
                        inputs[name] = array
                    }
                    let result = try await runPreviewSamplingStage(stage, folder: folder, inputs: inputs, aot: preparedAOT,
                        reuseStages: reuseThisStep, pool: &samplingPool, stats: &retention,
                        allowCacheRepair: window.isLong, cacheRepair: &cacheRepair,
                        phaseProgress: boundedSamplingProgress
                            ? interleavingPhaseProgress(prefix: stageLog.joined(separator: "\n"), fileName: samplingBreadcrumb)
                            : previewPhaseProgress(prefix: stageLog.joined(separator: "\n")))
                    if index == 0 && !diagnosticThreeSteps {
                        lines.append("First sampling step stage \(stageIndex+1)/14 \(stage.assetFile): load ms \(result.loadMs); \(result.phases.report)")
                    }
                    if diagnosticThreeSteps {
                        lines.append("Actual-prefix step \(index+1)/3 asset \(stage.assetFile): \(result.phases.report); thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                    }
                    guard let bindings = stage.outputBindings else { throw ShardError.invalidManifest }
                    for (name, array) in result.values {
                        guard let binding = bindings[name] else { throw ShardError.invalidManifest }
                        state[binding] = array
                    }
                }
                guard let velocity = state["velocity"] else { throw ProbeError.invalidOutputTensor }
                lastPhase = "Phase: sampling step \(index+1)/50, apg-euler.aimodel"
                try requirePreviewMemoryBudget()
                let apgLog = boundedSamplingProgress ? [runHeader,lastPhase] : lines + cacheRepair.events + [lastPhase]
                if boundedSamplingProgress {
                    interleavingPhaseProgress(prefix: runHeader, fileName: samplingBreadcrumb)(lastPhase)
                    await progress("\(diagnosticThreeSteps ? "Uji langkah aktual" : "Sampling \(window.instrumentLabel)") \(index+1)/\(plannedSteps): APG Euler")
                } else {
                    saveProgress(apgLog, profile: .denoiser)
                    await progress(apgLog.joined(separator: "\n"))
                }
                let result = try await runPreviewSamplingStage(apgStage, folder: folder,
                    inputs: ["velocity":velocity,"latent":latent,"momentum":momentum,
                             "dt":NDArray(scalars: [times[index]-times[index+1]], shape: [1])], aot: preparedAOT,
                    reuseStages: reuseThisStep, pool: &samplingPool, stats: &retention,
                    allowCacheRepair: window.isLong, cacheRepair: &cacheRepair,
                    phaseProgress: boundedSamplingProgress
                        ? interleavingPhaseProgress(prefix: apgLog.joined(separator: "\n"), fileName: samplingBreadcrumb)
                        : previewPhaseProgress(prefix: apgLog.joined(separator: "\n")))
                if diagnosticThreeSteps {
                    lines.append("Actual-prefix step \(index+1)/3 asset apg-euler.aimodel: \(result.phases.report); thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                }
                guard let next = result.values["latent_next"], let nextMomentum = result.values["momentum_next"],
                      result.values["guided_velocity"] != nil else { throw ProbeError.invalidOutputTensor }
                latent = next; momentum = nextMomentum
                if referenceMode && ref.checkpoints.contains(index+1) {
                    for (name, array) in [("latent",latent)] {
                        let actual = try copiedValues(array, shape: [1,window.latentFrames,64])
                        for kind in ["reference","fp32-reference"] {
                            let file = "step-\(index+1)-\(name)-\(kind).f32le"
                            guard let expected = references[file] else { throw ShardError.invalidManifest }
                            let stats = try compare(actual: actual, expected: expected, absoluteTolerance: ref.absoluteTolerance,
                                                    relativeTolerance: ref.relativeTolerance, relativeRmseTolerance: ref.relativeRmseTolerance)
                            lines.append("Step \(index+1) \(name) vs \(kind): max abs error \(stats.maxError); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                            allParity = allParity && stats.passed; checks += 1
                        }
                    }
                }
                completedSteps += 1
                lines.append("Sampling step \(index+1)/50 complete; ms \(ms(stepStart.duration(to: clock.now)))")
                if boundedSamplingProgress {
                    let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                    lines.append("Sampling step \(index+1)/50 API deltas: backend-await ms \(retention.phaseTotals.backendAwaitMs-backendBefore); output-copy ms \(retention.phaseTotals.outputCopyMs-copyBefore); thermal \(ProcessInfo.processInfo.thermalState.rawValue); footprint MiB \(footprint)")
                    saveProgress(lines + cacheRepair.events, profile: .denoiser)
                }
            }
            let expectedCalls = 15 * plannedSteps
            let expectedOpens = (reuseSamplingStages ? (sequentialFirstStep ? 30 : 15) : expectedCalls) + cacheRepair.attempts
            let expectedBinds = reuseSamplingStages ? (sequentialFirstStep ? 30 : 15) : expectedCalls
            let expectedHits = reuseSamplingStages ? expectedCalls - expectedBinds : 0
            try checkManifest(retention.modelOpens, expectedOpens, "preview.model_opens")
            try checkManifest(retention.functionBinds, expectedBinds, "preview.function_binds")
            try checkManifest(retention.completedCalls, expectedCalls, "preview.completed_calls")
            try checkManifest(retention.cacheHits, expectedHits, "preview.reuse_hits")
            try checkManifest(cacheRepair.successfulRetries, cacheRepair.attempts, "preview.cache_repair_successes")
            let scope = reuseSamplingStages ? (sequentialFirstStep ? "first-step-single-stage-then-retained" : "retained-stage") : "single-stage"
            lines.append("Sampling \(scope) scope: \(retention.modelOpens) opens, \(retention.functionBinds) binds, \(retention.cacheHits) hits, \(retention.completedCalls) calls")
            if reuseSamplingStages {
                lastPhase = "Phase: releasing retained sampling pool before decoder"
                saveProgress(lines + cacheRepair.events + [lastPhase], profile: .denoiser)
                await progress(boundedSamplingProgress ? lastPhase : (lines + cacheRepair.events + [lastPhase]).joined(separator: "\n"))
                samplingPool.removeAll()
                lines.append("Retained sampling pool released before decoder; native cache eviction NOT GUARANTEED")
            }
            if diagnosticThreeSteps {
                try checkManifest(checks, 0, "actual_prefix.no_matching_output_references")
                lines.append("Actual-input sampling diagnostic: COMPLETED (3/3 prefix steps, 45 calls); full generation NOT COMPLETED (3/50)")
                lines.append("Generation numerical parity: NOT EVALUATED; no matching host outputs. Existing conditioning checks retain their own scope.")
                lines.append("Decoder NOT RUN; WAV generated on device: NO (intentional diagnostic); no preview audio or baseline replacement")
            } else {
                let values = try copiedValues(latent, shape: [1,window.latentFrames,64])
                var decoderInput = [Float](repeating: 0, count: window.latentFrames*64)
                for t in 0..<window.latentFrames { for c in 0..<64 { decoderInput[c*window.latentFrames+t] = values[t*64+c] } }
                lastPhase = "Phase: decoder.aimodel loading/binding/inference"
                try requirePreviewMemoryBudget()
                lines.append("Phase: loading full \(window.seconds)-second VAE decoder; sampling Swift handles released, native eviction unverified")
                let decoderLog = lines + cacheRepair.events
                saveProgress(decoderLog, profile: .denoiser)
                await progress(boundedSamplingProgress ? lastPhase : decoderLog.joined(separator: "\n"))
                let decoderStage = Contract.Stage(assetFile: "decoder.aimodel", layerIndices: [],
                    inputShapes: ["latent":[1,64,window.latentFrames]], outputShapes: ["audio":[1,2,window.samples]], outputBindings: nil)
                let decoded = try await runSequentialPreviewStage(decoderStage, folder: folder,
                    inputs: ["latent":NDArray(scalars: decoderInput, shape: [1,64,window.latentFrames])], aot: preparedAOT, stats: &retention,
                    phaseProgress: boundedSamplingProgress
                        ? interleavingPhaseProgress(prefix: runHeader + "\n" + lastPhase, fileName: samplingBreadcrumb)
                        : previewPhaseProgress(prefix: (decoderLog + [lastPhase]).joined(separator: "\n")))
                guard let audioArray = decoded.values["audio"] else { throw ProbeError.invalidOutputTensor }
                let audio = try copiedValues(audioArray, shape: [1,2,window.samples])
                if referenceMode {
                    for kind in ["reference","fp32-reference"] {
                        guard let expected = references["audio-\(kind).f32le"] else { throw ShardError.invalidManifest }
                        let stats = try compare(actual: audio, expected: expected, absoluteTolerance: ref.absoluteTolerance,
                                                relativeTolerance: ref.relativeTolerance, relativeRmseTolerance: ref.relativeRmseTolerance)
                        lines.append("Decoded audio vs \(kind): max abs error \(stats.maxError); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                        allParity = allParity && stats.passed; checks += 1
                    }
                }
                try checkManifest(checks, expectedOutputChecks, "preview.output_checks")
                lastPhase = "writing bass and listening-mix WAV files"
                let sourceRef = try decoder.decode(SourceContextContract.self, from: Data(contentsOf: sourceFolder.appendingPathComponent("reference.json")))
                let piano: [Float]
                if let importedPiano {
                    piano = importedPiano.planarPCM
                } else {
                    piano = try readFloats(folder: sourceFolder, name: "audio.f32le", count: 2*window.samples, hashes: sourceRef.artifactSha256)
                }
                guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { throw ShardError.missingPreparedBundle }
                let output = documents.appendingPathComponent(window.stemID + "-preview-" + UUID().uuidString)
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                let stem = output.appendingPathComponent(window.stemID + ".wav"), mixed = output.appendingPathComponent("piano-with-" + window.stemID + ".wav")
                try writeStereoFloatWAV(audio, frames: window.samples, to: stem)
                let pianoClip = output.appendingPathComponent("piano.wav")
                try writeStereoFloatWAV(piano, frames: window.samples, to: pianoClip)
                if let importedPiano {
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                    try encoder.encode(importedPiano.metadata).write(to: output.appendingPathComponent("input.json"), options: .atomic)
                }
                var mix = zip(piano,audio).map { Float(0.78)*$0.0 + Float(0.55)*$0.1 }
                let mixPeak = mix.reduce(Float(0)) { max($0,abs($1)) }
                let gain = mixPeak > 0.98 ? Float(0.98)/mixPeak : Float(1)
                if gain < 1 { mix = mix.map { $0 * gain } }
                try writeStereoFloatWAV(mix, frames: window.samples, to: mixed)
                bassURL = stem; mixURL = mixed
                if referenceMode {
                    lines.append("Integrated \(window.stemID) preview numerical parity: \(allParity ? "PASS" : "NOT PASSED") (10 comparisons; fixture scope)")
                } else {
                    lines.append("Imported-recording generation: COMPLETED; numerical parity NOT EVALUATED (0 matching output-reference comparisons)")
                }
                lines.append("WAV generated on device: YES; \(window.stemID).wav and piano-with-\(window.stemID).wav, \(window.seconds) s stereo Float32/48 kHz")
                lines.append("WAV \(window.stemID) SHA-256: \(try streamingSHA256(stem))")
                lines.append("WAV listening mix SHA-256: \(try streamingSHA256(mixed))")
                lines.append("WAV input piano SHA-256: \(try streamingSHA256(pianoClip))")
                lines.append("Raw \(window.stemID) peak: \(audio.reduce(Float(0)) { max($0,abs($1)) }); listening mix gain: \(gain)")
                if referenceMode && !allParity { lines.append("Audio is diagnostic: numerical gate did not pass; numerical baseline NOT accepted. Listener approval is tracked separately.") }
                lines.append("Musical quality is not evaluated by this probe; full-song behavior and total model/device RAM: NOT VALIDATED")
                try lines.joined(separator: "\n").write(to: output.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
            }
        } catch {
            samplingPool.removeAll()
            lines.append(diagnosticThreeSteps ? "Actual-input sampling diagnostic: NOT COMPLETED" : "Integrated \(window.stemID) preview: NOT COMPLETED")
            lines.append("Last pipeline phase: \(lastPhase)")
            if let stageError = error as? StageExecutionFailure {
                lines.append("Failing asset: \(stageError.asset)")
                lines.append("Failing operation: \(stageError.operation)")
                lines.append("Retained handle reused for failing asset: \(stageError.reused)")
                lines.append(contentsOf: stageError.details)
            } else {
                lines.append(contentsOf: modelErrorDetails(error))
            }
            lines.append("Sampling/decoder calls completed: \(retention.completedCalls); model opens: \(retention.modelOpens); function binds: \(retention.functionBinds); reuse hits: \(retention.cacheHits)")
        }
        lines.append(contentsOf: cacheRepair.events)
        lines.append("Targeted CPU sampling cache repair: \(cacheRepair.attempts)/\(window.isLong ? 1 : 0) attempts; successful retried stage executions: \(cacheRepair.successfulRetries); failed bind attempts do not count as inference calls")
        lines.append("Completed sampling/decoder asset API timings (\(retention.phaseTotals.count) calls): \(retention.phaseTotals.report)")
        lines.append("First observed call per asset (\(retention.firstCallPhaseTotals.count) calls, NOT guaranteed cold cache): \(retention.firstCallPhaseTotals.report)")
        lines.append("Repeated asset calls (\(retention.repeatedCallPhaseTotals.count) calls; sampling handle reuse requested: \(reuseSamplingStages)): \(retention.repeatedCallPhaseTotals.report)")
        for asset in retention.phaseTotalsByAsset.keys.sorted() {
            if let phases = retention.phaseTotalsByAsset[asset] {
                lines.append("Asset timing total \(asset) (\(phases.count) calls): \(phases.report)")
            }
        }
        lines.append("Timing scope: completed sampling/decoder API intervals; backend-await is not isolated kernel time; total pipeline includes components, validation, logging, UI waits, scope release and file writing")
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed sampling steps: \(completedSteps)/50; output checks: \(checks)/\(expectedOutputChecks); total pipeline ms: \(ms(start.duration(to: clock.now)))")
        lines.append("Memory scope: sampled app process, not exact peak or total model/device RAM")
        saveProgress(lines, profile: .denoiser)
        if let bassURL { try? lines.joined(separator: "\n").write(to: bassURL.deletingLastPathComponent().appendingPathComponent("report.txt"), atomically: true, encoding: .utf8) }
        return BassPreviewResult(report: lines.joined(separator: "\n"), bassURL: bassURL, mixURL: mixURL)
    }

    static func writeStereoFloatWAV(_ planar: [Float], frames: Int, to url: URL) throws {
        guard frames > 0, planar.count == frames*2, planar.allSatisfy({ $0.isFinite }), frames <= (Int(UInt32.max)-50)/8 else {
            throw ProbeError.invalidFloatData
        }
        var data = Data(); data.reserveCapacity(58 + frames*8)
        func number<T: FixedWidthInteger>(_ value: T) {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: "RIFF".utf8); number(UInt32(50+frames*8)); data.append(contentsOf: "WAVEfmt ".utf8)
        number(UInt32(18)); number(UInt16(3)); number(UInt16(2)); number(UInt32(48000)); number(UInt32(48000*8))
        number(UInt16(8)); number(UInt16(32)); number(UInt16(0))
        data.append(contentsOf: "fact".utf8); number(UInt32(4)); number(UInt32(frames))
        data.append(contentsOf: "data".utf8); number(UInt32(frames*8))
        for index in 0..<frames { number(planar[index].bitPattern); number(planar[frames+index].bitPattern) }
        try data.write(to: url, options: .atomic)
    }

    private struct PairReuseContract: Decodable {
        let schemaVersion: Int
        let profile: String
        let assetFile: String
        let functionName: String
        let outputName: String
        let layerIndices: [Int]
        let inputShapes: [String: [Int]]
        let outputShape: [Int]
        let runCount: Int
        let scalarType: String
        let byteOrder: String
        let weightStorageDtype: String
        let computeDtype: String
        let checkpointDigest: String
        let checkpointFullSha256Verified: Bool
        let expectedFile: String
        let originalFp32ReferenceFile: String
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    private struct PrecisionProbeContract: Decodable {
        let schemaVersion: Int
        let profile: String
        let functionName: String
        let outputName: String
        let layerIndices: [Int]
        let inputShapes: [String: [Int]]
        let outputShape: [Int]
        let assets: [String: String]
        let passOrder: [String]
        let runsPerPass: Int
        let scalarType: String
        let byteOrder: String
        let computeDtype: String
        let checkpointDigest: String
        let checkpointFullSha256Verified: Bool
        let learnedParameterValuesIdentical: Bool
        let referenceFile: String
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
        let sourcePairManifestProfile: String
        let sourceWeightStorageDtype: String
        let sourcePairArtifactSha256: [String: String]
        let hostBf16VsOriginalFp32: BassPreviewContract.HostComparison
        let hostExpandedFp32VsBf16: BassPreviewContract.HostComparison
        let artifactSha256: [String: String]
    }

    private struct PrecisionParityStats: Sendable {
        var comparisons = 0
        var passed = 0
        var failed: Int { comparisons - passed }
    }

    private static func precisionPointwiseDetails(actual: [Float], expected: [Float]) throws -> String {
        guard actual.count == expected.count, !actual.isEmpty else { throw ProbeError.invalidOutputTensor }
        var failures = 0, worstIndex = 0
        var worstRatio = -Double.infinity, worstError = 0.0, worstBound = 0.0
        for index in actual.indices {
            guard actual[index].isFinite, expected[index].isFinite else { throw ProbeError.invalidFloatData }
            let error = abs(Double(actual[index]) - Double(expected[index]))
            let bound = 1e-4 + 1e-3 * abs(Double(expected[index]))
            if error > bound { failures += 1 }
            let ratio = error / bound
            if ratio > worstRatio { worstRatio = ratio; worstIndex = index; worstError = error; worstBound = bound }
        }
        let batch = worstIndex / (128 * 2048), token = (worstIndex / 2048) % 128, channel = worstIndex % 2048
        return "Pointwise errors over original bound: \(failures)/\(actual.count) (fraction \(Double(failures)/Double(actual.count))); worst bound ratio \(worstRatio); index [\(batch),\(token),\(channel)], flat \(worstIndex); actual \(Double(actual[worstIndex])); reference \(Double(expected[worstIndex])); abs error \(worstError); bound \(worstBound) = 1e-4 + 1e-3*abs(reference)"
    }

    static func runPrecisionProbe(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var stats = RetentionStats()
        var parity = PrecisionParityStats()
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "Probe run: \(UUID().uuidString); started UTC: \(ISO8601DateFormatter().string(from: Date()))",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: long-profile layers 0+1, existing BF16 asset vs expanded FP32 constants, A/B/A",
                     "Scope: fixed synthetic inputs; actual CFG batch 2, 128 hidden / 118 condition tokens; NOT full denoiser or audio generation",
                     "Requested compute policy: cpuOnly; actual accelerator placement NOT MEASURED",
                     "All learned values identical; FP32 calculations/inputs/outputs in both variants; original FP32 host reference required",
                     "Diagnostic timing collection: finite pointwise/RMSE parity failures are recorded and remain NOT PASSED; scheduled calls continue without changing bounds",
                     "Shape/type/hash/nonfinite/native/memory failures still stop; this component probe generates no waveform and does not replace the sampler",
                     "Three passes, four calls each; same model AND function retained within a pass, Swift scope released between passes",
                     "No hidden warmup, cache deletion, retry or automatic baseline replacement; first call is not guaranteed cold cache",
                     "backend-await is API wall time, NOT isolated kernel time or proof of full-loop speedup",
                     "Thermal state codes: 0 nominal, 1 fair, 2 serious, 3 critical; low-power mode: \(ProcessInfo.processInfo.isLowPowerModeEnabled)",
                     "Process guard 2000 MiB before load and each call, rechecked after bind; NOT a native compiler peak guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        lines.append("App build configuration: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeBuildConfiguration") as? String ?? "unrecorded")")
        lines.append("Swift app optimization: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeSwiftOptimizationLevel") as? String ?? "unrecorded")")
        do {
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates/dit-precision-1024") else {
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(PrecisionProbeContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            let pair = bassPreviewStages(window: .long1024)[1]
            let assets = ["bfloat16":"blocks-bf16.aimodel", "float32":"blocks-fp32.aimodel"]
            try checkManifest(ref.schemaVersion, 1, "precision.schema")
            try checkManifest(ref.profile, "dit-precision-1024", "precision.profile")
            try checkManifest(ref.functionName, "main", "precision.function")
            try checkManifest(ref.outputName, "hidden_out", "precision.output")
            try checkManifest(ref.layerIndices, [0,1], "precision.layers")
            try checkManifest(ref.inputShapes, pair.inputShapes, "precision.inputs")
            try checkManifest(ref.outputShape, [2,128,2048], "precision.output_shape")
            try checkManifest(ref.assets, assets, "precision.assets")
            try checkManifest(ref.passOrder, ["bfloat16","float32","bfloat16"], "precision.order")
            try checkManifest(ref.runsPerPass, 4, "precision.runs")
            try checkManifest(ref.scalarType, "float32", "precision.scalar_type")
            try checkManifest(ref.byteOrder, "little", "precision.byte_order")
            try checkManifest(ref.computeDtype, "float32", "precision.compute")
            try checkManifest(ref.checkpointDigest, Profile.denoiser.checkpointDigest, "precision.checkpoint")
            try checkManifest(ref.checkpointFullSha256Verified, true, "precision.checkpoint_verified")
            try checkManifest(ref.learnedParameterValuesIdentical, true, "precision.learned_values")
            try checkManifest(ref.referenceFile, "hidden-reference.f32le", "precision.reference")
            try checkManifest(ref.absoluteTolerance, 1e-4, "precision.absolute_tolerance")
            try checkManifest(ref.relativeTolerance, 1e-3, "precision.relative_tolerance")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "precision.relative_rmse_tolerance")
            try checkManifest(ref.sourcePairManifestProfile, "bass-preview-1024", "precision.original_profile")
            try checkManifest(ref.sourceWeightStorageDtype, "bfloat16", "precision.original_storage")
            for metric in [ref.hostBf16VsOriginalFp32, ref.hostExpandedFp32VsBf16] {
                guard metric.finite, metric.parityPassed, metric.maxAbsoluteError == 0, metric.rmse == 0,
                      metric.normalizedRmse == 0, metric.elementwiseFractionOverBound == 0 else { throw ShardError.invalidManifest }
            }
            let nativeFiles: Set<String> = ["metadata.json","main.hash","main.mlirb"]
            var required = Set(ref.inputShapes.keys.map { "\($0).f32le" }); required.insert(ref.referenceFile)
            for asset in assets.values { for file in nativeFiles { required.insert(asset + "/" + file) } }
            try checkManifest(Set(ref.artifactSha256.keys), required, "precision.payload")
            try checkManifest(Set(ref.sourcePairArtifactSha256.keys), nativeFiles, "precision.original_bf16_payload")
            for file in nativeFiles {
                try checkManifest(ref.sourcePairArtifactSha256[file], ref.artifactSha256["blocks-bf16.aimodel/" + file], "precision.original_bf16_hash")
            }
            lines.append("Phase: verifying model binaries and fixed tensors; no model open yet")
            saveProgress(lines, profile: .denoiser); await progress(lines.joined(separator: "\n"))
            for asset in assets.values {
                for file in nativeFiles {
                    let name = asset + "/" + file
                    guard let expectedHash = ref.artifactSha256[name] else { throw ShardError.invalidManifest }
                    try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)), expectedHash, "precision.asset_sha256")
                }
            }
            var inputs: [String: NDArray] = [:]
            for (name,shape) in ref.inputShapes {
                inputs[name] = NDArray(scalars: try readFloats(folder: folder, name: name + ".f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256), shape: shape)
            }
            let expected = try readFloats(folder: folder, name: ref.referenceFile, count: ref.outputShape.reduce(1,*), hashes: ref.artifactSha256)
            lines.append("SHA-256: VERIFIED both native assets, eight fixed inputs and original FP32 host reference; existing BF16 asset identity matched")
            var means: [Double] = []
            for (pass,kind) in ref.passOrder.enumerated() {
                guard let asset = assets[kind] else { throw ShardError.invalidManifest }
                let stage = Contract.Stage(assetFile: asset, layerIndices: [0,1], inputShapes: ref.inputShapes,
                                           outputShapes: ["hidden_out":ref.outputShape], outputBindings: nil)
                let times = try await runPrecisionPass(stage, folder: folder, inputs: inputs, expected: expected,
                    pass: pass+1, stats: &stats, parity: &parity, lines: &lines, progress: progress)
                let mean = times.dropFirst().reduce(0,+) / 3
                means.append(mean)
                lines.append("Pass \(pass+1)/3 \(kind): first backend-await ms \(times[0]); repeated mean (calls 2-4) ms \(mean)")
                lines.append("Pass \(pass+1) Swift model/function scope released; native eviction NOT GUARANTEED")
            }
            try checkManifest(stats.modelOpens, 3, "precision.opens")
            try checkManifest(stats.functionBinds, 3, "precision.binds")
            try checkManifest(stats.cacheHits, 9, "precision.hits")
            try checkManifest(stats.completedCalls, 12, "precision.calls")
            try checkManifest(parity.comparisons, 12, "precision.comparisons")
            lines.append("Precision timing collection: COMPLETED (12/12 finite component calls)")
            lines.append("Precision component strict parity: \(parity.passed == 12 ? "PASS" : "NOT PASSED") (\(parity.passed)/12 calls pass; original pointwise and normalized-RMSE bounds unchanged)")
            if parity.failed > 0 {
                lines.append("Timing ratios below are DIAGNOSTIC ONLY: at least one strict comparison failed; neither variant is accepted as a sampler replacement by this result")
            }
            if means[1] > 0 {
                lines.append("Component BF16-A1/FP32-B repeated backend-await ratio: \(means[0]/means[1])")
                lines.append("Component BF16-A2/FP32-B repeated backend-await ratio: \(means[2]/means[1]); NOT full-loop speedup")
            }
        } catch {
            lines.append("Precision component probe: NOT COMPLETED; baseline unchanged")
            if let failure = error as? StageExecutionFailure {
                lines.append("Failing asset: \(failure.asset); operation: \(failure.operation)")
                lines.append(contentsOf: failure.details)
            } else { lines.append(contentsOf: modelErrorDetails(error)) }
        }
        let usage = memory.finish()
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Strict parity comparisons completed: \(parity.comparisons)/12; passed \(parity.passed); NOT PASSED \(parity.failed)")
        lines.append("Completed component calls: \(stats.completedCalls)/12; opens \(stats.modelOpens); binds \(stats.functionBinds); hits \(stats.cacheHits); total probe ms \(ms(start.duration(to: clock.now)))")
        lines.append("Memory scope: sampled app process, not exact peak/total RAM. First-call cache state and thermal effects are not controlled.")
        saveProgress(lines, profile: .denoiser)
        return lines.joined(separator: "\n")
    }

    private static func runPrecisionPass(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray],
                                         expected: [Float], pass: Int, stats: inout RetentionStats,
                                         parity: inout PrecisionParityStats,
                                         lines: inout [String], progress: @escaping @MainActor @Sendable (String) -> Void) async throws -> [Double] {
        var owned: [String: RetainedStage] = [:]
        defer { owned.removeAll() }
        var times: [Double] = []
        for call in 1...4 {
            try requirePreviewMemoryBudget()
            let phase = "Phase: pass \(pass)/3, \(stage.assetFile), call \(call)/4; thermal state \(ProcessInfo.processInfo.thermalState.rawValue)"
            saveProgress(lines + [phase], profile: .denoiser)
            await progress((lines + [phase]).joined(separator: "\n"))
            let result = try await runRetainedStage(stage, folder: folder, inputs: inputs, pool: &owned, stats: &stats,
                options: .cpuOnly, phaseProgress: previewPhaseProgress(prefix: (lines + [phase]).joined(separator: "\n")))
            guard let output = result.values["hidden_out"] else { throw ProbeError.invalidOutputTensor }
            let actual = try copiedValues(output, shape: [2,128,2048])
            let comparison = try compare(actual: actual, expected: expected, absoluteTolerance: 1e-4,
                                         relativeTolerance: 1e-3, relativeRmseTolerance: 1e-3)
            let pointwise = try precisionPointwiseDetails(actual: actual, expected: expected)
            parity.comparisons += 1
            if comparison.passed { parity.passed += 1 }
            lines.append("Pass \(pass) call \(call): \(result.phases.report); original FP32 reference max abs error \(comparison.maxError), RMSE \(comparison.rmse), normalized RMSE \(comparison.normalizedRmse); \(comparison.passed ? "PASS" : "NOT PASSED"); thermal state after \(ProcessInfo.processInfo.thermalState.rawValue)")
            lines.append("Pass \(pass) call \(call): \(pointwise)")
            if !comparison.passed {
                lines.append("Pass \(pass) call \(call): strict parity remains NOT PASSED; continuing scheduled finite-only timing diagnostics, not retrying or accepting output")
            }
            times.append(result.phases.backendAwaitMs)
        }
        return times
    }

    static func runNativeAssetPreflight(window: CoreAIPreviewWindow,
        progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(), start = ContinuousClock().now
        let memory = CoreAIProcessFootprintSampler()
        let prior = savedDenoiserReport()
        let runID = UUID().uuidString
        var lines = ["Runtime: CoreAI native-asset preflight; run \(runID)",
            "Started UTC: \(ISO8601DateFormatter().string(from: Date())); OS: \(ProcessInfo.processInfo.operatingSystemVersionString); architecture: \(AIModel.deviceArchitectureName)",
            "Selected instrument: \(window.stemID); preset SHA-256 \(window.presetSHA256)",
            "Instrumentation revision: native-preflight-v1; EXPLICIT synthetic warmup/diagnostic, NOT generation",
            "No encoder/Qwen/conditioning; each of 16 sampling/decoder assets runs ONCE on independent synthetic inputs; no inter-asset feedback, no WAV",
            "Synthetic recipe: Float32 zeros, except timestep/rope_cos=1 and dt=0.02; no RNG. No claim these are real piano activations",
            "cpuOnly requested; actual placement unmeasured. One Swift scope per asset; native eviction/cache persistence NOT GUARANTEED",
            "Timings are API wall intervals, not kernel time; first-call cache state uncontrolled",
            "No cache deletion, retry, repair or sampler replacement. Production still 50 steps and keeps its original ONE targeted bind-repair budget",
            "Shape/type/hash/nonfinite/native/memory error stops immediately. Finite outputs are NOT numerical parity; no host comparisons",
            "Fresh process requested by UI workflow; freshness/absence of native residency not independently verified",
            "Thermal before \(ProcessInfo.processInfo.thermalState.rawValue); low-power \(ProcessInfo.processInfo.isLowPowerModeEnabled); 2000 MiB process guard is NOT a native compiler peak guarantee",
            "Build: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeBuildConfiguration") as? String ?? "unrecorded"); optimization: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeSwiftOptimizationLevel") as? String ?? "unrecorded")"]
        var stats = RetentionStats()
        var attemptedAssets = 0
        var reportURL: URL?
        var lastPhase = "manifest validation"
        do {
            guard window.isLong, let resources = Bundle.main.resourceURL,
                  let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { throw ShardError.invalidManifest }
            let output = documents.appendingPathComponent("native-preflight-" + runID)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            reportURL = output.appendingPathComponent("report.txt")
            if let prior { try prior.write(to: output.appendingPathComponent("previous-report.txt"), atomically: true, encoding: .utf8) }
            let folder = resources.appendingPathComponent("ModelCandidates/" + window.previewBundle)
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(BassPreviewContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            let expected = bassPreviewStages(window: window)
            try checkManifest(ref.schemaVersion, 1, "preflight.schema")
            try checkManifest(ref.profile, window.previewBundle, "preflight.profile")
            try checkManifest(ref.functionName, "main", "preflight.function")
            try checkManifest(ref.scalarType, "float32", "preflight.scalar")
            try checkManifest(ref.byteOrder, "little", "preflight.byte_order")
            try checkManifest(ref.latentFrames, window.latentFrames, "preflight.frames")
            try checkManifest(ref.conditionTokens, window.conditionTokens, "preflight.tokens")
            try checkManifest(ref.inferenceSteps, 50, "preflight.production_steps")
            try checkManifest(ref.guidanceScale, 7, "preflight.guidance")
            try checkManifest(ref.absoluteTolerance, 1e-4, "preflight.abs")
            try checkManifest(ref.relativeTolerance, 1e-3, "preflight.rel")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "preflight.nrmse")
            try checkManifest(ref.vaeCheckpointDigest, "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0", "preflight.vae")
            try checkManifest(ref.checkpointDigest, Profile.denoiser.checkpointDigest, "preflight.checkpoint")
            try checkManifest(ref.allCheckpointsVerified, true, "preflight.verified_checkpoint")
            try checkManifest(ref.weightStorageDtype, Optional("bfloat16"), "preflight.storage")
            try checkManifest(ref.stages.count, expected.count, "preflight.stages")
            for (actual,stage) in zip(ref.stages,expected) {
                try checkManifest(actual.assetFile, stage.assetFile, "preflight.asset")
                try checkManifest(actual.layerIndices, stage.layerIndices, "preflight.layers")
                try checkManifest(actual.inputShapes, stage.inputShapes, "preflight.inputs")
                try checkManifest(actual.outputShapes, stage.outputShapes, "preflight.outputs")
                try checkManifest(actual.outputBindings, stage.outputBindings, "preflight.bindings")
            }
            var stages = expected
            stages.append(Contract.Stage(assetFile: "apg-euler.aimodel", layerIndices: [],
                inputShapes: ["velocity":[2,window.latentFrames,64],"latent":[1,window.latentFrames,64],"momentum":[1,window.latentFrames,64],"dt":[1]],
                outputShapes: ["latent_next":[1,window.latentFrames,64],"momentum_next":[1,window.latentFrames,64],"guided_velocity":[1,window.latentFrames,64]], outputBindings: nil))
            stages.append(Contract.Stage(assetFile: "decoder.aimodel", layerIndices: [],
                inputShapes: ["latent":[1,64,window.latentFrames]], outputShapes: ["audio":[1,2,window.samples]], outputBindings: nil))
            // Capture the old report BEFORE replacing live progress. Only an
            // allowlisted asset from the same instrument may affect order.
            if let prior, prior.contains("Selected instrument: \(window.stemID);") {
                let marker = "Failing asset: "
                if let line = prior.components(separatedBy: "\n").first(where: { $0.hasPrefix(marker) }),
                   let index = stages.firstIndex(where: { $0.assetFile == String(line.dropFirst(marker.count)) }) {
                    let target = stages.remove(at: index); stages.insert(target, at: 0)
                    lines.append("Prior matching failure asset FIRST: \(target.assetFile); no other model opens before this asset")
                }
            }
            lines.append("Planned order: " + stages.map { $0.assetFile }.joined(separator: ", "))
            for (index,stage) in stages.enumerated() {
                lastPhase = "Preflight \(index+1)/16 SHA-256: \(stage.assetFile)"
                await progress(lastPhase)
                for file in ["metadata.json","main.hash","main.mlirb"] {
                    let name = stage.assetFile + "/" + file
                    guard let hash = ref.artifactSha256[name], hash.count == 64,
                          hash.allSatisfy({ "0123456789abcdef".contains($0) }) else { throw ShardError.invalidManifest }
                    try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)), hash, "preflight.native_hash")
                    lines.append("SHA-256 verified \(name): \(hash)")
                }
                try requirePreviewMemoryBudget()
                lastPhase = "Preflight \(index+1)/16: \(stage.assetFile)"
                saveProgress(lines + [lastPhase], profile: .denoiser)
                if let reportURL { try (lines + [lastPhase]).joined(separator: "\n").write(to: reportURL, atomically: true, encoding: .utf8) }
                await progress(lastPhase)
                var inputs: [String: NDArray] = [:]
                for (name,shape) in stage.inputShapes {
                    let value: Float = name == "timestep" || name == "rope_cos" ? 1 : (name == "dt" ? 0.02 : 0)
                    inputs[name] = NDArray(scalars: [Float](repeating: value, count: shape.reduce(1,*)), shape: shape)
                }
                attemptedAssets += 1
                let result = try await runSequentialPreviewStage(stage, folder: folder, inputs: inputs, stats: &stats,
                    phaseProgress: interleavingPhaseProgress(prefix: "Preflight run \(runID)\n" + lastPhase, fileName: "last-native-preflight-phase.txt"))
                lines.append("\(lastPhase) finite outputs verified: \(result.phases.report); thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                if let reportURL { try lines.joined(separator: "\n").write(to: reportURL, atomically: true, encoding: .utf8) }
            }
            try checkManifest(stats.modelOpens, 16, "preflight.opens")
            try checkManifest(stats.functionBinds, 16, "preflight.binds")
            try checkManifest(stats.completedCalls, 16, "preflight.calls")
            try checkManifest(stats.cacheHits, 0, "preflight.hits")
            lines.append("Native preflight: COMPLETED (16/16 synthetic calls); cache readiness for later generation NOT PROVEN")
        } catch {
            lines.append("Native preflight: NOT COMPLETED; STOP, no automatic retry; last phase: \(lastPhase)")
            if let failure = error as? StageExecutionFailure {
                lines.append("Failing asset: \(failure.asset)")
                lines.append("Failing operation: \(failure.operation)")
                lines.append(contentsOf: failure.details)
            } else { lines.append(contentsOf: modelErrorDetails(error)) }
        }
        let usage = memory.finish()
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Preflight asset attempts \(attemptedAssets)/16; opens \(stats.modelOpens); binds \(stats.functionBinds); completed calls \(stats.completedCalls)/16; hits \(stats.cacheHits); total ms \(ms(start.duration(to: clock.now)))")
        lines.append("Numerical parity NOT EVALUATED; no WAV or musical acceptance; no generation steps replayed, no cache deletion requested")
        if let reportURL { try? lines.joined(separator: "\n").write(to: reportURL, atomically: true, encoding: .utf8) }
        saveProgress(lines, profile: .denoiser)
        return lines.joined(separator: "\n")
    }

    private struct InterleavingContract: Decodable {
        let schemaVersion: Int, pairCount: Int, callsPerAsset: Int
        let profile: String, weightStorageDtype: String, computeDtype: String
        let scalarType: String, byteOrder: String, checkpointDigest: String
        let checkpointFullSha256Verified: Bool
        let hostExact: [Bool]
        let assets: [String], inputShapes: [String: [Int]], outputShape: [Int]
        let absoluteTolerance: Double, relativeTolerance: Double, relativeRmseTolerance: Double
        let artifactSha256: [String: String], sourceAssetSha256: [String: String]
        let sourceManifestSha256: String
    }

    // Keep live native breadcrumbs separate from the complete report. Capturing
    // the growing report here makes repeated writes/rendering grow with call
    // position and can confound an order-sensitive timing experiment.
    private static func interleavingPhaseProgress(prefix: String, fileName: String = "last-interleaving-phase.txt") -> @Sendable (String) -> Void {
        return { phase in
            guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
            try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try? (prefix + "\n" + phase).write(to: root.appendingPathComponent(fileName),
                atomically: true, encoding: .utf8)
        }
    }

    static func runInterleavingProbe(count: Int, reverse: Bool,
        progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(), start = ContinuousClock().now
        let memory = CoreAIProcessFootprintSampler()
        var lines = ["Runtime: CoreAI BF16 interleaving diagnostic; run \(UUID().uuidString)",
            "Instrumentation revision: bounded-live-progress-v1; full report checkpointed between modes and on completion/error; native breadcrumb stored separately",
            "Started UTC: \(ISO8601DateFormatter().string(from: Date())); OS \(ProcessInfo.processInfo.operatingSystemVersionString); architecture \(AIModel.deviceArchitectureName)",
            "Fixed synthetic inputs; CFG batch 2 / hidden 128 / condition 118; no output feedback or waveform",
            "BF16 storage / FP32 compute; cpuOnly requested; actual accelerator placement NOT MEASURED",
            "No hidden warmup, cache purge, retry or baseline replacement. First call not guaranteed cold cache",
            "Finite parity failures remain NOT PASSED while scheduled timing continues; hash/shape/type/nonfinite/native/memory failure stops",
            "Process guard 2000 MiB before load/call and after bind; not a native peak guarantee",
            "backend-await is API wall time, not kernel time; footprint sampled process, not exact peak/total RAM",
            "Build: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeBuildConfiguration") as? String ?? "unrecorded"); optimization: \(Bundle.main.object(forInfoDictionaryKey: "CoreAIProbeSwiftOptimizationLevel") as? String ?? "unrecorded")",
            "Low-power: \(ProcessInfo.processInfo.isLowPowerModeEnabled); thermal before: \(ProcessInfo.processInfo.thermalState.rawValue); K=\(count); reversed mode order=\(reverse)"]
        #if targetEnvironment(simulator)
        lines.append("Simulator: not physical-device evidence")
        #endif
        var parity = PrecisionParityStats()
        var totalCalls = 0
        do {
            guard [2,4,8,12].contains(count), let resources = Bundle.main.resourceURL else { throw ShardError.invalidManifest }
            let folder = resources.appendingPathComponent("ModelCandidates/dit-interleaving-1024")
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(InterleavingContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            guard [2,4,8,12].contains(ref.pairCount), count <= ref.pairCount else { throw ShardError.invalidManifest }
            let names = (0..<ref.pairCount).map { String(format: "blocks-%02d-%02d.aimodel", 2*$0, 2*$0+1) }
            let shape = [2,128,2048], native = ["metadata.json","main.hash","main.mlirb"]
            try checkManifest(ref.schemaVersion, 1, "interleaving.schema")
            try checkManifest(ref.profile, "dit-interleaving-1024", "interleaving.profile")
            try checkManifest(ref.callsPerAsset, 20, "interleaving.calls")
            try checkManifest(ref.assets, names, "interleaving.assets")
            try checkManifest(ref.inputShapes, bassPreviewStages(window: .long1024)[1].inputShapes, "interleaving.inputs")
            try checkManifest(ref.outputShape, shape, "interleaving.output")
            try checkManifest(ref.weightStorageDtype, "bfloat16", "interleaving.storage")
            try checkManifest(ref.computeDtype, "float32", "interleaving.compute")
            try checkManifest(ref.scalarType, "float32", "interleaving.scalar")
            try checkManifest(ref.byteOrder, "little", "interleaving.byte_order")
            try checkManifest(ref.checkpointDigest, Profile.denoiser.checkpointDigest, "interleaving.checkpoint")
            try checkManifest(ref.checkpointFullSha256Verified, true, "interleaving.checkpoint_verified")
            try checkManifest(ref.hostExact, Array(repeating: true, count: ref.pairCount), "interleaving.host_exact")
            try checkManifest(ref.absoluteTolerance, 1e-4, "interleaving.abs")
            try checkManifest(ref.relativeTolerance, 1e-3, "interleaving.rel")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "interleaving.rmse")
            func isHash(_ s: String) -> Bool { s.count == 64 && s.allSatisfy { "0123456789abcdef".contains($0) } }
            guard isHash(ref.sourceManifestSha256) else { throw ShardError.invalidManifest }
            let sourceKeys = Set(names.flatMap { a in native.map { a + "/" + $0 } })
            var required = sourceKeys
            for name in ref.inputShapes.keys { required.insert(name + ".f32le") }
            for i in 0..<ref.pairCount { required.insert(String(format: "reference-%02d.f32le", i)) }
            try checkManifest(Set(ref.artifactSha256.keys), required, "interleaving.payload")
            try checkManifest(Set(ref.sourceAssetSha256.keys), sourceKeys, "interleaving.source_payload")
            for name in sourceKeys {
                try checkManifest(ref.sourceAssetSha256[name], ref.artifactSha256[name], "interleaving.source_identity")
            }
            guard Set(names.compactMap { ref.artifactSha256[$0 + "/main.mlirb"] }).count == ref.pairCount else { throw ShardError.invalidManifest }
            lines.append("Verifying SHA-256 before opening any model; source manifest \(ref.sourceManifestSha256)")
            await progress(lines.joined(separator: "\n"))
            for name in required.sorted() {
                guard let hash = ref.artifactSha256[name], isHash(hash) else { throw ShardError.invalidManifest }
                try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)), hash, "interleaving.hash")
                lines.append("Artifact \(name) SHA-256 \(hash)")
            }
            var inputs: [String: NDArray] = [:]
            for (name,dimensions) in ref.inputShapes {
                inputs[name] = NDArray(scalars: try readFloats(folder: folder, name: name + ".f32le",
                    count: dimensions.reduce(1,*), hashes: ref.artifactSha256), shape: dimensions)
            }
            var references: [[Float]] = []
            for i in 0..<count {
                references.append(try readFloats(folder: folder, name: String(format: "reference-%02d.f32le", i),
                    count: shape.reduce(1,*), hashes: ref.artifactSha256))
            }
            let modes = ["single-before", "pool-only"] + (reverse ? ["interleaved","grouped"] : ["grouped","interleaved"]) + ["single-after"]
            saveProgress(lines, profile: .denoiser)
            var means: [String: [Int: Double]] = [:]
            for mode in modes {
                let result = try await runInterleavingMode(mode, count: count, folder: folder, ref: ref,
                    inputs: inputs, references: references, parity: &parity, lines: &lines, progress: progress)
                totalCalls += result.calls
                means[mode] = result.means
                lines.append("Mode \(mode) Swift scope released; native eviction NOT GUARANTEED")
                saveProgress(lines, profile: .denoiser)
            }
            try checkManifest(totalCalls, 60 + 40*count, "interleaving.total_calls")
            for index in 0..<count {
                if let grouped = means["grouped"]?[index], let interleaved = means["interleaved"]?[index], grouped > 0 {
                    lines.append("DIAGNOSTIC asset \(ref.assets[index]) interleaved/grouped repeated mean ratio \(interleaved/grouped)")
                }
            }
            if let single = means["single-before"]?[0], single > 0 {
                for mode in ["pool-only","single-after"] {
                    if let value = means[mode]?[0] { lines.append("DIAGNOSTIC A \(mode)/single-before repeated mean ratio \(value/single)") }
                }
            }
            lines.append("Interleaving timing collection: COMPLETED; strict parity \(parity.passed == parity.comparisons ? "PASS" : "NOT PASSED") \(parity.passed)/\(parity.comparisons)")
        } catch {
            lines.append("Interleaving timing collection: NOT COMPLETED; prior calls retained in report; baseline unchanged")
            lines.append(contentsOf: modelErrorDetails(error))
        }
        let usage = memory.finish()
        lines.append("Strict comparisons completed \(parity.comparisons); passed \(parity.passed); failed \(parity.failed)")
        if let peak = usage.peak { lines.append("Sampled maximum process MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Total ms \(ms(start.duration(to: clock.now))); timings DIAGNOSTIC ONLY if any strict failure; no full-loop speedup claim")
        saveProgress(lines, profile: .denoiser)
        return lines.joined(separator: "\n")
    }

    private static func runInterleavingMode(_ mode: String, count: Int, folder: URL, ref: InterleavingContract,
        inputs: [String: NDArray], references: [[Float]], parity: inout PrecisionParityStats,
        lines: inout [String], progress: @escaping @MainActor @Sendable (String) -> Void) async throws -> (calls: Int, means: [Int: Double]) {
        var pool: [String: RetainedStage] = [:]
        defer { pool.removeAll() }
        var stats = RetentionStats()
        let clock = ContinuousClock(), start = ContinuousClock().now
        lines.append("Mode \(mode) begins: thermal \(ProcessInfo.processInfo.thermalState.rawValue); low-power \(ProcessInfo.processInfo.isLowPowerModeEnabled)")
        let size = mode.hasPrefix("single-") ? 1 : count
        let stages = (0..<size).map { i in Contract.Stage(assetFile: ref.assets[i], layerIndices: [2*i,2*i+1],
            inputShapes: ref.inputShapes, outputShapes: ["hidden_out":ref.outputShape], outputBindings: nil) }
        for stage in stages {
            try requirePreviewMemoryBudget()
            let phase = "Mode \(mode): opening/binding \(stage.assetFile); no inference yet"
            interleavingPhaseProgress(prefix: lines[0])(phase)
            await progress(phase)
            let begin = clock.now
            let probe = try await CoreAIModelProbe(contentsOf: folder.appendingPathComponent(stage.assetFile), options: .cpuOnly)
            stats.modelOpens += 1
            let openMs = ms(begin.duration(to: clock.now))
            var setup = CoreAIModelProbe.StatelessTimings()
            let function = try probe.loadStatelessFunction(functionNamed: "main", inputs: inputs, timings: &setup)
            stats.functionBinds += 1
            pool[stage.assetFile] = RetainedStage(probe: probe, function: function)
            try requirePreviewMemoryBudget()
            lines.append("Mode \(mode) setup \(stage.assetFile): open ms \(openMs); bind ms \(setup.functionBindMs); validation ms \(setup.inputValidationMs)")
        }
        let order: [Int]
        if mode == "grouped" { order = (0..<count).flatMap { Array(repeating: $0, count: 20) } }
        else if mode == "interleaved" { order = (0..<20).flatMap { _ in Array(0..<count) } }
        else { order = Array(repeating: 0, count: 20) }
        var times: [Int: [Double]] = [:]
        var firstHashes: [Int: String] = [:]
        for (position,index) in order.enumerated() {
            try requirePreviewMemoryBudget()
            let phase = "Mode \(mode), K=\(count), call \(position+1)/\(order.count), \(stages[index].assetFile)"
            let livePrefix = lines[0] + "\n" + phase
            interleavingPhaseProgress(prefix: livePrefix)("Before call")
            await progress(phase)
            let result = try await runRetainedStage(stages[index], folder: folder, inputs: inputs, pool: &pool, stats: &stats,
                options: .cpuOnly, phaseProgress: interleavingPhaseProgress(prefix: livePrefix))
            guard let tensor = result.values["hidden_out"] else { throw ProbeError.invalidOutputTensor }
            let values = try copiedValues(tensor, shape: ref.outputShape)
            let comparison = try compare(actual: values, expected: references[index], absoluteTolerance: 1e-4,
                relativeTolerance: 1e-3, relativeRmseTolerance: 1e-3)
            let details = try precisionPointwiseDetails(actual: values, expected: references[index])
            let hash = values.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
            let stable = firstHashes[index].map { $0 == hash } ?? true
            if firstHashes[index] == nil { firstHashes[index] = hash }
            parity.comparisons += 1
            if comparison.passed { parity.passed += 1 }
            times[index, default: []].append(result.phases.backendAwaitMs)
            let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
            lines.append("\(phase): \(result.phases.report); max abs \(comparison.maxError); RMSE \(comparison.rmse); normalized RMSE \(comparison.normalizedRmse); \(comparison.passed ? "PASS" : "NOT PASSED"); output SHA-256 \(hash); matches first output in mode \(stable); thermal \(ProcessInfo.processInfo.thermalState.rawValue); footprint MiB \(footprint)")
            lines.append(details)
        }
        try checkManifest(stats.modelOpens, size, "interleaving.mode_opens")
        try checkManifest(stats.functionBinds, size, "interleaving.mode_binds")
        // All functions were bound before inference; every call is a pool lookup hit.
        try checkManifest(stats.cacheHits, order.count, "interleaving.mode_hits")
        try checkManifest(stats.completedCalls, order.count, "interleaving.mode_calls")
        var means: [Int: Double] = [:]
        for index in times.keys.sorted() {
            let values = times[index]!, repeated = Array(values.dropFirst()).sorted()
            let mean = repeated.reduce(0,+)/Double(repeated.count)
            means[index] = mean
            lines.append("Mode \(mode) asset \(ref.assets[index]): first ms \(values[0]); calls 2-20 mean \(mean); median \(repeated[repeated.count/2]); p95 nearest rank \(repeated[Int(ceil(0.95*Double(repeated.count)))-1]); min \(repeated.first!); max \(repeated.last!)")
        }
        lines.append("Mode \(mode) completed: opens \(stats.modelOpens); binds \(stats.functionBinds); prebound pool hits \(stats.cacheHits); calls \(stats.completedCalls); wall ms \(ms(start.duration(to: clock.now)))")
        return (stats.completedCalls, means)
    }

    static func runPairReuse(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: pretrained ACE Base layers 0+1 / one retained model AND function / fixed synthetic inputs",
                     "Scope: first call + three repeated calls; 13 hidden / 64 condition tokens; NOT full denoiser or instrument generation",
                     "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
                     "One successful model open and function bind intended; no hidden warmup, cache clearing or reload between runs",
                     "backend-await is an API wall interval, not isolated kernel time; first call is not a guaranteed cold-cache run",
                     "Process footprint guard before load and each call: 2000 MiB; NOT an OS peak-memory guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var completed = 0; var opens = 0; var binds = 0
        var backendTimes: [Double] = []
        do {
            guard let resources = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let models = resources.appendingPathComponent("ModelCandidates/denoiser-sharded")
            let fixture = resources.appendingPathComponent("ModelCandidates/denoiser-pair-reuse")
            let contract = try readShardedContract(folder: models)
            guard let stages = contract.stages else { throw ShardError.manifestMismatch("sharded.stages missing") }
            let pair = stages[1]
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(PairReuseContract.self, from: Data(contentsOf: fixture.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion, 1, "reuse.schema_version")
            try checkManifest(ref.profile, "denoiser-pair-reuse", "reuse.profile")
            try checkManifest(ref.assetFile, pair.assetFile, "reuse.asset_file")
            try checkManifest(ref.functionName, "main", "reuse.function_name")
            try checkManifest(ref.outputName, "hidden_out", "reuse.output_name")
            try checkManifest(ref.layerIndices, [0,1], "reuse.layer_indices")
            try checkManifest(ref.inputShapes, pair.inputShapes, "reuse.input_shapes")
            try checkManifest(ref.outputShape, [1,13,2048], "reuse.output_shape")
            try checkManifest(ref.runCount, 4, "reuse.run_count")
            try checkManifest(ref.scalarType, "float32", "reuse.scalar_type")
            try checkManifest(ref.byteOrder, "little", "reuse.byte_order")
            try checkManifest(ref.weightStorageDtype, "float16", "reuse.weight_storage_dtype")
            try checkManifest(ref.computeDtype, "float32", "reuse.compute_dtype")
            try checkManifest(ref.checkpointDigest, contract.checkpoint.digest, "reuse.checkpoint_digest")
            try checkManifest(ref.checkpointFullSha256Verified, true, "reuse.checkpoint_full_sha256_verified")
            try checkManifest(ref.expectedFile, "hidden-reference.f32le", "reuse.expected_file")
            try checkManifest(ref.originalFp32ReferenceFile, "hidden-fp32-reference.f32le", "reuse.original_fp32_reference_file")
            try checkManifest(ref.absoluteTolerance, contract.absoluteTolerance, "reuse.absolute_tolerance")
            try checkManifest(ref.relativeTolerance, contract.relativeTolerance, "reuse.relative_tolerance")
            try checkManifest(ref.relativeRmseTolerance, contract.relativeRmseTolerance, "reuse.relative_rmse_tolerance")
            var filenames = Set(ref.inputShapes.keys.map { "\($0).f32le" })
            filenames.formUnion([ref.expectedFile, ref.originalFp32ReferenceFile])
            try checkManifest(Set(ref.artifactSha256.keys), filenames, "reuse.artifact_sha256.keys")
            var prepared: [String: NDArray] = [:]
            for (name, shape) in ref.inputShapes {
                let values = try readFloats(folder: fixture, name: "\(name).f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                prepared[name] = NDArray(scalars: values, shape: shape)
            }
            let inputs = prepared
            let count = ref.outputShape.reduce(1,*)
            let expected = try readFloats(folder: fixture, name: ref.expectedFile, count: count, hashes: ref.artifactSha256)
            let original = try readFloats(folder: fixture, name: ref.originalFp32ReferenceFile, count: count, hashes: ref.artifactSha256)
            lines.append("SHA-256: VERIFIED eight fixed input tensors and both host output references; checkpoint contract matched")
            guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
            guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
            lines.append("Phase: opening one pair model and binding its stateless function once")
            saveProgress(lines, profile: .denoiser)
            await progress(lines.joined(separator: "\n"))
            let openStart = clock.now
            let probe = try await CoreAIModelProbe(contentsOf: models.appendingPathComponent(ref.assetFile), options: .cpuOnly)
            opens += 1
            let openMs = ms(openStart.duration(to: clock.now))
            var setup = CoreAIModelProbe.StatelessTimings()
            let function = try probe.loadStatelessFunction(functionNamed: ref.functionName, inputs: inputs, timings: &setup)
            binds += 1
            // Explicit lifetime extension keeps BOTH handles through every run.
            // It does not assert that the backend pins all weights in RAM.
            defer {
                withExtendedLifetime(function) {}
                withExtendedLifetime(probe) {}
            }
            lines.append("Setup: model-open ms \(openMs); function-bind ms \(setup.functionBindMs); input-validation ms \(setup.inputValidationMs)")
            for iteration in 1...4 {
                guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
                lines.append("Before retained run \(iteration)/4 footprint MiB: \(Double(footprint)/1048576)")
                guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
                let phase = "Phase: retained run \(iteration)/4, same model/function and fixed inputs"
                saveProgress(lines + [phase], profile: .denoiser)
                await progress((lines + [phase]).joined(separator: "\n"))
                let result = try await runRetainedPair(function: function, inputs: inputs, shape: ref.outputShape)
                lines.append("Retained run \(iteration)/4: backend-await ms \(result.backendMs); output-readback+checks ms \(result.copyMs)")
                let stats = try compare(actual: result.values, expected: expected, contract: contract)
                let baseline = try compare(actual: result.values, expected: original, contract: contract)
                lines.append("Retained run \(iteration) storage16 reference: max abs error \(stats.maxError); RMSE \(stats.rmse); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                lines.append("Retained run \(iteration) original FP32 reference: max abs error \(baseline.maxError); RMSE \(baseline.rmse); normalized RMSE \(baseline.normalizedRmse); \(baseline.passed ? "PASS" : "NOT PASSED")")
                guard stats.passed, baseline.passed else { throw ProbeError.numericalParityFailed }
                completed += 1; backendTimes.append(result.backendMs)
                saveProgress(lines, profile: .denoiser)
                await progress(lines.joined(separator: "\n"))
                await Task.yield()
            }
            let repeatedMean = backendTimes.dropFirst().reduce(0,+) / 3
            lines.append("First retained backend-await ms: \(backendTimes[0])")
            lines.append("Repeated retained backend-await mean ms (runs 2-4): \(repeatedMean)")
            if repeatedMean > 0 {
                lines.append("First/repeated backend-await ratio: \(backendTimes[0] / repeatedMean); component-only, NOT a full-loop speedup")
            }
            lines.append("Retained pair dual parity: PASS (4/4 calls; same model AND function)")
        } catch {
            lines.append("Retained pair dual parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Successful model opens: \(opens); function binds: \(binds); completed parity calls: \(completed)/4; total probe ms: \(ms(start.duration(to: clock.now)))")
        lines.append("Memory: sampled app process, NOT total model/device RAM or exact peak; native residency and cold-cache state unmeasured")
        saveProgress(lines, profile: .denoiser)
        return lines.joined(separator: "\n")
    }

    private static func runRetainedPair(function: InferenceFunction, inputs: [String: NDArray], shape: [Int]) async throws
        -> (values: [Float], backendMs: Double, copyMs: Double) {
        let clock = ContinuousClock(); let start = clock.now
        var outputs = try await function.run(inputs: inputs)
        let copyStart = clock.now
        guard let value = outputs.remove("hidden_out"), let tensor = value.ndArray else { throw ProbeError.invalidOutputTensor }
        let values = try copiedValues(tensor, shape: shape)
        return (values, ms(start.duration(to: copyStart)), ms(copyStart.duration(to: clock.now)))
    }

    static func runDenoiserLoop(profilePhases: Bool = false, retainStages: Bool = false, progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: COMPLETE 24-layer denoiser / 4 feedback Euler ODE steps / NOT instrument generation",
                     "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
                     "Schedule: 1 -> 0.75 -> 0.5 -> 0.25 -> 0; shift=1; guidance=1; timestep_r=timestep",
                     "Synthetic conditioning; no CFG, KV cache, recording or waveform generation",
                     "Existing 14 assets per step; detached CPU copies; native cache eviction NOT GUARANTEED",
                     "Process footprint stop before each stage: 2000 MiB; NOT an OS peak-memory guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        if retainStages {
            lines.append("Execution strategy: retain all 14 model/function handles across four steps, loaded lazily on first use")
            lines.append("No extra warmup or stage retry; numerical inputs/schedule/references unchanged")
            lines.append("Retention does NOT guarantee backend weight residency or a 122x whole-loop speedup")
        }
        if profilePhases {
            lines.append("Timing profile: API boundaries / same four-step case / no extra warmup or stage retries")
            lines.append("model-open includes asset validation + AIModel init; function-bind measures loadFunction")
            lines.append("backend-await measures function.run await, NOT isolated kernel time; deferred work may appear in output-copy")
            lines.append("output-copy includes output validation, Float read-back, finite checks and fresh CPU NDArray creation")
            lines.append("Phase totals cover completed stages only; outside-stage includes logging, UI waits, resource release and any unfinished stage")
        }
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var completedSteps = 0; var completedStages = 0
        var phaseTotals = StagePhases()
        var retainedStages: [String: RetainedStage] = [:]
        var retention = RetentionStats()
        do {
            // Hold every cached model/function through the whole measured loop.
            // Cleanup occurs after this scope, including failure paths.
            defer { withExtendedLifetime(retainedStages) {} }
            guard let resources = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let folder = resources.appendingPathComponent("ModelCandidates/denoiser-sharded")
            let fixture = resources.appendingPathComponent("ModelCandidates/denoiser-loop-4")
            lines.append("Phase: validating sharded model manifest")
            saveProgress(lines, profile: .denoiser)
            let contract = try readShardedContract(folder: folder)
            guard let stages = contract.stages else { throw ShardError.manifestMismatch("sharded.stages missing") }
            lines.append("Sharded model manifest: PASS; decoding self-contained loop fixture")
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let loop = try decoder.decode(LoopContract.self, from: Data(contentsOf: fixture.appendingPathComponent("reference.json")))
            let schedule: [Float] = [1, 0.75, 0.5, 0.25, 0]
            try checkManifest(loop.schemaVersion, 2, "loop.schema_version")
            try checkManifest(loop.profile, "denoiser-loop-4", "loop.profile")
            try checkManifest(loop.method, "euler_ode", "loop.method")
            try checkManifest(loop.timesteps, schedule, "loop.timesteps")
            try checkManifest(loop.steps.count, 4, "loop.steps.count")
            try checkManifest(loop.shift, 1, "loop.shift")
            try checkManifest(loop.guidanceScale, 1, "loop.guidance_scale")
            try checkManifest(loop.kvCacheEnabled, false, "loop.kv_cache_enabled")
            try checkManifest(loop.inputShapes, Profile.denoiser.inputShapes, "loop.input_shapes")
            try checkManifest(loop.outputShape, [1,25,64], "loop.output_shape")
            try checkManifest(loop.weightStorageDtype, "float16", "loop.weight_storage_dtype")
            try checkManifest(loop.computeDtype, "float32", "loop.compute_dtype")
            try checkManifest(loop.scalarType, "float32", "loop.scalar_type")
            try checkManifest(loop.byteOrder, "little", "loop.byte_order")
            try checkManifest(loop.checkpointFullSha256Verified, true, "loop.checkpoint_full_sha256_verified")
            try checkManifest(loop.checkpointDigest, contract.checkpoint.digest, "loop.checkpoint_digest")
            try checkManifest(loop.absoluteTolerance, contract.absoluteTolerance, "loop.absolute_tolerance")
            try checkManifest(loop.relativeTolerance, contract.relativeTolerance, "loop.relative_tolerance")
            try checkManifest(loop.relativeRmseTolerance, contract.relativeRmseTolerance, "loop.relative_rmse_tolerance")
            try checkManifest(Set(loop.sharedInputSha256.keys), Set(["noise", "condition", "context"]), "loop.shared_input_sha256.keys")
            try checkManifest(loop.inputFiles, ["noise":"input-noise.f32le", "condition":"input-condition.f32le", "context":"input-context.f32le"], "loop.input_files")
            var expectedFiles = Set<String>()
            for (index, step) in loop.steps.enumerated() {
                let n = index + 1
                let field = "loop.steps[\(index)]"
                try checkManifest(step.index, n, "\(field).index")
                try checkManifest(step.timestep, schedule[index], "\(field).timestep")
                try checkManifest(step.nextTimestep, schedule[index+1], "\(field).next_timestep")
                try checkManifest(step.timestepR, step.timestep, "\(field).timestep_r")
                try checkManifest(step.velocityFile, "step-\(n)-velocity.f32le", "\(field).velocity_file")
                try checkManifest(step.latentFile, "step-\(n)-latent.f32le", "\(field).latent_file")
                try checkManifest(step.originalVelocityFile, "step-\(n)-velocity-fp32.f32le", "\(field).original_velocity_file")
                try checkManifest(step.originalLatentFile, "step-\(n)-latent-fp32.f32le", "\(field).original_latent_file")
                expectedFiles.formUnion([step.velocityFile, step.latentFile, step.originalVelocityFile, step.originalLatentFile])
            }
            let referenceFiles = expectedFiles
            expectedFiles.formUnion(loop.inputFiles.values)
            try checkManifest(Set(loop.artifactSha256.keys), expectedFiles, "loop.artifact_sha256.keys")
            var references: [String: [Float]] = [:]
            for name in referenceFiles {
                references[name] = try readFloats(folder: fixture, name: name, count: 1600, hashes: loop.artifactSha256)
            }
            var shared: [String: NDArray] = [:]
            var latent: [Float] = []
            for name in ["noise", "condition", "context"] {
                guard let filename = loop.inputFiles[name],
                      let shape = Profile.denoiser.inputShapes[name] else { throw ShardError.invalidManifest }
                try checkManifest(loop.sharedInputSha256[name], loop.artifactSha256[filename], "loop.input_sha256[\(name)]")
                // Inputs belong to the reference fixture, not to a separately
                // generated Mac random case. Native model weights stay unchanged.
                let values = try readFloats(folder: fixture, name: filename, count: shape.reduce(1,*), hashes: loop.artifactSha256)
                if name == "noise" { latent = values }
                else { shared[name] = NDArray(scalars: values, shape: shape) }
            }
            lines.append("SHA-256: VERIFIED 16 references + 3 self-contained inputs; checkpoint contract matched")
            for step in loop.steps {
                let stepStart = clock.now
                var stepPhases = StagePhases()
                var state = shared
                state["noise"] = NDArray(scalars: latent, shape: [1,25,64])
                state["timestep"] = NDArray(scalars: [step.timestep], shape: [1])
                state["timestep_r"] = NDArray(scalars: [step.timestepR], shape: [1])
                lines.append("Step \(step.index)/4: t=\(step.timestep), next=\(step.nextTimestep), timestep_r=\(step.timestepR)")
                for (index, stage) in stages.enumerated() {
                    guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
                    lines.append("Before step \(step.index) stage \(index+1)/14 footprint MiB: \(Double(footprint)/1048576)")
                    guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
                    let reused = retainedStages[stage.assetFile] != nil
                    let action = retainStages ? (reused ? "running retained function" : "opening/binding/first run") : "loading/running"
                    let phase = "Phase: step \(step.index)/4 stage \(index+1)/14 \(stage.assetFile) \(action)"
                    saveProgress(lines + [phase], profile: .denoiser)
                    await progress((lines + [phase]).joined(separator: "\n"))
                    var inputs: [String: NDArray] = [:]
                    for name in stage.inputShapes.keys { inputs[name] = state[name] }
                    let result: StageResult
                    if retainStages {
                        result = try await runRetainedStage(stage, folder: folder, inputs: inputs,
                                                           pool: &retainedStages, stats: &retention)
                    } else {
                        result = try await runDetachedStage(stage, folder: folder, inputs: inputs)
                    }
                    for (name, tensor) in result.values {
                        guard let target = stage.outputBindings?[name] else { throw ShardError.invalidManifest }
                        state[target] = tensor
                    }
                    completedStages += 1
                    lines.append("Step \(step.index) stage \(index+1)/14 \(stage.assetFile): load ms \(result.loadMs); run+detached-copy ms \(result.runMs)")
                    if retainStages {
                        lines.append("Retention step \(step.index) stage \(index+1): \(reused ? "reused" : "first-use"); live cached model/function entries: \(retainedStages.count)")
                    }
                    if profilePhases {
                        stepPhases.add(result.phases)
                        phaseTotals.add(result.phases)
                        lines.append("Timing step \(step.index) stage \(index+1)/14: \(result.phases.report)")
                    }
                    saveProgress(lines, profile: .denoiser)
                    await progress(lines.joined(separator: "\n"))
                    await Task.yield()
                }
                guard let tensor = state["velocity"] else { throw ProbeError.invalidOutputTensor }
                let view = tensor.view(as: Float.self)
                let velocity = view.withUnsafePointer { pointer, _, _ in Array(UnsafeBufferPointer(start: pointer, count: 1600)) }
                let dt = step.timestep - step.nextTimestep
                // Float32 multiply then subtract; feed THIS result back, never a host reference.
                latent = zip(latent, velocity).map { x, v in
                    let delta = v * dt
                    return x - delta
                }
                for (label, actual, file) in [("velocity partition", velocity, step.velocityFile),
                                               ("velocity original FP32", velocity, step.originalVelocityFile),
                                               ("latent partition", latent, step.latentFile),
                                               ("latent original FP32", latent, step.originalLatentFile)] {
                    guard let expected = references[file] else { throw ShardError.invalidManifest }
                    let stats = try compare(actual: actual, expected: expected, contract: contract)
                    lines.append("Step \(step.index) \(label): max abs error \(stats.maxError); RMSE \(stats.rmse); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                    guard stats.passed else { throw ProbeError.numericalParityFailed }
                }
                completedSteps += 1
                lines.append("Step \(step.index)/4 dual velocity+latent parity: PASS; total step ms: \(ms(stepStart.duration(to: clock.now)))")
                if profilePhases {
                    lines.append("Timing step \(step.index) totals (\(stepPhases.count)/14 completed stages): \(stepPhases.report)")
                }
                saveProgress(lines, profile: .denoiser)
                await progress(lines.joined(separator: "\n"))
            }
            if retainStages {
                try checkManifest(retention.modelOpens, 14, "retention.model_opens")
                try checkManifest(retention.functionBinds, 14, "retention.function_binds")
                try checkManifest(retention.cacheHits, 42, "retention.cache_hits")
                try checkManifest(retention.completedCalls, 56, "retention.completed_calls")
                try checkManifest(retainedStages.count, 14, "retention.cached_entries")
            }
            lines.append("Four-step Euler loop dual parity: PASS (feedback at all four steps)")
        } catch {
            lines.append("Four-step Euler loop dual parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
        }
        if retainStages {
            lines.append("Retention totals: model opens \(retention.modelOpens); function binds \(retention.functionBinds); cache hits \(retention.cacheHits); completed calls \(retention.completedCalls); cached entries before release \(retainedStages.count)")
            lines.append("Phase: releasing Swift-held model/function pool; native cache eviction NOT GUARANTEED")
            saveProgress(lines, profile: .denoiser)
        }
        retainedStages.removeAll()
        let usage = memory.finish()
        let totalMs = ms(start.duration(to: clock.now))
        if profilePhases {
            lines.append("Timing completed stages: \(phaseTotals.count)/56")
            lines.append("Timing totals: \(phaseTotals.report)")
            lines.append("Timing completed-stage measured ms: \(phaseTotals.measuredMs)")
            lines.append("Timing outside-stage ms: \(totalMs - phaseTotals.measuredMs)")
            lines.append("Largest measured phase: \(phaseTotals.dominantPhase)")
        }
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed parity steps: \(completedSteps)/4; completed stages: \(completedStages)/56; total loop probe ms: \(totalMs)")
        lines.append("Memory scope: sampled app process footprint, NOT total model/device RAM or exact peak; cache behavior unverified")
        saveProgress(lines, profile: .denoiser)
        return lines.joined(separator: "\n")
    }

    static func runShardedDenoiser(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let result = await runShardedDenoiserAttempt(priorReports: [], progress: progress)
        return result.report
    }

    static func runDenoiserStability(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        var lines = ["Test: sharded denoiser repeatability / 3 attempts in the SAME app process",
                     "Same prepared assets and synthetic inputs; fresh input tensors each attempt",
                     "No cache clearing, app restart or automatic retry of a stage",
                     "Cache residency, compilation reuse and accelerator placement NOT MEASURED",
                     "This is NOT cold-launch testing or instrument generation"]
        var passed = 0
        var attempted = 0
        for iteration in 1...3 {
            let prefix = lines + ["", "=== Attempt \(iteration)/3 ==="]
            let result = await runShardedDenoiserAttempt(priorReports: prefix) { text in
                progress(prefix.joined(separator: "\n") + "\n" + text)
            }
            attempted += 1
            if result.passed { passed += 1 }
            lines = prefix + [result.report]
            saveProgress(lines, profile: .denoiser)
            if result.stopRepeating {
                lines.append("Remaining attempts skipped: preparation/manifest or process-footprint guard failed.")
                break
            }
        }
        lines.append("Repeatability: \(passed == 3 ? "PASS" : "NOT PASSED"); dual parity passed \(passed)/3; attempted \(attempted)/3")
        saveProgress(lines, profile: .denoiser)
        return lines.joined(separator: "\n")
    }

    private static func runShardedDenoiserAttempt(priorReports: [String], progress: @escaping @MainActor @Sendable (String) -> Void)
        async -> (report: String, passed: Bool, stopRepeating: Bool) {
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: COMPLETE 24-layer denoiser / 14 sequential assets / synthetic latents / NOT instrument generation",
                     "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
                     "Execution: one forward; detached CPU copies between assets; native cache eviction NOT GUARANTEED",
                     "Process footprint stop before next stage: 2000 MiB; not an OS peak-memory guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        let clock = ContinuousClock()
        let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var completed = 0
        var passed = false
        var stopRepeating = false
        do {
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates/denoiser-sharded"),
                  FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append("Prepare on Mac: bash scripts/prepare_coreai_denoiser.sh; rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let contract = try readShardedContract(folder: folder)
            guard let stages = contract.stages else { throw ShardError.invalidManifest }
            var state: [String: NDArray] = [:]
            for (name, shape) in Profile.denoiser.inputShapes {
                let values = try readFloats(folder: folder, name: "\(name).f32le", count: shape.reduce(1,*), hashes: contract.artifactSha256)
                state[name] = NDArray(scalars: values, shape: shape)
            }
            let expected = try readFloats(folder: folder, name: "velocity-reference.f32le", count: 1600, hashes: contract.artifactSha256)
            let original = try readFloats(folder: folder, name: "velocity-fp32-reference.f32le", count: 1600, hashes: contract.artifactSha256)
            lines.append("Reference binary SHA-256: VERIFIED; both FP32 references; all 24 layers in ordered pairs")
            for (index, stage) in stages.enumerated() {
                guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
                lines.append("Before stage \(index+1)/14 footprint MiB: \(Double(footprint)/1048576)")
                guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
                let phase = "Phase: stage \(index+1)/14 \(stage.assetFile) loading/running"
                saveProgress(priorReports + lines + [phase], profile: .denoiser)
                await progress((lines + [phase]).joined(separator: "\n"))
                var inputs: [String: NDArray] = [:]
                for name in stage.inputShapes.keys { inputs[name] = state[name] }
                let result = try await runDetachedStage(stage, folder: folder, inputs: inputs)
                for (name, tensor) in result.values {
                    guard let target = stage.outputBindings?[name] else { throw ShardError.invalidManifest }
                    state[target] = tensor
                }
                completed += 1
                lines.append("Stage \(completed)/14 \(stage.assetFile): load ms \(result.loadMs); run+detached-copy ms \(result.runMs)")
                saveProgress(priorReports + lines, profile: .denoiser)
                await progress(lines.joined(separator: "\n"))
                await Task.yield()
            }
            guard let output = state["velocity"] else { throw ProbeError.invalidOutputTensor }
            let view = output.view(as: Float.self)
            let actual = view.withUnsafePointer { pointer, _, _ in Array(UnsafeBufferPointer(start: pointer, count: 1600)) }
            let stats = try compare(actual: actual, expected: expected, contract: contract)
            let baseline = try compare(actual: actual, expected: original, contract: contract)
            lines.append("Partition reference max abs error: \(stats.maxError); RMSE: \(stats.rmse); normalized RMSE: \(stats.normalizedRmse)")
            lines.append("Original FP32 baseline max abs error: \(baseline.maxError); RMSE: \(baseline.rmse); normalized RMSE: \(baseline.normalizedRmse)")
            guard stats.passed, baseline.passed else { throw ProbeError.numericalParityFailed }
            lines.append("Sharded denoiser dual parity: PASS (one complete forward, all 24 layers)")
            passed = true
        } catch {
            lines.append("Sharded denoiser dual parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
            stopRepeating = error is ShardError
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed stages: \(completed)/14; complete forward executed: \(completed == 14 ? "YES" : "NO"); total probe ms: \(ms(start.duration(to: clock.now)))")
        lines.append("Memory scope: whole process; sampled, not exact peak; native cache behavior unverified")
        saveProgress(priorReports + lines, profile: .denoiser)
        return (lines.joined(separator: "\n"), passed, stopRepeating)
    }

    private static func readShardedContract(folder: URL) throws -> Contract {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let contract = try decoder.decode(Contract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
        let pairInputs = ["hidden": [1,13,2048], "condition": [1,64,2048], "temb": [1,6,2048],
                          "rope_cos": [1,13,128], "rope_sin": [1,13,128], "self_mask": [1,1,13,13],
                          "local_mask": [1,1,13,13], "cross_mask": [1,1,13,64]]
        var frontOutputs = pairInputs; frontOutputs.removeValue(forKey: "condition")
        frontOutputs["condition_projected"] = [1,64,2048]; frontOutputs["final_temb"] = [1,2048]
        var frontBindings: [String: String] = [:]
        for name in frontOutputs.keys { frontBindings[name] = name == "condition_projected" ? "condition" : name }
        let assets = ["front.aimodel"] + stride(from: 0, to: 24, by: 2).map {
            String(format: "blocks-%02d-%02d.aimodel", $0, $0 + 1)
        } + ["tail.aimodel"]
        try checkManifest(contract.schemaVersion, 1, "sharded.schema_version")
        try checkManifest(contract.profile, "denoiser-sharded", "sharded.profile")
        try checkManifest(contract.assetFile, "sharded-pipeline", "sharded.asset_file")
        try checkManifest(contract.layerCount, 24, "sharded.layer_count")
        try checkManifest(contract.outputName, "velocity", "sharded.output_name")
        try checkManifest(contract.outputShape, [1,25,64], "sharded.output_shape")
        try checkManifest(contract.scalarType, "float32", "sharded.scalar_type")
        try checkManifest(contract.byteOrder, "little", "sharded.byte_order")
        try checkManifest(contract.latentFrames, 25, "sharded.latent_frames")
        try checkManifest(contract.inputShapes, Profile.denoiser.inputShapes, "sharded.input_shapes")
        try checkManifest(contract.weightStorageDtype, "float16", "sharded.weight_storage_dtype")
        try checkManifest(contract.computeDtype, "float32", "sharded.compute_dtype")
        try checkManifest(contract.originalFp32ReferenceFile, "velocity-fp32-reference.f32le", "sharded.original_fp32_reference_file")
        try checkManifest(contract.checkpointFullSha256Verified, true, "sharded.checkpoint_full_sha256_verified")
        try checkManifest(contract.checkpoint.digest, Profile.denoiser.checkpointDigest, "sharded.checkpoint.digest")
        try checkManifest(contract.absoluteTolerance, 1e-4, "sharded.absolute_tolerance")
        try checkManifest(contract.relativeTolerance, 1e-3, "sharded.relative_tolerance")
        try checkManifest(contract.relativeRmseTolerance, 1e-3, "sharded.relative_rmse_tolerance")
        guard let stages = contract.stages else { throw ShardError.manifestMismatch("sharded.stages missing") }
        try checkManifest(stages.count, 14, "sharded.stages.count")
        try checkManifest(stages.map({ $0.assetFile }), assets, "sharded.stages.asset_file order")
        for (index, stage) in stages.enumerated() {
            let field = "sharded.stages[\(index)]"
            if index == 0 {
                try checkManifest(stage.layerIndices, [], "\(field).layer_indices")
                try checkManifest(stage.inputShapes, Profile.denoiser.inputShapes, "\(field).input_shapes")
                try checkManifest(stage.outputShapes, frontOutputs, "\(field).output_shapes")
                try checkManifest(stage.outputBindings, frontBindings, "\(field).output_bindings")
            } else if index == 13 {
                try checkManifest(stage.layerIndices, [], "\(field).layer_indices")
                try checkManifest(stage.inputShapes, ["hidden":[1,13,2048],"final_temb":[1,2048]], "\(field).input_shapes")
                try checkManifest(stage.outputShapes, ["velocity":[1,25,64]], "\(field).output_shapes")
                try checkManifest(stage.outputBindings, ["velocity":"velocity"], "\(field).output_bindings")
            } else {
                try checkManifest(stage.layerIndices, [(index-1)*2,(index-1)*2+1], "\(field).layer_indices")
                try checkManifest(stage.inputShapes, pairInputs, "\(field).input_shapes")
                try checkManifest(stage.outputShapes, ["hidden_out":[1,13,2048]], "\(field).output_shapes")
                try checkManifest(stage.outputBindings, ["hidden_out":"hidden"], "\(field).output_bindings")
            }
        }
        return contract
    }

    // All model/function/output references stay inside this async scope. Fresh
    // CPU-owned NDArrays cross its boundary, not model-backed output buffers.
    private static func runDetachedStage(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray]) async throws
        -> StageResult {
        let clock = ContinuousClock(); let loadStart = clock.now
        let probe = try await CoreAIModelProbe(contentsOf: folder.appendingPathComponent(stage.assetFile), options: .cpuOnly)
        let runStart = clock.now
        let loadMs = ms(loadStart.duration(to: runStart))
        var timings = CoreAIModelProbe.StatelessTimings()
        var outputs = try await probe.runStateless(functionNamed: "main", inputs: inputs, timings: &timings)
        let copyStart = clock.now
        var detached: [String: NDArray] = [:]
        for (name, shape) in stage.outputShapes {
            guard let value = outputs.remove(name), let tensor = value.ndArray, tensor.scalarType == .float32,
                  tensor.shape == shape, tensor.interleaveLayout == nil else { throw ProbeError.invalidOutputTensor }
            let values = try copiedValues(tensor, shape: shape)
            detached[name] = NDArray(scalars: values, shape: shape)
        }
        let end = clock.now
        let runMs = ms(runStart.duration(to: end))
        let copyMs = ms(copyStart.duration(to: end))
        let otherMs = runMs - timings.functionBindMs - timings.inputValidationMs - timings.backendAwaitMs - copyMs
        let phases = StagePhases(count: 1, modelOpenMs: loadMs, functionBindMs: timings.functionBindMs,
                                 inputValidationMs: timings.inputValidationMs, backendAwaitMs: timings.backendAwaitMs,
                                 outputCopyMs: copyMs, runOtherMs: otherMs)
        return (detached, loadMs, runMs, phases)
    }

    private struct StageExecutionFailure: Error {
        let asset: String
        let operation: String
        let reused: Bool
        let isBNNSGraphLoadFailure: Bool
        let details: [String]
    }

    private static func modelErrorDetails(_ error: any Error) -> [String] {
        var lines = ["Error: \(String(reflecting: error))"]
        var current = error as NSError
        for depth in 0..<4 {
            let prefix = depth == 0 ? "Error" : "Underlying error \(depth)"
            lines.append("\(prefix) domain/code: \(current.domain) / \(current.code)")
            lines.append("\(prefix) description: \(current.localizedDescription.prefix(2048))")
            if let reason = current.localizedFailureReason { lines.append("\(prefix) reason: \(reason.prefix(2048))") }
            if let suggestion = current.localizedRecoverySuggestion { lines.append("\(prefix) recovery: \(suggestion.prefix(2048))") }
            guard let underlying = current.userInfo[NSUnderlyingErrorKey] as? NSError, underlying !== current else { break }
            current = underlying
        }
        return lines
    }

    // Select explicit per-generation CPU handle reuse or the one-scope route.
    private static func runPreviewSamplingStage(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray],
                                                aot: PreviewAOTSelection?, reuseStages: Bool,
                                                pool: inout [String: RetainedStage], stats: inout RetentionStats,
                                                allowCacheRepair: Bool, cacheRepair: inout CacheRepairStats,
                                                phaseProgress: (@Sendable (String) -> Void)?) async throws -> StageResult {
        do {
            return try await runPreviewSamplingStageOnce(stage, folder: folder, inputs: inputs, aot: aot,
                reuseStages: reuseStages, pool: &pool, stats: &stats, phaseProgress: phaseProgress)
        } catch let failure as StageExecutionFailure {
            // Only a failed *initial bind*, before function.run(), is eligible.
            // The throwing helper has unwound; this asset never entered the pool.
            // Preserve earlier detached outputs and use the exact same inputs.
            guard allowCacheRepair, aot == nil, cacheRepair.attempts == 0,
                  failure.asset == stage.assetFile, !failure.reused,
                  failure.operation == "function-bind/input-validation",
                  failure.isBNNSGraphLoadFailure, pool[stage.assetFile] == nil else { throw failure }
            cacheRepair.attempts += 1
            let marker = "Targeted CPU cache repair 1/1: \(stage.assetFile); BNNS graphLoadFailed during initial bind; no inference executed for this attempt"
            cacheRepair.events.append(marker)
            cacheRepair.events.append(contentsOf: failure.details)
            phaseProgress?(marker + "\n" + failure.details.joined(separator: "\n"))
            do {
                try AIModelCache.default.deleteEntry(for: folder.appendingPathComponent(stage.assetFile), options: .cpuOnly)
                cacheRepair.events.append("Targeted CPU cache deletion returned for \(stage.assetFile); native eviction not independently verified; retrying this stage once with unchanged inputs")
            } catch {
                throw StageExecutionFailure(asset: stage.assetFile, operation: "targeted-cache-deletion", reused: false,
                    isBNNSGraphLoadFailure: false, details: modelErrorDetails(error))
            }
            try requirePreviewMemoryBudget()
            let repairContext = cacheRepair.events.joined(separator: "\n")
            let retryProgress: @Sendable (String) -> Void = { phase in
                phaseProgress?(repairContext + "\n" + phase)
            }
            phaseProgress?(repairContext)
            // Direct call, not recursive: no second recovery on any retry error.
            let result = try await runPreviewSamplingStageOnce(stage, folder: folder, inputs: inputs, aot: nil,
                reuseStages: reuseStages, pool: &pool, stats: &stats, phaseProgress: retryProgress)
            cacheRepair.successfulRetries += 1
            let completed = "Targeted CPU cache repair: retried stage \(stage.assetFile) completed; generation repair budget exhausted"
            cacheRepair.events.append(completed)
            phaseProgress?(completed)
            return result
        }
    }

    private static func runPreviewSamplingStageOnce(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray],
                                                    aot: PreviewAOTSelection?, reuseStages: Bool,
                                                    pool: inout [String: RetainedStage], stats: inout RetentionStats,
                                                    phaseProgress: (@Sendable (String) -> Void)?) async throws -> StageResult {
        if reuseStages {
            // Fixed case: fourteen DiT shards and one APG asset. No decoder,
            // recording encoder or Qwen handles enter this per-generation pool.
            guard aot == nil, pool[stage.assetFile] != nil || pool.count < 15 else { throw ShardError.invalidManifest }
            return try await runRetainedStage(stage, folder: folder, inputs: inputs, pool: &pool, stats: &stats,
                                              options: .cpuOnly, phaseProgress: phaseProgress)
        }
        return try await runSequentialPreviewStage(stage, folder: folder, inputs: inputs, aot: aot,
                                                   stats: &stats, phaseProgress: phaseProgress)
    }

    // The result owns fresh CPU NDArrays. The only model/function pair lives
    // inside this helper and is dropped on success or failure before the next
    // stage starts. This bounds our Swift ownership, not backend cache residency.
    private static func runSequentialPreviewStage(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray],
                                                   aot: PreviewAOTSelection? = nil, stats: inout RetentionStats,
                                                   phaseProgress: (@Sendable (String) -> Void)? = nil) async throws -> StageResult {
        var owned: [String: RetainedStage] = [:]
        defer { owned.removeAll() }
        let resource: URL
        let options: SpecializationOptions
        if let aot {
            guard let name = aot.assets[stage.assetFile] else { throw ShardError.invalidManifest }
            resource = aot.folder.appendingPathComponent(name)
            options = .default
        } else {
            resource = folder.appendingPathComponent(stage.assetFile)
            options = .cpuOnly
        }
        let result = try await runRetainedStage(stage, folder: folder, inputs: inputs, pool: &owned, stats: &stats,
                                               modelURL: resource, options: options, phaseProgress: phaseProgress)
        if let phaseProgress {
            phaseProgress("Native phase: Swift stage scope release; asset: \(resource.lastPathComponent)")
        }
        return result
    }

    private static func runRetainedStage(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray],
                                         pool: inout [String: RetainedStage], stats: inout RetentionStats, modelURL: URL? = nil,
                                         options: SpecializationOptions = .cpuOnly,
                                         phaseProgress: (@Sendable (String) -> Void)? = nil) async throws -> StageResult {
        let resource = modelURL ?? folder.appendingPathComponent(stage.assetFile)
        var operation = "retained-handle-selection"
        let reused = pool[stage.assetFile] != nil
        do {
            let clock = ContinuousClock()
            let entry: RetainedStage
            var loadMs = 0.0
            var setup = CoreAIModelProbe.StatelessTimings()
            let runStart: ContinuousClock.Instant
            if let cached = pool[stage.assetFile] {
                entry = cached
                stats.cacheHits += 1
                runStart = clock.now
                operation = "retained-input-validation"
                if let phaseProgress {
                    let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                    phaseProgress("Native phase: \(operation); asset: \(resource.lastPathComponent); cpuOnly requested: \(options == .cpuOnly); footprint MiB: \(footprint)")
                }
                let validationStart = clock.now
                try CoreAIModelProbe.validateStatelessInputs(function: entry.function, inputs: inputs)
                setup.inputValidationMs = ms(validationStart.duration(to: clock.now))
            } else {
                operation = "model-open/specialization"
                if let phaseProgress {
                    let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                    phaseProgress("Native phase: \(operation); asset: \(resource.lastPathComponent); cpuOnly requested: \(options == .cpuOnly); footprint MiB: \(footprint)")
                }
                let loadStart = clock.now
                let probe = try await CoreAIModelProbe(contentsOf: resource, options: options)
                stats.modelOpens += 1
                runStart = clock.now
                loadMs = ms(loadStart.duration(to: runStart))
                operation = "function-bind/input-validation"
                if let phaseProgress {
                    let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                    phaseProgress("Native phase: \(operation); asset: \(resource.lastPathComponent); cpuOnly requested: \(options == .cpuOnly); footprint MiB: \(footprint)")
                }
                let function = try probe.loadStatelessFunction(functionNamed: "main", inputs: inputs, timings: &setup)
                stats.functionBinds += 1
                entry = RetainedStage(probe: probe, function: function)
                pool[stage.assetFile] = entry
            }
            // Recheck after bind/validation too: a pre-load guard cannot bound
            // allocations during binding or the following backend invocation.
            operation = "post-bind-memory-guard"
            if let phaseProgress {
                let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                phaseProgress("Native phase: \(operation); asset: \(resource.lastPathComponent); cpuOnly requested: \(options == .cpuOnly); footprint MiB: \(footprint)")
            }
            guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
            guard footprint < 2000 * 1048576 else {
                throw ShardError.memoryBudgetReachedAfterBind(asset: stage.assetFile, mib: Double(footprint)/1048576)
            }
            operation = "backend-await"
            if let phaseProgress {
                let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                phaseProgress("Native phase: \(operation); asset: \(resource.lastPathComponent); cpuOnly requested: \(options == .cpuOnly); footprint MiB: \(footprint)")
            }
            let backendStart = clock.now
            var outputs = try await entry.function.run(inputs: inputs)
            let backendMs = ms(backendStart.duration(to: clock.now))
            operation = "output-readback"
            if let phaseProgress {
                let footprint = CoreAIProcessFootprintSampler.currentFootprint().map { String(Double($0)/1048576) } ?? "unavailable"
                phaseProgress("Native phase: \(operation); asset: \(resource.lastPathComponent); cpuOnly requested: \(options == .cpuOnly); footprint MiB: \(footprint)")
            }
            let copyStart = clock.now
            var detached: [String: NDArray] = [:]
            for (name, shape) in stage.outputShapes {
                guard let value = outputs.remove(name), let tensor = value.ndArray else { throw ProbeError.invalidOutputTensor }
                let values = try copiedValues(tensor, shape: shape)
                detached[name] = NDArray(scalars: values, shape: shape)
            }
            let end = clock.now
            let runMs = ms(runStart.duration(to: end))
            let copyMs = ms(copyStart.duration(to: end))
            let otherMs = runMs - setup.functionBindMs - setup.inputValidationMs - backendMs - copyMs
            let phases = StagePhases(count: 1, modelOpenMs: loadMs, functionBindMs: setup.functionBindMs,
                                     inputValidationMs: setup.inputValidationMs, backendAwaitMs: backendMs,
                                     outputCopyMs: copyMs, runOtherMs: otherMs)
            if stats.phaseTotalsByAsset[stage.assetFile] == nil {
                stats.firstCallPhaseTotals.add(phases)
            } else {
                stats.repeatedCallPhaseTotals.add(phases)
            }
            stats.phaseTotals.add(phases)
            stats.phaseTotalsByAsset[stage.assetFile, default: StagePhases()].add(phases)
            stats.completedCalls += 1
            return (detached, loadMs, runMs, phases)
        } catch {
            let nativeError = error as NSError
            let graphLoadFailure = nativeError.domain == "BNNSCoreAIDelegate.BNNSDelegate.BNNSError"
                && nativeError.code == 1 && String(reflecting: error).contains(".graphLoadFailed(")
            throw StageExecutionFailure(asset: stage.assetFile, operation: operation, reused: reused,
                                        isBNNSGraphLoadFailure: graphLoadFailure,
                                        details: ["Requested model resource: \(resource.lastPathComponent)",
                                                  "Requested policy cpuOnly: \(options == .cpuOnly)"] + modelErrorDetails(error))
        }
    }

    private static func copiedValues(_ tensor: NDArray, shape: [Int]) throws -> [Float] {
        guard tensor.scalarType == .float32, tensor.shape == shape, tensor.interleaveLayout == nil else {
            throw ProbeError.invalidOutputTensor
        }
        let view = tensor.view(as: Float.self)
        let contiguous = view.isContiguous
        let count = shape.reduce(1,*)
        let values = try view.withUnsafePointer { pointer, _, strides in
            if contiguous { return Array(UnsafeBufferPointer(start: pointer, count: count)) }
            guard strides.count == shape.count else { throw ProbeError.invalidOutputTensor }
            for axis in 0..<shape.count {
                guard strides[axis] >= 0 else { throw ProbeError.invalidOutputTensor }
            }
            var copied: [Float] = []; copied.reserveCapacity(count)
            for index in 0..<count {
                var remaining = index; var offset = 0
                for axis in (0..<shape.count).reversed() {
                    offset += (remaining % shape[axis]) * strides[axis]
                    remaining /= shape[axis]
                }
                copied.append(pointer[offset])
            }
            return copied
        }
        guard values.allSatisfy({ $0.isFinite }) else { throw ProbeError.invalidOutputTensor }
        return values
    }

    // Recover existing output files without rerunning the models. A separate
    // per-output report survives subsequent component probes overwriting the
    // shared last-denoiser-probe.txt progress report.
    static func latestSavedBassPreview(window: CoreAIPreviewWindow = .bundled) -> BassPreviewResult? {
        let manager = FileManager.default
        guard let documents = manager.urls(for: .documentDirectory, in: .userDomainMask).first,
              let entries = try? manager.contentsOfDirectory(at: documents,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) else { return nil }
        let root = documents.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        var candidates: [(folder: URL, date: Date)] = []
        for folder in entries where folder.lastPathComponent.hasPrefix(window.stemID + "-preview-") {
            guard let metadata = try? folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]),
                  metadata.isDirectory == true, metadata.isSymbolicLink != true,
                  folder.standardizedFileURL.resolvingSymlinksInPath().path.hasPrefix(root) else { continue }
            let files = [window.stemID + ".wav", "piano-with-" + window.stemID + ".wav", "report.txt"].map { folder.appendingPathComponent($0) }
            let complete = files.allSatisfy { file in
                guard file.standardizedFileURL.resolvingSymlinksInPath().path.hasPrefix(root),
                      let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else { return false }
                return values.isRegularFile == true && values.isSymbolicLink != true
            }
            guard complete,
                  let values = try? files[2].resourceValues(forKeys: [.contentModificationDateKey]),
                  let date = values.contentModificationDate else { continue }
            candidates.append((folder, date))
        }
        for candidate in candidates.sorted(by: { $0.date > $1.date }) {
            guard let report = try? String(contentsOf: candidate.folder.appendingPathComponent("report.txt"), encoding: .utf8),
                  report.contains("WAV generated on device: YES") else { continue }
            return BassPreviewResult(report: report,
                bassURL: candidate.folder.appendingPathComponent(window.stemID + ".wav"),
                mixURL: candidate.folder.appendingPathComponent("piano-with-" + window.stemID + ".wav"))
        }
        return nil
    }

    static func savedDenoiserReport() -> String? {
        guard let url = denoiserReportURL(), let report = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        return "Laporan denoiser tersimpan dari percobaan sebelumnya:\n\n" + report
    }

    private static func denoiserReportURL() -> URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("last-denoiser-probe.txt")
    }

    private static func saveProgress(_ lines: [String], profile: Profile) {
        guard profile.isDenoiser, let url = denoiserReportURL() else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private static func run(profile: Profile) async -> String {
        let component = profile.component
        let testDescription: String
        if profile.isDenoiser {
            testDescription = "pretrained ACE Base COMPLETE 24-layer denoiser / one velocity prediction / synthetic latents"
        } else if profile.isDiT {
            testDescription = "pretrained ACE Base DiT \(profile.isDiTContext ? "blocks 0+1" : "block 0") / synthetic activations"
        } else {
            let inputDescription = profile.isEncoder ? "synthetic audio / posterior mean" : "seeded latent"
            testDescription = "pretrained ACE VAE \(component) / \(inputDescription)"
        }
        var lines = [
            "Runtime: CoreAI",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "CoreAI architecture: \(AIModel.deviceArchitectureName)",
            "Test: \(testDescription) / NOT instrument generation",
            "Requested compute policy: cpuOnly",
            "Accelerator used by a model: NOT MEASURED"
        ]
        if profile.isDenoiser {
            lines.append("Denoiser scope: all 24 layers + input/time/condition/output projections; 25 latent frames; no diffusion loop or waveform generation")
        } else if profile.isDiT {
            lines.append("Block scope: \(profile.isDiTContext ? "layers 0 (local radius 128) -> 1 (full)" : "decoder.layers.0."); \(profile.hiddenTokens) hidden tokens / \(profile.conditionTokens) condition tokens; no audio duration claim")
        } else {
            lines.append("Window profile: \(profile.frames == 25 ? "1.0 s" : "0.2 s")")
        }
        if profile.hasStoredHalfWeights {
            lines.append("Precision profile: FP16 stored weights; FP32 calculations, masks and input/output")
            lines.append("Original FP32 baseline parity: REQUIRED; runtime RAM savings NOT MEASURED")
        }
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
        saveProgress(lines + ["Phase: validating prepared references"], profile: profile)
        do {
            guard let folder = Bundle.main.resourceURL?
                .appendingPathComponent("ModelCandidates/\(profile.folderName)"),
                FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                let ditProfile = profile.isStorage16 ? "storage16" : (profile.isDiTContext ? "context" : "short")
                let script: String
                if profile.isDenoiser { script = "bash scripts/prepare_coreai_denoiser.sh" }
                else if profile.isDiT { script = "bash scripts/prepare_coreai_dit_block.sh --profile \(ditProfile)" }
                else { script = "bash scripts/prepare_coreai_decoder.sh --component \(component) --frames \(profile.frames)" }
                lines.append("Prepare on Mac: \(script); then rebuild the app.")
                throw ProbeError.missingPreparedBundle
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let contract = try decoder.decode(Contract.self, from:
                Data(contentsOf: folder.appendingPathComponent("reference.json")))
            guard contract.schemaVersion == 1, contract.assetFile == "\(component).aimodel",
                  contract.functionName == "main", contract.outputName == profile.outputName,
                  contract.outputShape == profile.outputShape,
                  contract.scalarType == "float32", contract.byteOrder == "little",
                  contract.checkpointFullSha256Verified,
                  contract.checkpoint.digest == profile.checkpointDigest,
                  contract.absoluteTolerance > 0, contract.absoluteTolerance <= 0.001,
                  contract.relativeTolerance > 0, contract.relativeTolerance <= 0.01,
                  contract.relativeRmseTolerance > 0, contract.relativeRmseTolerance <= 0.01 else {
                throw ProbeError.invalidContract
            }
            if profile.isDenoiser {
                guard contract.profile == "denoiser", contract.inputShapes == profile.inputShapes,
                      contract.layerCount == 24, contract.checkpointPrefix == "decoder.", contract.latentFrames == 25 else {
                    throw ProbeError.invalidContract
                }
            } else if profile.isDiT {
                guard contract.inputShapes == profile.inputShapes else {
                    throw ProbeError.invalidContract
                }
                if profile.isDiTContext {
                    guard contract.profile == (profile.isStorage16 ? "storage16" : "context"), contract.layerPrefixes == ["decoder.layers.0.", "decoder.layers.1."],
                          contract.slidingWindow == 128, contract.maskGeometryVerified == true else { throw ProbeError.invalidContract }
                } else if contract.layerPrefix != "decoder.layers.0." { throw ProbeError.invalidContract }
            } else {
                guard (contract.component ?? "decoder") == component,
                      (!profile.isEncoder || contract.posteriorMode == "mean"),
                      contract.inputName == profile.inputName, contract.inputShape == profile.inputShape,
                      contract.sampleRate == 48000, contract.inputFile == "\(profile.inputName).f32le",
                      contract.expectedFile == profile.expectedFile else { throw ProbeError.invalidContract }
            }
            if profile.hasStoredHalfWeights {
                guard contract.weightStorageDtype == "float16", contract.computeDtype == "float32",
                      contract.originalFp32ReferenceFile == profile.originalReferenceFile else { throw ProbeError.invalidContract }
            }
            var tensorInputs: [String: NDArray] = [:]
            for (name, shape) in profile.inputShapes {
                let input = try readFloats(folder: folder, name: "\(name).f32le",
                                           count: shape.reduce(1, *), hashes: contract.artifactSha256)
                tensorInputs[name] = NDArray(scalars: input, shape: shape)
            }
            let expected = try readFloats(folder: folder, name: profile.expectedFile,
                                          count: contract.outputShape.reduce(1, *), hashes: contract.artifactSha256)
            let fp32Baseline: [Float]?
            if profile.hasStoredHalfWeights {
                fp32Baseline = try readFloats(folder: folder, name: profile.originalReferenceFile,
                                              count: contract.outputShape.reduce(1, *), hashes: contract.artifactSha256)
            } else { fp32Baseline = nil }
            if !profile.isDiT && !profile.isDenoiser {
                lines.append("Audio duration / sample rate: \(Double(profile.frames) / 25) s / 48000 Hz stereo")
                lines.append("Input/output shapes: \(profile.inputShape) -> \(contract.outputShape)")
            } else {
                for name in profile.inputShapes.keys.sorted() { lines.append("Input \(name): \(profile.inputShapes[name]!)") }
                lines.append("Output \(profile.outputName): \(contract.outputShape)")
            }
            lines.append("Reference binary SHA-256: VERIFIED; pretrained checkpoint verified by Mac preparation")
            lines.append("Tolerances: abs \(contract.absoluteTolerance), rel \(contract.relativeTolerance), normalized RMSE \(contract.relativeRmseTolerance)")
            let url = folder.appendingPathComponent(contract.assetFile)
            saveProgress(lines + ["Phase: native asset validation"], profile: profile)
            let validationStart = clock.now
            lines.append("Source asset validation: \(AIModelAsset.isValid(at: url))")
            lines.append("Source asset validation ms: \(ms(validationStart.duration(to: clock.now)))")
            saveProgress(lines + ["Phase: loading/specializing complete model"], profile: profile)
            let loadStart = clock.now
            let probe = try await CoreAIModelProbe(contentsOf: url, options: .cpuOnly)
            loaded = true
            lines.append("Validated load/specialization ms: \(ms(loadStart.duration(to: clock.now)))")
            lines.append("Function names: \(probe.model.functionNames)")
            saveProgress(lines + ["Phase: model loaded; reading descriptors"], profile: profile)
            if let descriptor = probe.model.functionDescriptor(for: contract.functionName) {
                for name in profile.inputShapes.keys.sorted() {
                    if let value = descriptor.inputDescriptor(of: name), case .ndArray(let array) = value {
                        lines.append("Runtime input \(name) descriptor: \(array.shape), \(array.scalarType), interleave \(String(describing: array.interleaveLayout))")
                    }
                }
                if let value = descriptor.outputDescriptor(of: contract.outputName), case .ndArray(let array) = value {
                    lines.append("Runtime output descriptor: \(array.shape), \(array.scalarType), interleave \(String(describing: array.interleaveLayout))")
                }
            }
            for iteration in 0..<3 {
                saveProgress(lines + ["Phase: inference \(iteration + 1) running"], profile: profile)
                let start = clock.now
                var outputs = try await probe.runStateless(functionNamed: contract.functionName,
                                                           inputs: tensorInputs)
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
                let stats = try compare(actual: actual, expected: expected, contract: contract)
                let label = iteration == 0 ? "First inference" : "Warm inference \(iteration)"
                lines.append("\(label) run+read-back ms: \(elapsed)")
                lines.append("Max abs error: \(stats.maxError); RMSE: \(stats.rmse); normalized RMSE: \(stats.normalizedRmse)")
                guard stats.passed else {
                    throw ProbeError.numericalParityFailed
                }
                if let fp32Baseline {
                    let original = try compare(actual: actual, expected: fp32Baseline, contract: contract)
                    lines.append("Original FP32 baseline max abs error: \(original.maxError); RMSE: \(original.rmse); normalized RMSE: \(original.normalizedRmse)")
                    guard original.passed else { throw ProbeError.originalFP32ParityFailed }
                }
                saveProgress(lines + ["Phase: inference \(iteration + 1) passed"], profile: profile)
            }
            if profile.hasStoredHalfWeights { lines.append("Original FP32 baseline parity: PASS (three runs)") }
            lines.append("Pretrained \(component) parity: PASS (three runs)")
        } catch {
            let detail = error as NSError
            if profile.hasStoredHalfWeights { lines.append("Original FP32 baseline parity: NOT PASSED") }
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
        saveProgress(lines, profile: profile)
        return lines.joined(separator: "\n")
    }

    private static func compare(actual: [Float], expected: [Float], contract: Contract) throws
        -> (maxError: Double, rmse: Double, normalizedRmse: Double, passed: Bool) {
        try compare(actual: actual, expected: expected, absoluteTolerance: contract.absoluteTolerance,
                    relativeTolerance: contract.relativeTolerance, relativeRmseTolerance: contract.relativeRmseTolerance)
    }

    private static func compare(actual: [Float], expected: [Float], absoluteTolerance: Double,
                                relativeTolerance: Double, relativeRmseTolerance: Double) throws
        -> (maxError: Double, rmse: Double, normalizedRmse: Double, passed: Bool) {
        guard actual.count == expected.count, !expected.isEmpty else { throw ProbeError.invalidOutputTensor }
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
            if error > absoluteTolerance + relativeTolerance * abs(Double(reference)) { elementwisePass = false }
        }
        let rmse = sqrt(squaredError / Double(expected.count))
        let referenceRms = sqrt(squaredReference / Double(expected.count))
        let normalizedRmse = rmse / max(referenceRms, 1e-12)
        return (maxError, rmse, normalizedRmse, elementwisePass && normalizedRmse <= relativeRmseTolerance)
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

// Fixed accepted-input component test, separate from production previews.
extension CoreAIDecoderProbe {
    struct FullSongWindowInput: Sendable {
        let jobSHA: String; let run: FullSongJob.Run; let condition: [Float]
        let context: [Float]; let noise: [Float]; let times: [Float]
        let outputParent: URL; let staticAssetsVerified: Bool
    }
    private struct Drum30Contract: Decodable {
        struct Component: Decodable {
            let assetFile: String
            let inputShapes: [String: [Int]]
            let outputShapes: [String: [Int]]
        }
        let profile: String
        let status: String
        let latentFrames: Int
        let conditionTokens: Int
        let weightStorageDtype: String
        let computeDtype: String
        let nativeExports: Int
        let nativeParity: String
        let fullTrajectoryParity: String
        let acceptedReportSha256: String
        let stages: [Contract.Stage]
        let apg: Component
        let decoder: Component
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    static func savedDrumReference30Report(decoderOnly: Bool = false) -> String? {
        guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let kind = decoderOnly ? "drums-reference30-decoder" : "drums-reference30"
        let phaseKind = decoderOnly ? "drum-reference30-decoder" : "drum-reference30"
        let completion = decoderOnly ? "Decoder isolation: COMPLETED" : "Component collection: COMPLETED"
        guard let text = try? String(contentsOf: root.appendingPathComponent("last-\(kind)-report.txt"), encoding: .utf8) else { return nil }
        guard !text.contains(completion),
              let first = text.components(separatedBy: "\n").first,
              let runID = first.components(separatedBy: "; run ").last, UUID(uuidString: runID) != nil,
              let phase = try? String(contentsOf: root.appendingPathComponent("\(phaseKind)-\(runID)-phase.txt"), encoding: .utf8) else { return text }
        return text + "\nLast persisted native phase for this run:\n" + phase
    }

    private static func saveDrum30(_ lines: [String], runID: String, decoderOnly: Bool = false) {
        guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let group = decoderOnly ? "DrumReference30DecoderReports" : "DrumReference30Reports"
        let kind = decoderOnly ? "drums-reference30-decoder" : "drums-reference30"
        let directory = root.appendingPathComponent(group + "/" + runID, isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let text = lines.joined(separator: "\n")
        try? text.write(to: directory.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
        try? text.write(to: root.appendingPathComponent("last-\(kind)-report.txt"), atomically: true, encoding: .utf8)
    }

    private static func drum30Comparison(_ label: String, actual: [Float], expected: [Float],
                                         comparisons: inout Int, passed: inout Int, lines: inout [String]) throws {
        let metric = try compare(actual: actual, expected: expected, absoluteTolerance: 1e-4,
                                 relativeTolerance: 1e-3, relativeRmseTolerance: 1e-3)
        var failures = 0, worstIndex = 0
        var worstRatio = -Double.infinity
        for i in actual.indices {
            guard expected[i].isFinite else { throw ProbeError.invalidFloatData }
            let bound = 1e-4 + 1e-3 * abs(Double(expected[i]))
            let error = abs(Double(actual[i]) - Double(expected[i]))
            if error > bound { failures += 1 }
            if error / bound > worstRatio { worstRatio = error / bound; worstIndex = i }
        }
        comparisons += 1
        if metric.passed { passed += 1 }
        lines.append("Component \(label): \(metric.passed ? "PASS" : "NOT PASSED"); max abs \(metric.maxError); RMSE \(metric.rmse); normalized RMSE \(metric.normalizedRmse)")
        lines.append("Pointwise violations \(failures)/\(actual.count); worst bound ratio \(worstRatio), flat index \(worstIndex); actual \(actual[worstIndex]); reference \(expected[worstIndex]); original abs1e-4+rel1e-3*abs(reference), nRMSE1e-3")
    }

    private static func drum30Stage(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray],
                                     index: Int, runID: String, stats: inout RetentionStats,
                                     repairs: inout CacheRepairStats,
                                     expectedCalls: Int = 16, allowCacheRepair: Bool = true, decoderOnly: Bool = false,
                                     progress: @escaping @MainActor @Sendable (String) -> Void) async throws -> StageResult {
        try requirePreviewMemoryBudget()
        let phase = "Drum30 component \(index)/\(expectedCalls): \(stage.assetFile); decoder-only: \(decoderOnly)"
        await progress(decoderOnly ? "Memeriksa decoder 30 detik saja…" : "Memeriksa komponen drum 30 detik: \(index)/\(expectedCalls)…")
        let phaseKind = decoderOnly ? "drum-reference30-decoder" : "drum-reference30"
        let breadcrumbs = interleavingPhaseProgress(prefix: "Drum30 run \(runID)\n" + phase,
                                                     fileName: "\(phaseKind)-\(runID)-phase.txt")
        breadcrumbs("Before native call; original fixed inputs; cpuOnly requested")
        var unusedPool: [String: RetainedStage] = [:]
        defer { unusedPool.removeAll() }
        return try await runPreviewSamplingStage(stage, folder: folder, inputs: inputs,
            aot: nil, reuseStages: false, pool: &unusedPool, stats: &stats,
            allowCacheRepair: allowCacheRepair, cacheRepair: &repairs, phaseProgress: breadcrumbs)
    }

    static func runDrumReference30DecoderOnly(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(), start = ContinuousClock().now
        let runID = UUID().uuidString
        let memory = CoreAIProcessFootprintSampler()
        var stats = RetentionStats(), repairs = CacheRepairStats()
        var comparisons = 0, passed = 0, completed = false
        var lines = ["Runtime: CoreAI drums30 decoder-only diagnostic; run \(runID)",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "Revision: drums30-decoder-isolation-v1; one unchanged full750-frame decoder call; no DiT/APG models in this process",
            "Exact original exported decoder, saved final latent and raw host audio; FP32; bounds abs1e-4/rel1e-3/nRMSE1e-3",
            "cpuOnly requested; actual placement unmeasured. Native cache state is not controlled. No retry/cache deletion in this isolation test.",
            "Native full50-step trajectory and music listening: NOT EVALUATED; baseline unchanged",
            "Thermal start: \(ProcessInfo.processInfo.thermalState.rawValue); low power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)"]
        saveDrum30(lines, runID: runID, decoderOnly: true)
        do {
            guard CoreAIPreviewWindow.drumReference30DecoderOnlyEnabled,
                  let root = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let folder = root.appendingPathComponent("ModelCandidates/drums-reference30-frozen-diagnostic", isDirectory: true)
            try verifyReferenceIdentity(folder.appendingPathComponent("reference.json"),
                expected: "f2f84e6c34b2fe42678dc38a48d9ca35b20c14ba53f10843aee8b218a24d53ed")
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(Drum30Contract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.decoder.inputShapes, ["latent":[1,64,750]], "decoder-isolation.input")
            try checkManifest(ref.decoder.outputShapes, ["audio":[1,2,1440000]], "decoder-isolation.output")
            try checkManifest(ref.absoluteTolerance, 1e-4, "decoder-isolation.atol")
            try checkManifest(ref.relativeTolerance, 1e-3, "decoder-isolation.rtol")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "decoder-isolation.nrmse")
            let files = ["decoder.aimodel/main.hash", "decoder.aimodel/main.mlirb", "decoder.aimodel/metadata.json",
                         "capture/final-latent.f32le", "capture/final-audio.f32le", "capture/report.json"]
            await progress("Memverifikasi decoder 30 detik dan input acuan…")
            for name in files {
                guard let expected = ref.artifactSha256[name] else { throw ShardError.invalidManifest }
                try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)), expected, "decoder-isolation.hash")
            }
            lines.append("Original manifest, unchanged decoder payload and all captured inputs/references: SHA256 VERIFIED")
            let final = try readFloats(folder: folder, name: "capture/final-latent.f32le", count: 48000, hashes: ref.artifactSha256)
            var planar = [Float](repeating: 0, count: 48000)
            for channel in 0..<64 { for frame in 0..<750 { planar[channel*750+frame] = final[frame*64+channel] } }
            let stage = Contract.Stage(assetFile: ref.decoder.assetFile, layerIndices: [],
                inputShapes: ref.decoder.inputShapes, outputShapes: ref.decoder.outputShapes, outputBindings: nil)
            lines.append("Phase: isolated decoder model open/bind/run; planned calls1, prior DiT/APG opens0")
            saveDrum30(lines, runID: runID, decoderOnly: true)
            let result = try await drum30Stage(stage, folder: folder,
                inputs: ["latent": NDArray(scalars: planar, shape: [1,64,750])], index: 1,
                runID: runID, stats: &stats, repairs: &repairs,
                expectedCalls: 1, allowCacheRepair: false, decoderOnly: true, progress: progress)
            guard let audio = result.values["audio"] else { throw ProbeError.invalidOutputTensor }
            try drum30Comparison("Isolated full30s decoder", actual: try copiedValues(audio, shape: [1,2,1440000]),
                expected: try readFloats(folder: folder, name: "capture/final-audio.f32le", count: 2880000, hashes: ref.artifactSha256),
                comparisons: &comparisons, passed: &passed, lines: &lines)
            lines.append("Backend-await ms \(result.phases.backendAwaitMs); model-open ms \(result.phases.modelOpenMs); bind ms \(result.phases.functionBindMs)")
            try checkManifest(stats.completedCalls, 1, "decoder-isolation.calls")
            completed = true
        } catch {
            if let failure = error as? StageExecutionFailure {
                lines.append("Failing asset: \(failure.asset); operation: \(failure.operation)")
                lines.append(contentsOf: failure.details)
            } else { lines.append(contentsOf: modelErrorDetails(error)) }
        }
        lines.append("Decoder isolation: \(completed ? "COMPLETED" : "NOT COMPLETED"); calls \(stats.completedCalls)/1, opens \(stats.modelOpens), binds \(stats.functionBinds)")
        let verdict = completed ? (passed == 1 ? "PASS" : "NOT PASSED") : "NOT EVALUATED"
        lines.append("Decoder-only strict parity: \(verdict); \(passed)/\(comparisons) comparisons passed; target1")
        lines.append("Cache repair/deletion: DISABLED; retries0; full native trajectory: NOT EVALUATED; baseline/mix/bounds unchanged")
        let usage = memory.finish()
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10ms): \(Double(peak)/1048576); not exact transient/native memory peak") }
        lines.append("Thermal end: \(ProcessInfo.processInfo.thermalState.rawValue); total isolated decoder ms \(ms(start.duration(to: clock.now)))")
        saveDrum30(lines, runID: runID, decoderOnly: true)
        return lines.joined(separator: "\n")
    }

    static func runDrumReference30Components(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(), start = ContinuousClock().now
        let runID = UUID().uuidString
        let memory = CoreAIProcessFootprintSampler()
        var stats = RetentionStats(), repairs = CacheRepairStats()
        var comparisons = 0, passed = 0, completed = false
        var lines = ["Runtime: CoreAI drums30 fixed-input component diagnostic; run \(runID)",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "Revision: drums30-components-v1; 16 serial component calls, no generation or WAV",
            "BF16 learned DiT storage, FP32 compute/I/O; VAE FP32; original bounds unchanged",
            "cpuOnly requested; actual placement unmeasured; Swift scopes released per asset, native eviction not guaranteed",
            "APG receives captured gold velocities and decoder receives captured final latent: independent component checks, not native feedback trajectory",
            "Finite strict failures remain NOT PASSED while other independent components are collected. Nonfinite/native/type/memory errors stop.",
            "Native full 50-step trajectory: NOT EVALUATED; full-song bass/drums/guitar: NOT EVALUATED",
            "Thermal start: \(ProcessInfo.processInfo.thermalState.rawValue); low power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)"]
        saveDrum30(lines, runID: runID)
        do {
            guard let root = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let folder = root.appendingPathComponent("ModelCandidates/drums-reference30-frozen-diagnostic", isDirectory: true)
            try verifyReferenceIdentity(folder.appendingPathComponent("reference.json"),
                expected: "f2f84e6c34b2fe42678dc38a48d9ca35b20c14ba53f10843aee8b218a24d53ed")
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(Drum30Contract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.profile, "drums-reference30-frozen-diagnostic", "drum30.profile")
            try checkManifest(ref.status, "prepared_inactive_component_diagnostic", "drum30.status")
            try checkManifest(ref.latentFrames, 750, "drum30.frames")
            try checkManifest(ref.conditionTokens, 127, "drum30.tokens")
            try checkManifest(ref.weightStorageDtype, "bfloat16", "drum30.storage")
            try checkManifest(ref.computeDtype, "float32", "drum30.compute")
            try checkManifest(ref.nativeExports, 16, "drum30.exports")
            try checkManifest(ref.nativeParity, "NOT EVALUATED", "drum30.native")
            try checkManifest(ref.fullTrajectoryParity, "NOT EVALUATED", "drum30.trajectory")
            try checkManifest(ref.absoluteTolerance, 1e-4, "drum30.atol")
            try checkManifest(ref.relativeTolerance, 1e-3, "drum30.rtol")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "drum30.nrmse")
            try verifyReferenceIdentity(folder.appendingPathComponent("capture/report.json"), expected: ref.acceptedReportSha256)
            try checkManifest(ref.stages.count, 14, "drum30.stages")
            let assets = ref.stages.map { $0.assetFile } + [ref.apg.assetFile, ref.decoder.assetFile]
            try checkManifest(Set(assets).count, 16, "drum30.unique_assets")
            await progress("Memverifikasi asset dan input drum 30 detik…")
            for asset in assets {
                for file in ["main.hash", "main.mlirb", "metadata.json"] {
                    let name = asset + "/" + file
                    guard let expected = ref.artifactSha256[name] else { throw ShardError.invalidManifest }
                    try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)), expected, "drum30.asset_sha256")
                }
            }
            let fixtureNames = ["initial-noise", "condition", "source-context", "timesteps", "final-latent", "final-audio",
                                "step-01-conditional", "step-01-unconditional", "step-01-latent", "step-01-momentum", "step-01-guided"]
            for file in fixtureNames {
                let name = "capture/" + file + ".f32le"
                guard let expected = ref.artifactSha256[name] else { throw ShardError.invalidManifest }
                try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)), expected, "drum30.capture_sha256")
            }
            lines.append("Reviewed export manifest, all 16 native payloads and all 11 captured tensor hashes: VERIFIED before model open")
            let count = 750 * 64
            func floats(_ name: String, _ size: Int) throws -> [Float] {
                try readFloats(folder: folder, name: "capture/" + name + ".f32le", count: size, hashes: ref.artifactSha256)
            }
            let times = try floats("timesteps", 51)
            guard times[0] == 1, times[50] == 0,
                  zip(times, times.dropFirst()).allSatisfy({ pair in pair.0 > pair.1 }) else { throw ShardError.invalidManifest }
            let noise = try floats("initial-noise", count)
            let cond = try floats("condition", 127 * 2048)
            let context = try floats("source-context", 750 * 128)
            let goldConditional = try floats("step-01-conditional", count)
            let goldUnconditional = try floats("step-01-unconditional", count)
            var state: [String: NDArray] = [
                "noise": NDArray(scalars: noise, shape: [1,750,64]),
                "condition": NDArray(scalars: cond, shape: [1,127,2048]),
                "context": NDArray(scalars: context, shape: [1,750,128]),
                "timestep": NDArray(scalars: [times[0]], shape: [1])]
            for (i, stage) in ref.stages.enumerated() {
                var inputs: [String: NDArray] = [:]
                for (name, shape) in stage.inputShapes {
                    guard let value = state[name], value.shape == shape else { throw ProbeError.invalidOutputTensor }
                    inputs[name] = value
                }
                lines.append("Phase: component \(i+1)/16 \(stage.assetFile)")
                saveDrum30(lines, runID: runID)
                let result = try await drum30Stage(stage, folder: folder, inputs: inputs, index: i+1,
                    runID: runID, stats: &stats, repairs: &repairs, progress: progress)
                for (name, value) in result.values {
                    guard let binding = stage.outputBindings?[name] else { throw ShardError.invalidManifest }
                    state[binding] = value
                }
                lines.append("Completed \(stage.assetFile); backend-await ms \(result.phases.backendAwaitMs)")
            }
            guard let velocity = state["velocity"] else { throw ProbeError.invalidOutputTensor }
            let values = try copiedValues(velocity, shape: [2,750,64])
            try drum30Comparison("DiT conditional step1", actual: Array(values.prefix(count)), expected: goldConditional,
                                 comparisons: &comparisons, passed: &passed, lines: &lines)
            try drum30Comparison("DiT unconditional step1", actual: Array(values.suffix(count)), expected: goldUnconditional,
                                 comparisons: &comparisons, passed: &passed, lines: &lines)
            state.removeAll()
            // Isolate APG from any DiT rounding difference: actual captured velocity inputs.
            let update = Contract.Stage(assetFile: ref.apg.assetFile, layerIndices: [],
                inputShapes: ref.apg.inputShapes, outputShapes: ref.apg.outputShapes, outputBindings: nil)
            let updateInputs: [String: NDArray] = [
                "velocity": NDArray(scalars: goldConditional + goldUnconditional, shape: [2,750,64]),
                "latent": NDArray(scalars: noise, shape: [1,750,64]),
                "momentum": NDArray(scalars: [Float](repeating: 0, count: count), shape: [1,750,64]),
                "dt": NDArray(scalars: [times[0]-times[1]], shape: [1])]
            lines.append("Phase: component 15/16 APG on saved gold velocities, initial latent and zero prior momentum")
            saveDrum30(lines, runID: runID)
            let updated = try await drum30Stage(update, folder: folder, inputs: updateInputs, index: 15,
                runID: runID, stats: &stats, repairs: &repairs, progress: progress)
            for (name, gold) in [("latent_next", "latent"), ("momentum_next", "momentum"), ("guided_velocity", "guided")] {
                guard let tensor = updated.values[name] else { throw ProbeError.invalidOutputTensor }
                try drum30Comparison("APG " + name, actual: try copiedValues(tensor, shape: [1,750,64]),
                    expected: try floats("step-01-" + gold, count), comparisons: &comparisons, passed: &passed, lines: &lines)
            }
            let final = try floats("final-latent", count)
            var planar = [Float](repeating: 0, count: count)
            for channel in 0..<64 { for frame in 0..<750 { planar[channel*750+frame] = final[frame*64+channel] } }
            let vae = Contract.Stage(assetFile: ref.decoder.assetFile, layerIndices: [],
                inputShapes: ref.decoder.inputShapes, outputShapes: ref.decoder.outputShapes, outputBindings: nil)
            lines.append("Phase: component 16/16 full 30s decoder from saved final latent; no generated trajectory")
            saveDrum30(lines, runID: runID)
            let decoded = try await drum30Stage(vae, folder: folder,
                inputs: ["latent": NDArray(scalars: planar, shape: [1,64,750])], index: 16,
                runID: runID, stats: &stats, repairs: &repairs, progress: progress)
            guard let audio = decoded.values["audio"] else { throw ProbeError.invalidOutputTensor }
            try drum30Comparison("VAE full30s audio", actual: try copiedValues(audio, shape: [1,2,1440000]),
                expected: try floats("final-audio", 2880000), comparisons: &comparisons, passed: &passed, lines: &lines)
            try checkManifest(stats.completedCalls, 16, "drum30.calls")
            try checkManifest(comparisons, 6, "drum30.comparisons")
            completed = true
        } catch {
            lines.append("Drum30 component collection: NOT COMPLETED")
            if let failure = error as? StageExecutionFailure {
                lines.append("Failing asset: \(failure.asset); operation: \(failure.operation)")
                lines.append(contentsOf: failure.details)
            } else { lines.append(contentsOf: modelErrorDetails(error)) }
        }
        lines.append(contentsOf: repairs.events)
        lines.append("Targeted initial-bind repair: \(repairs.attempts)/1; successful retries \(repairs.successfulRetries)")
        lines.append("Component collection: \(completed ? "COMPLETED" : "NOT COMPLETED"); calls \(stats.completedCalls)/16, opens \(stats.modelOpens), binds \(stats.functionBinds)")
        lines.append("Drum30 component strict parity: \(completed && passed == 6 ? "PASS" : "NOT PASSED"); \(passed)/\(comparisons) comparisons passed; target 6")
        lines.append("Native full 50-step trajectory: NOT EVALUATED; musical listening: NOT EVALUATED; baseline/mix/bounds unchanged")
        let usage = memory.finish()
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10ms): \(Double(peak)/1048576); not exact transient/native memory peak") }
        lines.append("Thermal end: \(ProcessInfo.processInfo.thermalState.rawValue); total component test ms \(ms(start.duration(to: clock.now)))")
        saveDrum30(lines, runID: runID)
        return lines.joined(separator: "\n")
    }
}

extension CoreAIDecoderProbe {
    private struct WindowDecoderContract: Decodable {
        struct Tile: Decodable {
            let windowStart: Int; let windowFrames: Int; let coreStart: Int; let coreEnd: Int; let cropStart: Int
        }
        struct Metric: Decodable {
            let finite: Bool; let parityPassed: Bool; let maxAbsoluteError: Double
            let rmse: Double; let normalizedRmse: Double; let elementwiseFractionOverBound: Double
        }
        let schemaVersion: Int; let profile: String; let status: String; let assetFile: String; let functionName: String
        let inputShapes: [String:[Int]]; let outputShapes: [String:[Int]]
        let weightStorageDtype: String; let computeDtype: String; let scalarType: String; let byteOrder: String
        let vaeCheckpointSha256: String; let captureReportSha256: String
        let absoluteTolerance: Double; let relativeTolerance: Double; let relativeRmseTolerance: Double
        let windowPlan: [Tile]; let tileInputSha256: [String]; let hostChecks: [String:Metric]
        let hostNumericalGatePassed: Bool; let architectureGuardPassed: Bool; let coverageExactlyOnce: Bool
        let crossfade: Bool; let extraLatentPadding: Bool; let nativeParity: String; let fullTrajectoryParity: String
        let artifactSha256: [String:String]
    }

    static func savedDrumWindowReport() -> String? {
        guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let text = try? String(contentsOf: root.appendingPathComponent("last-drums-reference30-window-report.txt"), encoding: .utf8) else { return nil }
        guard !text.contains("Window decoder collection: COMPLETED"),
              let first = text.components(separatedBy: "\n").first,
              let id = first.components(separatedBy: "; run ").last, UUID(uuidString: id) != nil,
              let phase = try? String(contentsOf: root.appendingPathComponent("drum-reference30-window-\(id)-phase.txt"), encoding: .utf8) else { return text }
        return text + "\nLast persisted native phase for this run:\n" + phase
    }

    private static func saveDrumWindow(_ lines: [String], runID: String) {
        guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let folder = root.appendingPathComponent("DrumReference30WindowReports/" + runID, isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let text = lines.joined(separator: "\n")
        try? text.write(to: folder.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
        try? text.write(to: root.appendingPathComponent("last-drums-reference30-window-report.txt"), atomically: true, encoding: .utf8)
    }

    static func runDrumReference30WindowDecoder(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(), start = ContinuousClock().now, runID = UUID().uuidString
        let memory = CoreAIProcessFootprintSampler()
        var pool: [String:RetainedStage] = [:], stats = RetentionStats(), repairs = CacheRepairStats()
        defer { pool.removeAll() }
        var comparisons = 0, passed = 0, completed = false
        var lines = ["Runtime: CoreAI drums30 window decoder diagnostic; run \(runID)",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "Revision: drums30-window256-v1; six decoder calls, one retained model/function; no DiT/APG/generation/WAV",
            "FP32 VAE, fixed accepted full750-frame final latent; window256, core128, halo64; no extra latent padding/crossfade/filter",
            "Original abs1e-4/rel1e-3/nRMSE1e-3; compare six retained cores, five seam regions and full30s assembled audio",
            "cpuOnly requested; placement/cache residency unmeasured; no retry/cache deletion",
            "Memory readiness requires sampled process peak below the existing2000MiB budget; sampled peak is not exact transient/native peak",
            "Thermal start: \(ProcessInfo.processInfo.thermalState.rawValue); low power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)"]
        saveDrumWindow(lines, runID: runID)
        do {
            guard CoreAIPreviewWindow.drumReference30WindowEnabled, let root = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let relative = "drums-reference30-window-decoder-256/reference.json"
            let base = root.appendingPathComponent("ModelCandidates", isDirectory: true)
            guard let stamp = try JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("app-resource-manifest.json"))) as? [String:Any],
                  stamp["selected_profile"] as? String == "drums-reference30-window-decoder",
                  let files = stamp["files"] as? [String:[String:Any]], let identity = files[relative]?["sha256"] as? String else { throw ShardError.invalidManifest }
            try verifyReferenceIdentity(base.appendingPathComponent(relative), expected: identity)
            let folder = base.appendingPathComponent("drums-reference30-window-decoder-256", isDirectory: true)
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(WindowDecoderContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion, 1, "window.schema")
            try checkManifest(ref.profile, "drums-reference30-window-decoder", "window.profile")
            try checkManifest(ref.status, "prepared_window_decoder", "window.status")
            try checkManifest(ref.assetFile, "decoder.aimodel", "window.asset")
            try checkManifest(ref.functionName, "main", "window.function")
            try checkManifest(ref.inputShapes, ["latent":[1,64,256]], "window.input")
            try checkManifest(ref.outputShapes, ["audio":[1,2,491520]], "window.output")
            for value in [ref.weightStorageDtype, ref.computeDtype, ref.scalarType] { try checkManifest(value, "float32", "window.dtype") }
            try checkManifest(ref.byteOrder, "little", "window.byte_order")
            try checkManifest(ref.vaeCheckpointSha256, "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0", "window.checkpoint")
            try checkManifest(ref.captureReportSha256, "f5f84314fa37f49f73502a77f7e3364b31450898231731d64e216bf1b606bfdf", "window.capture")
            try checkManifest(ref.absoluteTolerance, 1e-4, "window.atol")
            try checkManifest(ref.relativeTolerance, 1e-3, "window.rtol")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "window.nrmse")
            guard ref.hostNumericalGatePassed, ref.architectureGuardPassed, ref.coverageExactlyOnce,
                  !ref.crossfade, !ref.extraLatentPadding, ref.hostChecks.count == 12,
                  ref.windowPlan.count == 6, ref.tileInputSha256.count == 6 else { throw ShardError.invalidManifest }
            try checkManifest(ref.nativeParity, "NOT EVALUATED", "window.native")
            try checkManifest(ref.fullTrajectoryParity, "NOT EVALUATED", "window.trajectory")
            let requiredChecks = Set(["overall"] + (0..<6).map { "tile-\($0)" } + (1...5).map { "seam-\($0)" })
            try checkManifest(Set(ref.hostChecks.keys), requiredChecks, "window.host_checks")
            for m in ref.hostChecks.values {
                guard m.finite, m.parityPassed, m.elementwiseFractionOverBound == 0, m.normalizedRmse <= 1e-3,
                      [m.maxAbsoluteError,m.rmse,m.normalizedRmse].allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw ShardError.invalidManifest }
            }
            let starts = [0,64,192,320,448,494]
            for (i,t) in ref.windowPlan.enumerated() {
                guard t.windowFrames == 256, t.windowStart == starts[i], t.coreStart == i*128,
                      t.coreEnd == min(i*128+128,750), t.cropStart == t.coreStart-t.windowStart else { throw ShardError.invalidManifest }
            }
            let names: Set<String> = ["decoder.aimodel/main.hash","decoder.aimodel/main.mlirb","decoder.aimodel/metadata.json",
                                     "final-latent.f32le","final-audio.f32le","capture-report.json"]
            try checkManifest(Set(ref.artifactSha256.keys), names, "window.files")
            try checkManifest(ref.artifactSha256["final-latent.f32le"], "54c574fbcec00d7404d5072258d738b3de5e167728e707a2b271264733d5740c", "window.latent_identity")
            try checkManifest(ref.artifactSha256["final-audio.f32le"], "5f81f20205c39fbba6418f95d5d42d8346fe48f457c8c455c77958c98fabd615", "window.audio_identity")
            try checkManifest(ref.artifactSha256["capture-report.json"], ref.captureReportSha256, "window.report_identity")
            await progress("Memverifikasi decoder berjendela dan acuan 30 detik…")
            for n in names {
                guard let expected = ref.artifactSha256[n] else { throw ShardError.invalidManifest }
                try checkManifest(try streamingSHA256(folder.appendingPathComponent(n)), expected, "window.payload_sha256")
            }
            let latent = try readFloats(folder: folder, name: "final-latent.f32le", count: 48000, hashes: ref.artifactSha256)
            let gold = try readFloats(folder: folder, name: "final-audio.f32le", count: 2880000, hashes: ref.artifactSha256)
            var assembled = [Float](repeating: 0, count: 2880000)
            let stage = Contract.Stage(assetFile: ref.assetFile, layerIndices: [], inputShapes: ref.inputShapes,
                                       outputShapes: ref.outputShapes, outputBindings: nil)
            for (i,t) in ref.windowPlan.enumerated() {
                try requirePreviewMemoryBudget()
                var input = [Float](repeating: 0, count: 16384)
                for c in 0..<64 { for f in 0..<256 { input[c*256+f] = latent[(t.windowStart+f)*64+c] } }
                let inputHash = input.withUnsafeBufferPointer { SHA256.hash(data: Data(buffer: $0)).map { String(format: "%02x", $0) }.joined() }
                try checkManifest(inputHash, ref.tileInputSha256[i], "window.exact_saved_input")
                let phase = "Window\(i+1)/6 start\(t.windowStart), keep\(t.coreStart):\(t.coreEnd); one retained decoder"
                lines.append("Phase: " + phase); saveDrumWindow(lines, runID: runID)
                await progress("Memeriksa decoder berjendela: \(i+1)/6…")
                let breadcrumbs = interleavingPhaseProgress(prefix: "Window decoder run \(runID)\n" + phase,
                    fileName: "drum-reference30-window-\(runID)-phase.txt")
                breadcrumbs("Before native call; fixed input SHA verified; no retry/cache deletion")
                let result = try await runPreviewSamplingStage(stage, folder: folder,
                    inputs: ["latent": NDArray(scalars: input, shape: [1,64,256])], aot: nil, reuseStages: true,
                    pool: &pool, stats: &stats, allowCacheRepair: false, cacheRepair: &repairs, phaseProgress: breadcrumbs)
                guard let tensor = result.values["audio"] else { throw ProbeError.invalidOutputTensor }
                let values = try copiedValues(tensor, shape: [1,2,491520])
                let count = (t.coreEnd-t.coreStart)*1920
                var kept = [Float](repeating: 0, count: count*2), expected = kept
                for c in 0..<2 { for j in 0..<count {
                    let value = values[c*491520+t.cropStart*1920+j], global = c*1440000+t.coreStart*1920+j
                    kept[c*count+j] = value; expected[c*count+j] = gold[global]; assembled[global] = value
                } }
                try drum30Comparison("Window core\(i+1)", actual: kept, expected: expected,
                    comparisons: &comparisons, passed: &passed, lines: &lines)
                lines.append("Window\(i+1) backend-await ms \(result.phases.backendAwaitMs)")
                try requirePreviewMemoryBudget()
            }
            pool.removeAll()
            for (i,t) in ref.windowPlan.dropFirst().enumerated() {
                let center = t.coreStart*1920
                var actual: [Float] = [], expected: [Float] = []
                for c in 0..<2 {
                    let a = c*1440000+center-3840, b = c*1440000+center+3840
                    actual.append(contentsOf: assembled[a..<b]); expected.append(contentsOf: gold[a..<b])
                }
                try drum30Comparison("Window seam\(i+1)", actual: actual, expected: expected,
                    comparisons: &comparisons, passed: &passed, lines: &lines)
            }
            try drum30Comparison("Window full30s assembled audio", actual: assembled, expected: gold,
                comparisons: &comparisons, passed: &passed, lines: &lines)
            try checkManifest(stats.completedCalls, 6, "window.calls")
            try checkManifest(stats.modelOpens, 1, "window.opens")
            try checkManifest(stats.functionBinds, 1, "window.binds")
            try checkManifest(stats.cacheHits, 5, "window.hits")
            try checkManifest(comparisons, 12, "window.comparisons")
            completed = true
        } catch {
            pool.removeAll()
            if let failure = error as? StageExecutionFailure {
                lines.append("Failing asset: \(failure.asset); operation: \(failure.operation)"); lines.append(contentsOf: failure.details)
            } else { lines.append(contentsOf: modelErrorDetails(error)) }
        }
        let usage = memory.finish()
        let memoryPassed = usage.peak.map { $0 < 2000*1048576 } ?? false
        let verdict = completed ? (passed == 12 ? "PASS" : "NOT PASSED") : "NOT EVALUATED"
        lines.append("Window decoder collection: \(completed ? "COMPLETED" : "NOT COMPLETED"); calls \(stats.completedCalls)/6; opens \(stats.modelOpens), binds \(stats.functionBinds), reuse hits \(stats.cacheHits)")
        lines.append("Window decoder strict parity: \(verdict); \(passed)/\(comparisons) comparisons passed; target12")
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10ms): \(Double(peak)/1048576); sampled budget <2000MiB: \(memoryPassed ? "PASS" : "NOT PASSED")") }
        else { lines.append("Sampled memory readiness: NOT EVALUATED; footprint unavailable") }
        lines.append("Window decoder readiness: \(completed && passed == 12 && memoryPassed ? "PASS" : "NOT PASSED"); sampled memory is not exact peak or combined-pipeline residency evidence")
        lines.append("Native full50-step trajectory/music listening: NOT EVALUATED; retries0, cache deletions0; BF16 baseline/FP32 VAE/bounds unchanged")
        lines.append("Thermal end: \(ProcessInfo.processInfo.thermalState.rawValue); total window decoder ms \(ms(start.duration(to: clock.now)))")
        saveDrumWindow(lines, runID: runID)
        return lines.joined(separator: "\n")
    }
}

extension CoreAIDecoderProbe {
    struct Reference30GenerationResult: Sendable {
        let report: String
        let folder: URL?
        let drumURL: URL?
        let mixURL: URL?
        let pianoURL: URL?
    }
    private struct Reference30Configuration: Decodable {
        let schemaVersion: Int; let profile: String; let scope: String
        let samplerBundle: String; let samplerReferenceSha256: String
        let decoderBundle: String; let decoderReferenceSha256: String; let captureReportSha256: String; let sourceAudioSha256: String
        let latentFrames: Int; let conditionTokens: Int; let inferenceSteps: Int; let checkpoints: [Int]
        let referenceListeningGain: Double; let pianoMixGain: Double; let drumMixGain: Double
        let absoluteTolerance: Double; let relativeTolerance: Double; let relativeRmseTolerance: Double
        let feedback: String; let baselineModified: Bool
    }

    private struct Instrument30Contract: Decodable {
        let schemaVersion: Int; let profile: String; let status: String; let roles: [String]
        let presetSha256: [String:String]; let conditionTokens: Int; let conditionShape: [Int]; let maskShape: [Int]
        let captureReportSha256: String; let hostDrumsControlExact: Bool
        let absoluteTolerance: Double; let relativeTolerance: Double; let relativeRmseTolerance: Double
        let samplerWeightStorageDtype: String; let computeDtype: String; let vaeDtype: String
        let nativeExports: Int; let baselineModified: Bool; let paddingTokens: Int; let truncatedTokens: Int
        let fullSongNativeReadiness: String; let bassGuitarNumericalParity: String; let artifactSha256: [String:String]
    }
    private static func reference30Directory(_ runID: String, instrument: String? = nil) -> URL? {
        if let instrument, !["bass","drums","guitar"].contains(instrument) { return nil }
        let group = instrument.map { "InstrumentReference30/" + $0 } ?? "DrumReference30Generation"
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent(group + "/" + runID, isDirectory: true)
    }
    private static func reference30Pointer(_ instrument: String?) -> String {
        instrument.map { "last-instrument-reference30-" + $0 + ".json" } ?? "last-drums-reference30-generation.json"
    }
    private static func saveReference30(_ lines: [String], runID: String, instrument: String? = nil) {
        guard let folder = reference30Directory(runID, instrument: instrument) else { return }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try? lines.joined(separator: "\n").write(to: folder.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
        if let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
           let data = try? JSONSerialization.data(withJSONObject: ["run_id":runID]) {
            try? data.write(to: root.appendingPathComponent(reference30Pointer(instrument)), options: .atomic)
        }
    }
    static func latestReference30Generation(instrument: String? = nil) -> Reference30GenerationResult? {
        if let instrument, !["bass","drums","guitar"].contains(instrument) { return nil }
        guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let data = try? Data(contentsOf: root.appendingPathComponent(reference30Pointer(instrument))),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String:String],
              let id = object["run_id"], UUID(uuidString: id) != nil, let folder = reference30Directory(id, instrument: instrument),
              let folderValues = try? folder.resourceValues(forKeys: [.isDirectoryKey,.isSymbolicLinkKey]),
              folderValues.isDirectory == true, folderValues.isSymbolicLink != true,
              let text = try? String(contentsOf: folder.appendingPathComponent("report.txt"), encoding: .utf8) else { return nil }
        var report = text
        if !text.contains("Reference30 execution: COMPLETED"),
           let phase = try? String(contentsOf: root.appendingPathComponent("drum-reference30-generation-\(id)-phase.txt"), encoding: .utf8) {
            report += "\nLast persisted native phase for this run:\n" + phase
        }
        func file(_ name: String) -> URL? {
            let url = folder.appendingPathComponent(name)
            guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey,.isSymbolicLinkKey]),
                  values.isRegularFile == true, values.isSymbolicLink != true else { return nil }
            return url
        }
        let hasAudio = text.contains("Reference30 execution: COMPLETED") && text.contains("WAV generated on device: YES")
        let stem = instrument ?? "drums"
        return Reference30GenerationResult(report: report, folder: folder, drumURL: hasAudio ? file(stem + ".wav") : nil,
            mixURL: hasAudio ? file("piano-with-" + stem + ".wav") : nil, pianoURL: hasAudio ? file("piano.wav") : nil)
    }
    private static func writeReference30Floats(_ values: [Float], to url: URL) throws {
        guard values.allSatisfy({ $0.isFinite }) else { throw ProbeError.invalidFloatData }
        try values.withUnsafeBufferPointer { try Data(buffer: $0).write(to: url, options: .atomic) }
    }

    static func generateDrumReference30(instrument: String? = nil, fullSongWindow: FullSongWindowInput? = nil, progress: @escaping @MainActor @Sendable (String) -> Void) async -> Reference30GenerationResult {
        if (fullSongWindow != nil && instrument == nil) || (instrument.map { !["bass","drums","guitar"].contains($0) } ?? false) {
            return Reference30GenerationResult(report:"Invalid instrument; no models opened",folder:nil,drumURL:nil,mixURL:nil,pianoURL:nil)
        }
        let stem = instrument ?? "drums"
        let compareOriginal = fullSongWindow == nil && (instrument == nil || instrument == "drums")
        let clock = ContinuousClock(), start = ContinuousClock().now, runID = UUID().uuidString
        let memory = CoreAIProcessFootprintSampler()
        var samplingPool: [String:RetainedStage] = [:], decodingPool: [String:RetainedStage] = [:]
        var sampling = RetentionStats(), decoding = RetentionStats(), repairs = CacheRepairStats()
        defer { samplingPool.removeAll(); decodingPool.removeAll() }
        var steps = 0, comparisons = 0, passed = 0, completed = false
        var velocityTraceHashes: [String:String] = [:]
        var drumURL: URL?, mixURL: URL?, pianoURL: URL?
        let output = fullSongWindow.map { $0.outputParent.appendingPathComponent("attempt-"+runID,isDirectory:true) } ?? reference30Directory(runID, instrument: instrument)
        func persist(_ text: [String]) {
            if fullSongWindow == nil { saveReference30(text,runID:runID,instrument:instrument) }
            else if let output {
                try? FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
                try? text.joined(separator:"\n").write(to:output.appendingPathComponent("report.txt"),atomically:true,encoding:.utf8)
            }
        }
        var lines = ["Runtime: CoreAI frozen drums30 generation; run \(runID)",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "Revision: reference30-live50-window256-v2; frozen accepted piano/condition/noise; no source/Qwen recomputation",
            "Diagnostic capture: all50 actual DiT velocity outputs before APG; CPU copies only, no feedback substitution",
            "BF16 learned DiT storage, FP32 compute/APG/I/O, FP32 VAE; original abs1e-4/rel1e-3/nRMSE1e-3",
            "50 live Euler/APG updates; guidance7, shift1, momentum-0.75, cap2.5, time projection dim1",
            "Actual step1 serial scopes then max15 retained sampling handles; release before one retained window decoder",
            "Gold states are comparisons only at1/5/10/25/50; no state/momentum reset from gold. Numerical parity covers these checkpoints and final audio, not every unsaved intermediate.",
            "Finite strict failures remain NOT PASSED while independent trajectory measurements continue; invalid/native/nonfinite/memory errors stop",
            "cpuOnly requested; placement/native eviction unmeasured. No broad cache deletion or precision/tolerance changes.",
            "Thermal start: \(ProcessInfo.processInfo.thermalState.rawValue); low power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)"]
        if instrument != nil {
            lines[0] = "Runtime: CoreAI instrument30 preview; instrument \(stem); run \(runID)"
            lines[2] = "Revision: instrument30-shared-sampler-v1; fixed accepted30s piano/context/noise; selected cached condition; no source/Qwen recomputation"
            lines[3] = "No all-step velocity trace in this preview; save five live latent checkpoints and final raw audio. Original drum diagnostic remains separate."
            lines[7] = compareOriginal
                ? "Drum control: compare original five latent checkpoints and final audio; no gold resets"
                : "Bass/guitar matching host trajectory/audio: NOT AVAILABLE; numerical parity NOT EVALUATED; drum gold is never used as their reference"
            lines.append("Selected instrument: \(stem); musical listening and full-song readiness: NOT EVALUATED; original drum4/6 gate remains unchanged")
        }
        if let input = fullSongWindow {
            lines[0] = "Runtime: CoreAI full-song native window; instrument \(stem); run \(runID)"
            lines[2] = "Revision: full-song-window-v1; actual saved global context/noise slices; no gold resets"
            lines[3] = "Window raw output only; native full-song trajectory/music numerical acceptance: NOT EVALUATED"
            lines[7] = "Full-song window numerical parity: NOT EVALUATED; no matching host trajectory/audio; drum gold is never substituted or compared"
            lines.append("Job SHA256: \(input.jobSHA); window index \(input.run.windowIndex); old30s results/pointers untouched; cache repair disabled")
        }
        persist(lines)
        do {
            let expectedProfile = fullSongWindow != nil ? "full-song-prepared-generation" : (instrument == nil ? "drums-reference30-generation" : "instrument-reference30-preview")
            guard (fullSongWindow != nil ? CoreAIPreviewWindow.fullSongEnabled : (instrument == nil ? CoreAIPreviewWindow.drumReference30GenerationEnabled : CoreAIPreviewWindow.instrumentReference30Enabled)),
                  let root = Bundle.main.resourceURL, let output else { throw ShardError.missingPreparedBundle }
            let base = root.appendingPathComponent("ModelCandidates", isDirectory: true)
            guard let stamp = try JSONSerialization.jsonObject(with: Data(contentsOf: base.appendingPathComponent("app-resource-manifest.json"))) as? [String:Any],
                  stamp["selected_profile"] as? String == expectedProfile,
                  let files = stamp["files"] as? [String:[String:Any]],
                  let configSHA = files["drums-reference30-generation.json"]?["sha256"] as? String else { throw ShardError.invalidManifest }
            try verifyReferenceIdentity(base.appendingPathComponent("drums-reference30-generation.json"), expected: configSHA)
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let config = try decoder.decode(Reference30Configuration.self, from: Data(contentsOf: base.appendingPathComponent("drums-reference30-generation.json")))
            try checkManifest(config.schemaVersion, 1, "reference30.schema")
            try checkManifest(config.profile, "drums-reference30-generation", "reference30.profile")
            try checkManifest(config.scope, "frozen_reference_diagnostic", "reference30.scope")
            try checkManifest(config.samplerBundle, "drums-reference30-frozen-diagnostic", "reference30.sampler")
            try checkManifest(config.decoderBundle, "drums-reference30-window-decoder-256", "reference30.decoder")
            try checkManifest(config.samplerReferenceSha256, "f2f84e6c34b2fe42678dc38a48d9ca35b20c14ba53f10843aee8b218a24d53ed", "reference30.original_export")
            try checkManifest(config.captureReportSha256, "f5f84314fa37f49f73502a77f7e3364b31450898231731d64e216bf1b606bfdf", "reference30.capture")
            try checkManifest(config.sourceAudioSha256, "9d62d4eccd36188a4170a08ee6a22fcc66b5ad8af3b0753dc6b021a7f507f4a8", "reference30.piano")
            try checkManifest(config.latentFrames, 750, "reference30.frames")
            try checkManifest(config.conditionTokens, 127, "reference30.tokens")
            try checkManifest(config.inferenceSteps, 50, "reference30.steps")
            try checkManifest(config.checkpoints, [1,5,10,25,50], "reference30.checkpoints")
            try checkManifest(config.absoluteTolerance, 1e-4, "reference30.atol")
            try checkManifest(config.relativeTolerance, 1e-3, "reference30.rtol")
            try checkManifest(config.relativeRmseTolerance, 1e-3, "reference30.nrmse")
            try checkManifest(config.feedback, "live_native_latent_and_momentum_no_gold_resets", "reference30.feedback")
            guard !config.baselineModified, config.referenceListeningGain == 0.36943485303214846,
                  config.pianoMixGain == 0.5, config.drumMixGain == 0.5 else { throw ShardError.invalidManifest }
            let sampler = base.appendingPathComponent(config.samplerBundle, isDirectory: true)
            let codec = base.appendingPathComponent(config.decoderBundle, isDirectory: true)
            try verifyReferenceIdentity(sampler.appendingPathComponent("reference.json"), expected: config.samplerReferenceSha256)
            try verifyReferenceIdentity(codec.appendingPathComponent("reference.json"), expected: config.decoderReferenceSha256)
            let ref = try decoder.decode(Drum30Contract.self, from: Data(contentsOf: sampler.appendingPathComponent("reference.json")))
            let window = try decoder.decode(WindowDecoderContract.self, from: Data(contentsOf: codec.appendingPathComponent("reference.json")))
            try checkManifest(ref.stages.count, 14, "reference30.stages")
            try checkManifest(ref.weightStorageDtype, "bfloat16", "reference30.storage")
            try checkManifest(ref.computeDtype, "float32", "reference30.compute")
            try checkManifest(window.inputShapes, ["latent":[1,64,256]], "reference30.window_input")
            try checkManifest(window.outputShapes, ["audio":[1,2,491520]], "reference30.window_output")
            guard window.windowPlan.count == 6, window.hostNumericalGatePassed, window.architectureGuardPassed,
                  window.coverageExactlyOnce, !window.crossfade, !window.extraLatentPadding,
                  window.weightStorageDtype == "float32", window.computeDtype == "float32",
                  window.vaeCheckpointSha256 == "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0" else { throw ShardError.invalidManifest }
            let starts = [0,64,192,320,448,494]
            for (i,t) in window.windowPlan.enumerated() {
                guard t.windowStart == starts[i], t.windowFrames == 256, t.coreStart == i*128,
                      t.coreEnd == min(i*128+128,750), t.cropStart == t.coreStart-t.windowStart else { throw ShardError.invalidManifest }
            }
            await progress("Memverifikasi sampler, decoder, dan input acuan…")
            if fullSongWindow?.staticAssetsVerified != true {
            for asset in ref.stages.map({ $0.assetFile }) + [ref.apg.assetFile] {
                for file in ["main.hash","main.mlirb","metadata.json"] {
                    let name = asset + "/" + file
                    guard let expected = ref.artifactSha256[name] else { throw ShardError.invalidManifest }
                    try checkManifest(try streamingSHA256(sampler.appendingPathComponent(name)), expected, "reference30.sampler_sha256")
                }
            }
            for (name,expected) in window.artifactSha256 { try checkManifest(try streamingSHA256(codec.appendingPathComponent(name)), expected, "reference30.decoder_sha256") }
            let inputNames = ["initial-noise","condition","source-context","timesteps","source48"] + config.checkpoints.map { String(format:"step-%02d-latent",$0) }
            for name in inputNames {
                let file = "capture/" + name + ".f32le"
                guard let expected = ref.artifactSha256[file] else { throw ShardError.invalidManifest }
                try checkManifest(try streamingSHA256(sampler.appendingPathComponent(file)), expected, "reference30.input_sha256")
            }
            try verifyReferenceIdentity(sampler.appendingPathComponent("capture/report.json"), expected: config.captureReportSha256)
            }
            func input(_ name: String, _ count: Int) throws -> [Float] { try readFloats(folder: sampler, name: "capture/"+name+".f32le", count: count, hashes: ref.artifactSha256) }
            let times = try (fullSongWindow?.times ?? input("timesteps",51))
            guard times.count == 51, times[0] == 1, times[50] == 0, zip(times,times.dropFirst()).allSatisfy({ $0.0 > $0.1 }) else { throw ShardError.invalidManifest }
            var conditionValues = try (fullSongWindow?.condition ?? input("condition",127*2048))
            if let input = fullSongWindow {
                guard input.run.instrument == stem, input.context.count == 750*128, input.noise.count == 48000,
                      input.condition.count == 127*2048, input.times.count == 51,
                      input.context.allSatisfy({ $0.isFinite }), input.noise.allSatisfy({ $0.isFinite }), input.condition.allSatisfy({ $0.isFinite }),
                      FullSongIO.hash(input.context) == input.run.sourceContextSha256,
                      FullSongIO.hash(input.noise) == input.run.initialNoiseSha256,
                      FullSongIO.hash(input.condition) == input.run.conditionSha256 else { throw ShardError.invalidManifest }
            } else if instrument != nil {
                let cacheName = "instrument-reference30-conditioning-127"
                let cache = base.appendingPathComponent(cacheName, isDirectory:true)
                guard let referenceSHA = files[cacheName+"/reference.json"]?["sha256"] as? String else { throw ShardError.invalidManifest }
                try verifyReferenceIdentity(cache.appendingPathComponent("reference.json"), expected:referenceSHA)
                let cached = try decoder.decode(Instrument30Contract.self,from:Data(contentsOf:cache.appendingPathComponent("reference.json")))
                let presets = ["bass":"73177b9d57ef961a66f3df375d24d2a869ad4f27fd6f007b6641d3d58c8614a1",
                    "drums":"bcfd73641fddbda1fb0975febfe28cccb3a3121ec48cec1d877933e64a6d70ca",
                    "guitar":"2f49a2db3ee92b2c52418794d38e4e29312dec5219c3f4ab4c7cf571c12df389"]
                guard cached.schemaVersion == 1, cached.profile == cacheName, cached.status == "prepared_instrument_preview30",
                      cached.roles == ["bass","drums","guitar"], cached.presetSha256 == presets,
                      cached.conditionTokens == 127, cached.conditionShape == [1,127,2048], cached.maskShape == [1,127],
                      cached.captureReportSha256 == config.captureReportSha256, cached.hostDrumsControlExact,
                      cached.absoluteTolerance == 1e-4, cached.relativeTolerance == 1e-3, cached.relativeRmseTolerance == 1e-3,
                      cached.samplerWeightStorageDtype == "bfloat16", cached.computeDtype == "float32", cached.vaeDtype == "float32",
                      cached.nativeExports == 0, !cached.baselineModified, cached.paddingTokens == 0, cached.truncatedTokens == 0,
                      cached.fullSongNativeReadiness == "NOT EVALUATED", cached.bassGuitarNumericalParity == "NOT EVALUATED" else { throw ShardError.invalidManifest }
                let required = Set(["conditioning-report.json"] + cached.roles.flatMap { [$0+"/condition.f32le", $0+"/condition-mask.f32le"] })
                guard Set(cached.artifactSha256.keys) == required else { throw ShardError.invalidManifest }
                for (name,expected) in cached.artifactSha256 {
                    try checkManifest(try streamingSHA256(cache.appendingPathComponent(name)),expected,"instrument30.cache_sha256")
                }
                let selected = try readFloats(folder:cache,name:stem+"/condition.f32le",count:127*2048,hashes:cached.artifactSha256)
                let mask = try readFloats(folder:cache,name:stem+"/condition-mask.f32le",count:127,hashes:cached.artifactSha256)
                guard mask.allSatisfy({ $0 == 1 }) else { throw ShardError.invalidManifest }
                if stem == "drums" { try checkManifest(selected,conditionValues,"instrument30.original_drum_condition") }
                conditionValues = selected
                lines.append("Condition preset SHA256: \(presets[stem] ?? "missing"); condition SHA256: \(cached.artifactSha256[stem+"/condition.f32le"] ?? "missing"); all127 mask values valid")
            }
            let condition = NDArray(scalars:conditionValues, shape:[1,127,2048])
            let context = NDArray(scalars: try (fullSongWindow?.context ?? input("source-context",750*128)), shape:[1,750,128])
            var latent = NDArray(scalars: try (fullSongWindow?.noise ?? input("initial-noise",48000)), shape:[1,750,64])
            var momentum = NDArray(scalars:[Float](repeating:0,count:48000), shape:[1,750,64])
            let update = Contract.Stage(assetFile:ref.apg.assetFile, layerIndices:[], inputShapes:ref.apg.inputShapes, outputShapes:ref.apg.outputShapes, outputBindings:nil)
            lines.append(fullSongWindow == nil
                ? "SHA256: all15 sampler assets, bounded decoder, exact inputs and five gold checkpoints VERIFIED; original full750 decoder never opened"
                : "SHA256: shared sampler/decoder verified once per invocation; actual global window context/noise/condition verified before use; no gold comparisons; original full750 decoder never opened")
            persist(lines)
            for i in 0..<50 {
                let stepStart = clock.now, before = sampling.phaseTotals.backendAwaitMs
                let reuse = i > 0
                if i == 1 { try checkManifest(samplingPool.count,0,"reference30.cold_step_handles"); lines.append("Actual step1 completed with zero retained handles; retaining from step2") }
                await progress("Membuat \(stem) 30 detik: langkah \(i+1)/50…")
                var state = ["noise":latent,"condition":condition,"context":context,"timestep":NDArray(scalars:[times[i]],shape:[1])]
                for (j,stage) in ref.stages.enumerated() {
                    try requirePreviewMemoryBudget()
                    let phase = "Run \(runID); step\(i+1)/50 stage\(j+1)/14 \(stage.assetFile); handles\(samplingPool.count)"
                    let breadcrumbs = interleavingPhaseProgress(prefix:phase,fileName:"drum-reference30-generation-\(runID)-phase.txt")
                    breadcrumbs("Before native sampler call")
                    var inputs: [String:NDArray] = [:]
                    for (name,shape) in stage.inputShapes { guard let value = state[name], value.shape == shape else { throw ShardError.invalidManifest }; inputs[name] = value }
                    let result = try await runPreviewSamplingStage(stage,folder:sampler,inputs:inputs,aot:nil,reuseStages:reuse,
                        pool:&samplingPool,stats:&sampling,allowCacheRepair:fullSongWindow == nil,cacheRepair:&repairs,phaseProgress:breadcrumbs)
                    for (name,value) in result.values { guard let binding = stage.outputBindings?[name] else { throw ShardError.invalidManifest };state[binding] = value }
                    guard samplingPool.count <= 15 else { throw ShardError.invalidManifest }
                }
                guard let velocity = state["velocity"] else { throw ProbeError.invalidOutputTensor }
                if instrument == nil {
                    let traceName = String(format:"native-step-%02d-velocity.f32le",i+1)
                    let traceURL = output.appendingPathComponent(traceName)
                    try writeReference30Floats(try copiedValues(velocity,shape:[2,750,64]),to:traceURL)
                    velocityTraceHashes[traceName] = try streamingSHA256(traceURL)
                }
                let breadcrumbs = interleavingPhaseProgress(prefix:"Run \(runID); step\(i+1)/50 APG live feedback",fileName:"drum-reference30-generation-\(runID)-phase.txt")
                let next = try await runPreviewSamplingStage(update,folder:sampler,
                    inputs:["velocity":velocity,"latent":latent,"momentum":momentum,"dt":NDArray(scalars:[times[i]-times[i+1]],shape:[1])],
                    aot:nil,reuseStages:reuse,pool:&samplingPool,stats:&sampling,allowCacheRepair:fullSongWindow == nil,cacheRepair:&repairs,phaseProgress:breadcrumbs)
                guard let x = next.values["latent_next"], let m = next.values["momentum_next"] else { throw ProbeError.invalidOutputTensor }
                latent = x; momentum = m; steps = i+1
                lines.append("Step\(steps)/50: total ms \(ms(stepStart.duration(to:clock.now))); backend delta ms \(sampling.phaseTotals.backendAwaitMs-before)")
                if config.checkpoints.contains(steps) && fullSongWindow == nil {
                    let actual = try copiedValues(latent,shape:[1,750,64])
                    try writeReference30Floats(actual,to:output.appendingPathComponent(String(format:"native-step-%02d-latent.f32le",steps)))
                    if compareOriginal {
                        try drum30Comparison("Live latent step\(steps)",actual:actual,expected:try input(String(format:"step-%02d-latent",steps),48000),
                            comparisons:&comparisons,passed:&passed,lines:&lines)
                    }
                }
                if config.checkpoints.contains(steps) || steps % 5 == 0 { persist(lines) }
            }
            try checkManifest(sampling.completedCalls,750,"reference30.sampling_calls")
            try checkManifest(sampling.modelOpens,30+repairs.attempts,"reference30.sampling_opens")
            try checkManifest(sampling.functionBinds,30,"reference30.sampling_binds")
            try checkManifest(sampling.cacheHits,720,"reference30.sampling_hits")
            if instrument == nil {
            try checkManifest(velocityTraceHashes.count,50,"reference30.velocity_trace_count")
            let traceURL = output.appendingPathComponent("velocity-trace.json")
            let traceMetadata: [String:Any] = [
                "schema_version":1,"profile":"native-frozen30-velocity-trace","run_id":runID,
                "steps":50,"shape":[2,750,64],"dtype":"float32_le",
                "semantics":"Actual conditional/unconditional DiT outputs before APG; diagnostic copies, no feedback reset",
                "artifact_sha256":velocityTraceHashes]
            let traceData = try JSONSerialization.data(withJSONObject:traceMetadata,options:[.sortedKeys])
            try traceData.write(to:traceURL,options:.atomic)
            lines.append("Velocity trace SHA256: \(try streamingSHA256(traceURL)); tensors50/50; no feedback substitution")
            }
            samplingPool.removeAll()
            lines.append("Sampling handles released before window decode; native eviction unverified")
            persist(lines)
            let final = try copiedValues(latent,shape:[1,750,64])
            try writeReference30Floats(final,to:output.appendingPathComponent("final-latent.f32le"))
            try writeReference30Floats(try copiedValues(momentum,shape:[1,750,64]),to:output.appendingPathComponent("final-momentum.f32le"))
            lines.append("LIVE final latent SHA256: \(try streamingSHA256(output.appendingPathComponent("final-latent.f32le")))")
            let codecStage = Contract.Stage(assetFile:window.assetFile,layerIndices:[],inputShapes:window.inputShapes,outputShapes:window.outputShapes,outputBindings:nil)
            var audio = [Float](repeating:0,count:2880000)
            for (i,t) in window.windowPlan.enumerated() {
                try requirePreviewMemoryBudget()
                await progress("Mendekode \(stem) hasil 50 langkah: \(i+1)/6…")
                var data = [Float](repeating:0,count:16384)
                for c in 0..<64 { for f in 0..<256 { data[c*256+f] = final[(t.windowStart+f)*64+c] } }
                let hash = data.withUnsafeBufferPointer { SHA256.hash(data:Data(buffer:$0)).map { String(format:"%02x",$0) }.joined() }
                lines.append("Decode window\(i+1): LIVE final-latent slice SHA256 \(hash); gold tile input is not substituted")
                persist(lines)
                let breadcrumbs = interleavingPhaseProgress(prefix:"Run \(runID); live decoder window\(i+1)/6",fileName:"drum-reference30-generation-\(runID)-phase.txt")
                let result = try await runPreviewSamplingStage(codecStage,folder:codec,inputs:["latent":NDArray(scalars:data,shape:[1,64,256])],
                    aot:nil,reuseStages:true,pool:&decodingPool,stats:&decoding,allowCacheRepair:false,cacheRepair:&repairs,phaseProgress:breadcrumbs)
                guard let tensor = result.values["audio"] else { throw ProbeError.invalidOutputTensor }
                let values = try copiedValues(tensor,shape:[1,2,491520])
                for c in 0..<2 { for j in 0..<((t.coreEnd-t.coreStart)*1920) { audio[c*1440000+t.coreStart*1920+j] = values[c*491520+t.cropStart*1920+j] } }
                try requirePreviewMemoryBudget()
                lines.append("Decode window\(i+1) backend-await ms \(result.phases.backendAwaitMs)")
            }
            decodingPool.removeAll()
            try checkManifest(decoding.completedCalls,6,"reference30.decode_calls")
            try checkManifest(decoding.modelOpens,1,"reference30.decode_opens")
            try checkManifest(decoding.functionBinds,1,"reference30.decode_binds")
            try checkManifest(decoding.cacheHits,5,"reference30.decode_hits")
            if compareOriginal {
                let expected = try readFloats(folder:codec,name:"final-audio.f32le",count:2880000,hashes:window.artifactSha256)
                try drum30Comparison("LIVE full30s decoded audio",actual:audio,expected:expected,comparisons:&comparisons,passed:&passed,lines:&lines)
                try checkManifest(comparisons,6,"reference30.checkpoint_audio_comparisons")
            } else { try checkManifest(comparisons,0,"instrument30.no_unrelated_gold_comparisons") }
            try writeReference30Floats(audio,to:output.appendingPathComponent("raw-"+stem+".f32le"))
            if fullSongWindow != nil {
                completed = true
                lines.append("Full-song window raw completion: YES; no per-window listening gain or crossfade")
            } else {
            let piano = try input("source48",2880000)
            var mix = zip(piano,audio).map { Float(config.pianoMixGain)*$0.0+Float(config.drumMixGain)*$0.1 }
            let drumPeak = audio.reduce(Float(0)){max($0,abs($1))}
            let pianoPeak = piano.reduce(Float(0)){max($0,abs($1))}
            let mixPeak = mix.reduce(Float(0)){max($0,abs($1))}
            let peak = max(drumPeak,max(pianoPeak,mixPeak))
            let gain = min(Float(config.referenceListeningGain),Float(pow(10.0,-3.0/20.0))/max(peak,1e-12))
            let drum = output.appendingPathComponent(stem+".wav"), mixed = output.appendingPathComponent("piano-with-"+stem+".wav"), source = output.appendingPathComponent("piano.wav")
            try writeStereoFloatWAV(audio.map{$0*gain},frames:1440000,to:drum)
            try writeStereoFloatWAV(piano.map{$0*gain},frames:1440000,to:source)
            for i in mix.indices { mix[i] *= gain }; try writeStereoFloatWAV(mix,frames:1440000,to:mixed)
            drumURL = drum; mixURL = mixed; pianoURL = source; completed = true
            lines.append("Listening copies: shared gain\(gain), diagnostic0.5piano+0.5\(stem); no second source gain; production mix unchanged. Raw comparisons precede attenuation.")
            lines.append("Artifact SHA256: \(stem) \(try streamingSHA256(drum)); mix \(try streamingSHA256(mixed)); piano \(try streamingSHA256(source))")
            }
        } catch {
            samplingPool.removeAll(); decodingPool.removeAll()
            if let failure = error as? StageExecutionFailure { lines.append("Failing asset: \(failure.asset); operation: \(failure.operation)"); lines.append(contentsOf:failure.details) }
            else { lines.append(contentsOf:modelErrorDetails(error)) }
        }
        let usage = memory.finish(), numerical = compareOriginal && completed && passed == 6
        let memoryPassed = usage.peak.map{$0 < 2000*1048576} ?? false
        lines.append(contentsOf:repairs.events)
        lines.append("Reference30 execution: \(completed ? "COMPLETED" : "NOT COMPLETED"); steps\(steps)/50; sampling calls\(sampling.completedCalls)/750; decoder calls\(decoding.completedCalls)/6")
        lines.append("Sampling opens\(sampling.modelOpens), binds\(sampling.functionBinds), hits\(sampling.cacheHits); decoder opens\(decoding.modelOpens), binds\(decoding.functionBinds), hits\(decoding.cacheHits)")
        if compareOriginal {
            lines.append("Original checkpoint/audio strict parity: \(completed ? (numerical ? "PASS" : "NOT PASSED") : "NOT EVALUATED"); \(passed)/\(comparisons) comparisons passed; target6. Sparse checkpoints are not every intermediate latent.")
        } else {
            lines.append("\(stem) numerical parity: NOT EVALUATED; no matching host trajectory/audio; no comparisons against drum gold, target0")
        }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10ms): \(Double(peak)/1048576); sampled budget <2000MiB: \(memoryPassed ? "PASS" : "NOT PASSED")") }
        else { lines.append("Sampled memory readiness: NOT EVALUATED; footprint unavailable") }
        lines.append("Reference30 numerical/resource readiness: \(compareOriginal ? (numerical && memoryPassed ? "PASS" : "NOT PASSED") : "NOT EVALUATED"); native music listening/full-song bass-drums-guitar: NOT EVALUATED")
        if fullSongWindow == nil { lines.append("WAV generated on device: \(completed ? "YES; diagnostic listening copies" : "NO"); targeted initial-bind repair\(repairs.attempts)/1, successful retries\(repairs.successfulRetries); no inference retry or baseline replacement") }
        else { lines.append("Raw window only; cache repair/retries0; original baseline preserved") }
        lines.append("Thermal end: \(ProcessInfo.processInfo.thermalState.rawValue); total reference30 ms \(ms(start.duration(to:clock.now)))")
        persist(lines)
        return Reference30GenerationResult(report:lines.joined(separator:"\n"),folder:output,drumURL:drumURL,mixURL:mixURL,pianoURL:pianoURL)
    }
}

extension CoreAIDecoderProbe {
    private struct SourceWindowContract: Decodable {
        let schemaVersion: Int; let profile: String; let status: String; let assetFile: String; let functionName: String
        let latentFrames: Int; let windowFrames: Int; let coreFrames: Int; let haloFrames: Int
        let inputShapes: [String:[Int]]; let outputShapes: [String:[Int]]
        let vaeCheckpointSha256: String; let captureReportSha256: String; let originalSourceReferenceSha256: String
        let weightStorageDtype: String; let computeDtype: String; let scalarType: String; let byteOrder: String
        let absoluteTolerance: Double; let relativeTolerance: Double; let relativeRmseTolerance: Double
        let windowPlan: [WindowDecoderContract.Tile]; let dependencySampleBounds: [[Int]]
        let tileInputSha256: [[String:String]]; let hostChecks: [String:WindowDecoderContract.Metric]
        let hostNumericalGatePassed: Bool; let coverageExactlyOnce: Bool; let architectureGuardPassed: Bool
        let epsilonRegenerated: Bool; let nativeExports: Int; let baselineModified: Bool
        let nativeParity: String; let nativeMemoryGate: String; let fullSongNativeReadiness: String
        let artifactSha256: [String:String]
    }

    static func savedSourceWindowReport() -> String? {
        guard let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first,
              let text = try? String(contentsOf: root.appendingPathComponent("last-source-window-report.txt"), encoding: .utf8) else { return nil }
        guard !text.contains("Source window collection: COMPLETED"),
              let first = text.components(separatedBy:"\n").first,
              let id = first.components(separatedBy:"; run ").last, UUID(uuidString:id) != nil,
              let phase = try? String(contentsOf:root.appendingPathComponent("source-window-\(id)-phase.txt"),encoding:.utf8) else { return text }
        return text + "\nLast persisted native phase for this run:\n" + phase
    }

    private static func saveSourceWindow(_ lines: [String], runID: String) {
        guard let root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask).first else { return }
        let folder = root.appendingPathComponent("SourceWindowReports/"+runID,isDirectory:true)
        try? FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let text = lines.joined(separator:"\n")
        try? text.write(to:folder.appendingPathComponent("report.txt"),atomically:true,encoding:.utf8)
        try? text.write(to:root.appendingPathComponent("last-source-window-report.txt"),atomically:true,encoding:.utf8)
    }

    static func runSourceWindowEncoder(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
        let clock = ContinuousClock(), start = ContinuousClock().now, runID = UUID().uuidString
        let memory = CoreAIProcessFootprintSampler()
        var pool: [String:RetainedStage] = [:], stats = RetentionStats(), repairs = CacheRepairStats()
        defer { pool.removeAll() }
        var comparisons = 0, passed = 0, completed = false
        var lines = ["Runtime: CoreAI source window encoder diagnostic; run \(runID)",
            "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "Revision: source-window256-core224-v1; four fixed-input encoder calls; one retained model/function",
            "FP32 source VAE, original750-frame audio/epsilon/mask; no regenerated RNG, source gain, DiT/APG/Qwen/decoder or WAV",
            "Compare four retained cores, three seam regions and full assembled mean/std/source/context: target32 original gates",
            "Original abs1e-4/rel1e-3/nRMSE1e-3; no retry/cache deletion; baseline unchanged",
            "cpuOnly requested; actual placement/native eviction unmeasured; sampled process budget <2000MiB is not an exact transient peak cap",
            "Thermal start: \(ProcessInfo.processInfo.thermalState.rawValue); low power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)"]
        saveSourceWindow(lines,runID:runID)
        do {
            guard CoreAIPreviewWindow.sourceWindowEncoderEnabled, let root = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let base = root.appendingPathComponent("ModelCandidates",isDirectory:true)
            let relative = "source-reference30-window-encoder-256/reference.json"
            guard let stamp = try JSONSerialization.jsonObject(with:Data(contentsOf:base.appendingPathComponent("app-resource-manifest.json"))) as? [String:Any],
                  stamp["selected_profile"] as? String == "source-reference30-window-encoder",
                  let files = stamp["files"] as? [String:[String:Any]],let identity = files[relative]?["sha256"] as? String else { throw ShardError.invalidManifest }
            try verifyReferenceIdentity(base.appendingPathComponent(relative),expected:identity)
            let folder = base.appendingPathComponent("source-reference30-window-encoder-256",isDirectory:true)
            let decoder = JSONDecoder();decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(SourceWindowContract.self,from:Data(contentsOf:folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion,1,"source.schema")
            try checkManifest(ref.profile,"source-reference30-window-encoder","source.profile")
            try checkManifest(ref.status,"prepared_source_window_encoder","source.status")
            try checkManifest(ref.assetFile,"source-context.aimodel","source.asset")
            try checkManifest(ref.functionName,"main","source.function")
            try checkManifest(ref.latentFrames,750,"source.frames")
            try checkManifest(ref.windowFrames,256,"source.window")
            try checkManifest(ref.coreFrames,224,"source.core")
            try checkManifest(ref.haloFrames,16,"source.halo")
            try checkManifest(ref.weightStorageDtype,"float32","source.storage")
            try checkManifest(ref.computeDtype,"float32","source.compute")
            try checkManifest(ref.scalarType,"float32","source.scalar")
            try checkManifest(ref.byteOrder,"little","source.byte_order")
            try checkManifest(ref.vaeCheckpointSha256,"da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0","source.VAE")
            try checkManifest(ref.captureReportSha256,"f5f84314fa37f49f73502a77f7e3364b31450898231731d64e216bf1b606bfdf","source.capture")
            try checkManifest(ref.absoluteTolerance,1e-4,"source.atol")
            try checkManifest(ref.relativeTolerance,1e-3,"source.rtol")
            try checkManifest(ref.relativeRmseTolerance,1e-3,"source.nrmse")
            guard ref.hostNumericalGatePassed,ref.coverageExactlyOnce,ref.architectureGuardPassed,
                  !ref.epsilonRegenerated,!ref.baselineModified,ref.nativeExports == 0,
                  ref.nativeParity == "NOT EVALUATED",ref.nativeMemoryGate == "NOT EVALUATED",
                  ref.fullSongNativeReadiness == "NOT EVALUATED",ref.windowPlan.count == 4,ref.tileInputSha256.count == 4 else { throw ShardError.invalidManifest }
            let inputs: [String:[Int]] = ["audio":[1,2,491520],"epsilon":[1,64,256],"chunk_mask":[1,256,64]]
            let shapes: [String:[Int]] = ["posterior_mean":[1,64,256],"posterior_std":[1,64,256],"source_latents":[1,256,64],"context_latents":[1,256,128]]
            try checkManifest(ref.inputShapes,inputs,"source.inputs")
            try checkManifest(ref.outputShapes,shapes,"source.outputs")
            let goldFiles = ["posterior_mean":"source-mean.f32le","posterior_std":"source-std.f32le",
                             "source_latents":"source-latent.f32le","context_latents":"source-context.f32le"]
            var requiredChecks = Set<String>()
            for name in goldFiles.keys {
                for i in 0..<4 { requiredChecks.insert("core-\(i)-"+name) }
                for i in 1..<4 { requiredChecks.insert("seam-\(i)-"+name) }
                requiredChecks.insert("overall-"+name)
            }
            try checkManifest(Set(ref.hostChecks.keys),requiredChecks,"source.host_checks")
            for m in ref.hostChecks.values {
                guard m.finite,m.parityPassed,m.elementwiseFractionOverBound == 0,m.normalizedRmse <= 1e-3,
                      [m.maxAbsoluteError,m.rmse,m.normalizedRmse].allSatisfy({$0.isFinite && $0 >= 0}) else { throw ShardError.invalidManifest }
            }
            try checkManifest(ref.dependencySampleBounds,[[0,442244],[417915,872324],[847995,1302404],[1278075,1439999]],"source.dependencies")
            let starts = [0,208,432,494]
            for (i,tile) in ref.windowPlan.enumerated() {
                guard tile.windowStart == starts[i],tile.windowFrames == 256,tile.coreStart == i*224,
                      tile.coreEnd == min(i*224+224,750),tile.cropStart == tile.coreStart-tile.windowStart else { throw ShardError.invalidManifest }
            }
            var names = Set(goldFiles.values)
            names.formUnion(["source48.f32le","source-epsilon.f32le","source-mask.f32le","capture-report.json","original-source-reference.json"])
            for name in ["main.hash","main.mlirb","metadata.json"] { names.insert("source-context.aimodel/"+name) }
            try checkManifest(Set(ref.artifactSha256.keys),names,"source.files")
            try checkManifest(ref.artifactSha256["capture-report.json"],ref.captureReportSha256,"source.report_identity")
            try checkManifest(ref.artifactSha256["original-source-reference.json"],ref.originalSourceReferenceSha256,"source.original_identity")
            await progress("Memverifikasi model dan input piano…")
            for (name,hash) in ref.artifactSha256 { try checkManifest(try streamingSHA256(folder.appendingPathComponent(name)),hash,"source.sha256") }
            guard let original = try JSONSerialization.jsonObject(with:Data(contentsOf:folder.appendingPathComponent("original-source-reference.json"))) as? [String:Any],
                  original["checkpoint_digest"] as? String == ref.vaeCheckpointSha256,
                  let originalHashes = original["artifact_sha256"] as? [String:String] else { throw ShardError.invalidManifest }
            for name in ["main.hash","main.mlirb","metadata.json"] {
                let key = "source-context.aimodel/"+name
                try checkManifest(originalHashes[key],ref.artifactSha256[key],"source.reused_payload")
            }
            lines.append("Manifest, unchanged encoder payload and exact original audio/epsilon/mask/references: SHA256 VERIFIED")
            func read(_ name: String,_ count: Int) throws -> [Float] { try readFloats(folder:folder,name:name,count:count,hashes:ref.artifactSha256) }
            let audio = try read("source48.f32le",2880000),epsilon = try read("source-epsilon.f32le",48000),mask = try read("source-mask.f32le",48000)
            guard mask.allSatisfy({$0 == 1}) else { throw ShardError.invalidManifest }
            var gold: [String:[Float]] = [:],assembled: [String:[Float]] = [:]
            for (name,file) in goldFiles {
                let count = name == "context_latents" ? 96000 : 48000
                gold[name] = try read(file,count);assembled[name] = [Float](repeating:0,count:count)
            }
            var coverage = [Int](repeating:0,count:750)
            let stage = Contract.Stage(assetFile:ref.assetFile,layerIndices:[],inputShapes:ref.inputShapes,outputShapes:ref.outputShapes,outputBindings:nil)
            func retained(_ values: [Float],_ name: String,_ a: Int,_ b: Int,_ frames: Int) -> [Float] {
                if name.hasPrefix("posterior") {
                    return (0..<64).flatMap { channel in Array(values[channel*frames+a..<channel*frames+b]) }
                }
                let channels = name == "context_latents" ? 128 : 64
                return Array(values[a*channels..<b*channels])
            }
            for (i,tile) in ref.windowPlan.enumerated() {
                try requirePreviewMemoryBudget()
                let start = tile.windowStart,a = tile.coreStart,b = tile.coreEnd,c = tile.cropStart
                let sourceAudio = (0..<2).flatMap { channel in Array(audio[channel*1440000+start*1920..<channel*1440000+(start+256)*1920]) }
                let sourceEpsilon = (0..<64).flatMap { channel in Array(epsilon[channel*750+start..<channel*750+start+256]) }
                let sourceMask = Array(mask[start*64..<(start+256)*64])
                let scalarInputs = ["audio":sourceAudio,"epsilon":sourceEpsilon,"chunk_mask":sourceMask]
                var tensors: [String:NDArray] = [:]
                for (name,values) in scalarInputs {
                    let hash = values.withUnsafeBufferPointer { SHA256.hash(data:Data(buffer:$0)).map{String(format:"%02x",$0)}.joined() }
                    try checkManifest(hash,ref.tileInputSha256[i][name],"source.tile_input")
                    guard let shape = ref.inputShapes[name] else { throw ShardError.invalidManifest }
                    tensors[name] = NDArray(scalars:values,shape:shape)
                }
                await progress("Memeriksa input piano: bagian \(i+1)/4…")
                saveSourceWindow(lines,runID:runID)
                let phase = interleavingPhaseProgress(prefix:"Run \(runID); source window\(i+1)/4",fileName:"source-window-\(runID)-phase.txt")
                let result = try await runPreviewSamplingStage(stage,folder:folder,inputs:tensors,aot:nil,reuseStages:true,
                    pool:&pool,stats:&stats,allowCacheRepair:false,cacheRepair:&repairs,phaseProgress:phase)
                for (name,shape) in ref.outputShapes {
                    guard let tensor = result.values[name],let expected = gold[name],var full = assembled[name] else { throw ProbeError.invalidOutputTensor }
                    let values = try copiedValues(tensor,shape:shape),kept = retained(values,name,c,c+b-a,256)
                    try drum30Comparison("Source core\(i+1) \(name)",actual:kept,expected:retained(expected,name,a,b,750),comparisons:&comparisons,passed:&passed,lines:&lines)
                    if name.hasPrefix("posterior") {
                        for channel in 0..<64 { for f in 0..<(b-a) { full[channel*750+a+f] = kept[channel*(b-a)+f] } }
                    } else {
                        let channels = name == "context_latents" ? 128 : 64
                        full.replaceSubrange(a*channels..<b*channels,with:kept)
                    }
                    assembled[name] = full
                }
                for f in a..<b { coverage[f] += 1 }
                lines.append("Source window\(i+1) backend-await ms \(result.phases.backendAwaitMs)")
                saveSourceWindow(lines,runID:runID)
            }
            guard coverage.allSatisfy({$0 == 1}) else { throw ShardError.invalidManifest }
            for (i,tile) in ref.windowPlan.dropFirst().enumerated() {
                let a = tile.coreStart-2,b = min(tile.coreStart+2,750)
                for name in goldFiles.keys {
                    guard let actual = assembled[name],let expected = gold[name] else { throw ShardError.invalidManifest }
                    try drum30Comparison("Source seam\(i+1) \(name)",actual:retained(actual,name,a,b,750),expected:retained(expected,name,a,b,750),comparisons:&comparisons,passed:&passed,lines:&lines)
                }
            }
            for name in goldFiles.keys {
                guard let actual = assembled[name],let expected = gold[name] else { throw ShardError.invalidManifest }
                try drum30Comparison("Source full750 \(name)",actual:actual,expected:expected,comparisons:&comparisons,passed:&passed,lines:&lines)
            }
            guard let documents = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask).first else { throw ShardError.invalidManifest }
            let captured = documents.appendingPathComponent("SourceWindowReports/"+runID,isDirectory:true)
            try FileManager.default.createDirectory(at:captured,withIntermediateDirectories:true)
            for (name,values) in assembled {
                let target = captured.appendingPathComponent("native-"+name+".f32le")
                try writeReference30Floats(values,to:target)
                lines.append("Source artifact \(target.lastPathComponent) SHA256: \(try streamingSHA256(target))")
            }
            try checkManifest(stats.completedCalls,4,"source.calls")
            try checkManifest(stats.modelOpens,1,"source.opens")
            try checkManifest(stats.functionBinds,1,"source.binds")
            try checkManifest(stats.cacheHits,3,"source.hits")
            try checkManifest(comparisons,32,"source.comparisons")
            pool.removeAll();completed = true
        } catch {
            if let failure = error as? StageExecutionFailure { lines.append("Failing asset: \(failure.asset); operation: \(failure.operation)");lines.append(contentsOf:failure.details) }
            else { lines.append(contentsOf:modelErrorDetails(error)) }
        }
        let usage = memory.finish(),memoryPassed = usage.peak.map{$0 < 2000*1048576} ?? false
        lines.append("Source window collection: \(completed ? "COMPLETED" : "NOT COMPLETED"); calls\(stats.completedCalls)/4, opens\(stats.modelOpens), binds\(stats.functionBinds), hits\(stats.cacheHits)")
        lines.append("Source original strict parity: \(completed && passed == 32 ? "PASS" : "NOT PASSED"); \(passed)/\(comparisons) comparisons passed; target32")
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10ms): \(Double(peak)/1048576); sampled budget <2000MiB: \(memoryPassed ? "PASS" : "NOT PASSED")") }
        else { lines.append("Sampled memory readiness: NOT EVALUATED; footprint unavailable") }
        lines.append("Source component numerical/resource readiness: \(completed && passed == 32 && memoryPassed ? "PASS" : "NOT PASSED"); full-song source/generation/music: NOT EVALUATED")
        lines.append("Cache deletion/retries: DISABLED; no DiT/APG/Qwen/decoder opened; BF16 sampler baseline and original bounds unchanged")
        lines.append("Thermal end: \(ProcessInfo.processInfo.thermalState.rawValue); total source window ms \(ms(start.duration(to:clock.now)))")
        saveSourceWindow(lines,runID:runID)
        return lines.joined(separator:"\n")
    }
}

extension CoreAIDecoderProbe {
    struct FullSongResult: Sendable {
        let report: String; let folder: URL?; let audio: [String:URL]
    }
    private struct FullSongCheckpoint: Codable {
        let schemaVersion: Int; let executionSha256: String; let jobReferenceSha256: String
        let instrument: String; let windowIndex: Int; let childDirectory: String
        let rawSha256: String; let reportSha256: String
    }
    private struct FullSongStemCheckpoint: Codable {
        let schemaVersion: Int; let executionSha256: String; let instrument: String
        let targetSamples48: Int; let rawSha256: String
    }
    private struct FullSongCompletion: Codable {
        let schemaVersion: Int; let executionSha256: String; let jobReferenceSha256: String
        let targetSamples48: Int; let completedWindows: Int; let listeningGain: Float
        let artifactSha256: [String:String]
    }
    private static func fullSongLocation() throws -> (URL,URL,String,FullSongExecution,URL) {
        guard CoreAIPreviewWindow.fullSongEnabled, let root = Bundle.main.resourceURL,
              let documents = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask).first else { throw ShardError.missingPreparedBundle }
        let base = root.appendingPathComponent("ModelCandidates",isDirectory:true)
        let config = base.appendingPathComponent("full-song-generation.json")
        let sha = try streamingSHA256(config)
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        let execution = try decoder.decode(FullSongExecution.self,from:Data(contentsOf:config))
        guard FullSongIO.isSHA(execution.jobReferenceSha256), FullSongIO.isSHA(execution.decoderReferenceSha256),
              execution.schemaVersion == 1, execution.profile == "full-song-prepared-generation",
              execution.scope == "prepared-input-full-song-diagnostic", execution.nativeImplementation == "prepared-full-song-v1",
              execution.jobBundle == "full-song-inputs-"+String(execution.jobReferenceSha256.prefix(16)),
              execution.samplerBundle == "drums-reference30-frozen-diagnostic", execution.decoderBundle == "drums-reference30-window-decoder-256",
              execution.samplerReferenceSha256 == "f2f84e6c34b2fe42678dc38a48d9ca35b20c14ba53f10843aee8b218a24d53ed",
              execution.fullSongStrictParity == "NOT EVALUATED", !execution.baselineModified,
              execution.sourceFrontend == "host-prepared context; native source NOT EVALUATED" else { throw ShardError.invalidManifest }
        return (base,base.appendingPathComponent(execution.jobBundle,isDirectory:true),sha,execution,
                documents.appendingPathComponent("FullSongGeneration/"+sha,isDirectory:true))
    }
    static func latestFullSong() -> FullSongResult? {
        do {
            let (_,input,sha,execution,folder) = try fullSongLocation()
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let job = try decoder.decode(FullSongJob.self,from:Data(contentsOf:input.appendingPathComponent("reference.json")))
            try job.validate()
            let report = (try? String(contentsOf:folder.appendingPathComponent("report.txt"),encoding:.utf8)) ?? "Lagu penuh belum dijalankan."
            var audio: [String:URL] = [:]
            if let data = try? Data(contentsOf:folder.appendingPathComponent("completed.json")),
               let complete = try? decoder.decode(FullSongCompletion.self,from:data), complete.executionSha256 == sha,
               complete.schemaVersion == 1, complete.jobReferenceSha256 == execution.jobReferenceSha256,
               complete.targetSamples48 == job.targetSamples48, complete.completedWindows == job.runs.count,
               complete.artifactSha256.values.allSatisfy(FullSongIO.isSHA),
               Set(complete.artifactSha256.keys) == Set(["piano.wav","bass.wav","drums.wav","guitar.wav","piano-with-band.wav"]) {
                for (name,hash) in complete.artifactSha256 {
                    let file = folder.appendingPathComponent(name)
                    try FullSongIO.regular(file,bytes:58+complete.targetSamples48*8)
                    guard FullSongIO.isSHA(hash) else { throw FullSongFailure.invalid("Invalid saved WAV identity") }
                    audio[name] = file
                }
            }
            return FullSongResult(report:report,folder:folder,audio:audio)
        } catch { return nil }
    }

    static func generateFullSong(progress: @escaping @MainActor @Sendable (String) -> Void,
                                 shouldPause: @escaping @MainActor @Sendable () -> Bool) async -> FullSongResult {
        let clock = ContinuousClock(), began = ContinuousClock().now
        let memory = CoreAIProcessFootprintSampler()
        var lines = ["Runtime: CoreAI full-song prepared-input diagnostic", "Revision: prepared-full-song-v1",
            "Host source context/conditioning; native BF16 DiT, FP32 APG/decoder,50 steps/window, original bounds unchanged",
            "Completed window SHA256 checkpoints can resume; no automatic inference retry/cache repair",
            "Native source encoder/full-trajectory numerical/music acceptance: NOT EVALUATED; original drum4/6 retained",
            "cpuOnly requested; placement/native eviction unmeasured; sampled process budget <2000MiB, not exact transient/native peak",
            "Thermal start: \(ProcessInfo.processInfo.thermalState.rawValue); low power: \(ProcessInfo.processInfo.isLowPowerModeEnabled)"]
        var folder: URL?, completed = false, paused = false, newWindows = 0, reusedWindows = 0
        func save() { if let folder { try? lines.joined(separator:"\n").write(to:folder.appendingPathComponent("report.txt"),atomically:true,encoding:.utf8) } }
        do {
            let (base,input,executionSHA,execution,output) = try fullSongLocation()
            folder = output
            try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
            try FullSongIO.preserveOrphan(output.appendingPathComponent("report.txt"))
            save()
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let encoder = JSONEncoder(); encoder.keyEncodingStrategy = .convertToSnakeCase; encoder.outputFormatting = [.sortedKeys]
            guard let stamp = try JSONSerialization.jsonObject(with:Data(contentsOf:base.appendingPathComponent("app-resource-manifest.json"))) as? [String:Any],
                  stamp["selected_profile"] as? String == "full-song-prepared-generation",
                  let files = stamp["files"] as? [String:[String:Any]],
                  files["full-song-generation.json"]?["sha256"] as? String == executionSHA else { throw ShardError.invalidManifest }
            try verifyReferenceIdentity(input.appendingPathComponent("reference.json"),expected:execution.jobReferenceSha256)
            let job = try decoder.decode(FullSongJob.self,from:Data(contentsOf:input.appendingPathComponent("reference.json")))
            try job.validate()
            for (name,expected) in job.artifactSha256 {
                let file = input.appendingPathComponent(name)
                try FullSongIO.regular(file,bytes:job.tensorShapes[name].map { $0.reduce(1,*)*4 })
                try checkManifest(try streamingSHA256(file),expected,"full_song.input_sha256")
            }
            let original = try decoder.decode(Drum30Contract.self,from:Data(contentsOf:base.appendingPathComponent(execution.samplerBundle+"/reference.json")))
            try verifyReferenceIdentity(base.appendingPathComponent(execution.samplerBundle+"/reference.json"),expected:execution.samplerReferenceSha256)
            try checkManifest(job.artifactSha256["timesteps.f32le"],original.artifactSha256["capture/timesteps.f32le"],"full_song.original_timesteps")
            try verifyReferenceIdentity(base.appendingPathComponent(execution.decoderBundle+"/reference.json"),expected:execution.decoderReferenceSha256)
            let windowConfiguration = try decoder.decode(Reference30Configuration.self,from:Data(contentsOf:base.appendingPathComponent("drums-reference30-generation.json")))
            try checkManifest(windowConfiguration.decoderReferenceSha256,execution.decoderReferenceSha256,"full_song.decoder_binding")
            let times = try FullSongIO.read(input.appendingPathComponent("timesteps.f32le"),offset:0,count:51)
            guard times[0] == 1, times[50] == 0, zip(times,times.dropFirst()).allSatisfy({ $0.0 > $0.1 }) else { throw ShardError.invalidManifest }
            if FileManager.default.fileExists(atPath:output.appendingPathComponent("completed.json").path) {
                let done = try decoder.decode(FullSongCompletion.self,from:Data(contentsOf:output.appendingPathComponent("completed.json")))
                guard done.schemaVersion == 1, done.executionSha256 == executionSHA, done.jobReferenceSha256 == execution.jobReferenceSha256,
                      done.targetSamples48 == job.targetSamples48, done.completedWindows == job.runs.count,
                      Set(done.artifactSha256.keys) == Set(["piano.wav","bass.wav","drums.wav","guitar.wav","piano-with-band.wav"]) else { throw FullSongFailure.invalid("Saved full-song completion differs") }
                for (name,hash) in done.artifactSha256 {
                    try FullSongIO.regular(output.appendingPathComponent(name),bytes:58+job.targetSamples48*8)
                    guard try streamingSHA256(output.appendingPathComponent(name)) == hash else { throw FullSongFailure.invalid("Saved full-song WAV SHA256 differs; no replacement") }
                }
                lines.append("Full-song collection: COMPLETED; existing final WAV hashes verified; model calls0, no rewritten audio")
                completed = true
            } else {
            let globalContext = try FullSongIO.read(input.appendingPathComponent("source-context.f32le"),offset:0,count:job.latentFrames*128)
            let globalEpsilon = try FullSongIO.read(input.appendingPathComponent("source-epsilon.f32le"),offset:0,count:job.latentFrames*64)
            for f in 0..<job.latentFrames { for c in 64..<128 { guard globalContext[f*128+c] == 1 else { throw ShardError.invalidManifest } } }
            lines.append("Job reference SHA256: \(execution.jobReferenceSha256); execution SHA256: \(executionSHA)")
            lines.append("Song frames48 \(job.targetSamples48); seconds \(Double(job.frames32)/32000); planned windows \(job.runs.count), instruments3; one shared sampler")
            lines.append("All global input SHA256/geometry verified; each consumed window context/noise/audio/epsilon checked against saved slice hashes")
            save()
            var staticAssetsVerified = false
            for (roleIndex,role) in job.instruments.enumerated() {
                let stemFile = output.appendingPathComponent("raw-"+role+".f32le"), stemPointer = output.appendingPathComponent(role+"-assembled.json")
                let stemDone: Bool
                if FileManager.default.fileExists(atPath:stemPointer.path) {
                    let saved = try decoder.decode(FullSongStemCheckpoint.self,from:Data(contentsOf:stemPointer))
                    try FullSongIO.regular(stemFile,bytes:job.targetSamples48*8)
                    guard saved.schemaVersion == 1, saved.executionSha256 == executionSHA, saved.instrument == role,
                          saved.targetSamples48 == job.targetSamples48, try streamingSHA256(stemFile) == saved.rawSha256 else { throw FullSongFailure.invalid("Assembled stem checkpoint differs") }
                    stemDone = true
                } else { stemDone = false }
                var overlap: FullSongOverlap?
                if !stemDone {
                    guard let current = CoreAIProcessFootprintSampler.currentFootprint(), current+UInt64(job.targetSamples48)*24+16777216 < 2000*1048576 else { throw ShardError.memoryBudgetReached }
                    overlap = FullSongOverlap(job)
                }
                let condition = try FullSongIO.read(input.appendingPathComponent(role+"/condition.f32le"),offset:0,count:127*2048)
                let mask = try FullSongIO.read(input.appendingPathComponent(role+"/condition-mask.f32le"),offset:0,count:127)
                guard mask.allSatisfy({ $0 == 1 }) else { throw ShardError.invalidManifest }
                let noise = try FullSongIO.read(input.appendingPathComponent(role+"/noise-global.f32le"),offset:0,count:job.latentFrames*64)
                for (i,w) in job.windows.enumerated() {
                    if await shouldPause() { paused = true; throw FullSongFailure.invalid("Dijeda pada batas window; hasil selesai tetap tersimpan") }
                    try requirePreviewMemoryBudget()
                    let row = job.runs[roleIndex*job.windows.count+i]
                    let checkpointFolder = output.appendingPathComponent(role+String(format:"/window-%03d",i),isDirectory:true)
                    try FileManager.default.createDirectory(at:checkpointFolder,withIntermediateDirectories:true)
                    let checkpoint = checkpointFolder.appendingPathComponent("completed.json")
                    var rawURL: URL
                    if FileManager.default.fileExists(atPath:checkpoint.path) {
                        let saved = try decoder.decode(FullSongCheckpoint.self,from:Data(contentsOf:checkpoint))
                        guard saved.schemaVersion == 1, saved.executionSha256 == executionSHA, saved.jobReferenceSha256 == execution.jobReferenceSha256,
                              saved.instrument == role, saved.windowIndex == i, saved.childDirectory.hasPrefix("attempt-"),
                              UUID(uuidString:String(saved.childDirectory.dropFirst(8))) != nil else { throw FullSongFailure.invalid("Window checkpoint identity differs") }
                        let child = checkpointFolder.appendingPathComponent(saved.childDirectory,isDirectory:true)
                        rawURL = child.appendingPathComponent("raw-"+role+".f32le")
                        try FullSongIO.regular(rawURL,bytes:2880000*4)
                        guard try streamingSHA256(rawURL) == saved.rawSha256,
                              try streamingSHA256(child.appendingPathComponent("report.txt")) == saved.reportSha256 else { throw FullSongFailure.invalid("Saved window SHA256 differs; no automatic replacement") }
                        reusedWindows += 1
                        lines.append("Reused verified \(role) window\(i+1)/\(job.windows.count); raw SHA256 \(saved.rawSha256)")
                    } else {
                        let context = Array(globalContext[w.latentStart*128..<w.latentEnd*128])
                        let windowNoise = Array(noise[w.latentStart*64..<w.latentEnd*64])
                        var epsilon: [Float] = []; epsilon.reserveCapacity(48000)
                        for c in 0..<64 { epsilon.append(contentsOf:globalEpsilon[c*job.latentFrames+w.latentStart..<c*job.latentFrames+w.latentEnd]) }
                        guard FullSongIO.hash(epsilon) == row.sourceEpsilonSha256 else { throw ShardError.invalidManifest }
                        do {
                            let piano = try FullSongIO.stereo(input.appendingPathComponent("piano48.f32le"),plane:job.latentFrames*1920,start:w.sampleStart48,count:1440000)
                            guard FullSongIO.hash(piano) == row.sourceAudioSha256 else { throw ShardError.invalidManifest }
                        }
                        let prefix = "\(role) — bagian \(i+1)/\(job.windows.count); selesai \(newWindows+reusedWindows)/\(job.runs.count)"
                        lines.append("Starting \(role) window\(i+1); restarting an unfinished window, if any, starts a new visible attempt; prior files retained")
                        save()
                        let update: @MainActor @Sendable (String) -> Void = { text in progress(prefix+"\n"+text) }
                        let result = await generateDrumReference30(instrument:role,fullSongWindow:FullSongWindowInput(jobSHA:execution.jobReferenceSha256,run:row,
                            condition:condition,context:context,noise:windowNoise,times:times,outputParent:checkpointFolder,staticAssetsVerified:staticAssetsVerified),progress:update)
                        guard result.report.contains("Full-song window raw completion: YES"), result.report.contains("Reference30 execution: COMPLETED"),
                              result.report.contains("sampled budget <2000MiB: PASS"), let child = result.folder else {
                            lines.append("Window incomplete; child report retained: "+(result.folder?.path ?? "unavailable"))
                            throw FullSongFailure.invalid("Window generation/resource gate did not complete; no checkpoint committed")
                        }
                        rawURL = child.appendingPathComponent("raw-"+role+".f32le")
                        try FullSongIO.regular(rawURL,bytes:2880000*4)
                        let saved = FullSongCheckpoint(schemaVersion:1,executionSha256:executionSHA,jobReferenceSha256:execution.jobReferenceSha256,
                            instrument:role,windowIndex:i,childDirectory:child.lastPathComponent,rawSha256:try streamingSHA256(rawURL),reportSha256:try streamingSHA256(child.appendingPathComponent("report.txt")))
                        try encoder.encode(saved).write(to:checkpoint,options:.atomic)
                        staticAssetsVerified = true; newWindows += 1
                        lines.append("Completed \(role) window\(i+1)/\(job.windows.count); raw SHA256 \(saved.rawSha256)")
                    }
                    if let overlap { try overlap.add(try FullSongIO.read(rawURL,offset:0,count:2880000),index:i) }
                    save()
                    await progress("Selesai \(newWindows+reusedWindows)/\(job.runs.count) bagian; hasil tersimpan")
                }
                if let overlap {
                    try requirePreviewMemoryBudget()
                    try overlap.finish(to:stemFile)
                    let saved = FullSongStemCheckpoint(schemaVersion:1,executionSha256:executionSHA,instrument:role,targetSamples48:job.targetSamples48,rawSha256:try streamingSHA256(stemFile))
                    try encoder.encode(saved).write(to:stemPointer,options:.atomic)
                }
                overlap = nil
                lines.append("Full raw stem assembled: \(role); frames\(job.targetSamples48); normalized overlap-add before gain")
                save()
            }
            guard newWindows+reusedWindows == job.runs.count else { throw FullSongFailure.invalid("Incomplete job; no final mix") }
            try requirePreviewMemoryBudget()
            await progress("Menyiapkan WAV lagu penuh dengan satu gain bersama…")
            let gain = try FullSongListening.write(job:job,input:input,output:output,progress:{ _ in })
            var hashes: [String:String] = [:]
            for name in ["piano.wav","bass.wav","drums.wav","guitar.wav","piano-with-band.wav"] { hashes[name] = try streamingSHA256(output.appendingPathComponent(name)) }
            let done = FullSongCompletion(schemaVersion:1,executionSha256:executionSHA,jobReferenceSha256:execution.jobReferenceSha256,
                targetSamples48:job.targetSamples48,completedWindows:job.runs.count,listeningGain:gain,artifactSha256:hashes)
            try encoder.encode(done).write(to:output.appendingPathComponent("completed.json"),options:.atomic)
            lines.append("Full-song collection: COMPLETED; frames48\(job.targetSamples48); windows\(job.runs.count)/\(job.runs.count)")
            lines.append("Shared listening gain \(gain); band mix0.25 each piano/bass/drums/guitar; raw comparisons/data precede gain; no per-window gain")
            for (name,sha) in hashes { lines.append("WAV \(name) SHA256: \(sha)") }
            completed = true
            }
        } catch {
            lines.append(paused ? "Full-song collection: PAUSED" : "Full-song collection: NOT COMPLETED")
            lines.append(contentsOf:modelErrorDetails(error))
        }
        let usage = memory.finish()
        lines.append("This invocation: new windows\(newWindows), reused verified windows\(reusedWindows); no inference retries/cache deletion")
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10ms): \(Double(peak)/1048576); sampled budget <2000MiB: \(peak < 2000*1048576 ? "PASS" : "NOT PASSED")") }
        else { lines.append("Sampled memory: NOT EVALUATED") }
        lines.append("Full-song numerical parity/music acceptance: NOT EVALUATED; actual source encoding was host-prepared; baseline/precision/bounds unchanged")
        lines.append("Thermal end: \(ProcessInfo.processInfo.thermalState.rawValue); invocation ms \(ms(began.duration(to:clock.now)))")
        save()
        let restored = completed ? latestFullSong() : nil
        return FullSongResult(report:lines.joined(separator:"\n"),folder:folder,audio:restored?.audio ?? [:])
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

    fileprivate static func currentFootprint() -> UInt64? {
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
