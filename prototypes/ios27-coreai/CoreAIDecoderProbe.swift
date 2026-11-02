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

    private static func executeSourceContext(repetitions: Int, progress: @escaping @MainActor @Sendable (String) -> Void) async
        -> (report: String, context: [Float]?) {
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: actual recording -> VAE posterior sample -> full-range lego audio context",
                     "Scope: audio side only; NO text embedding, ConditionEncoder, denoiser or bass generation",
                     "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
                     "Audio: 5.12 s / 48000 Hz stereo; 128 latent frames",
                     "Posterior: mean + (softplus(scale) + 1e-4) * explicit epsilon; fixture epsilon reused for parity",
                     "PCM decode/resampling happens during Mac preparation; raw recording import on iPhone NOT tested",
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
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates/source-context"),
                  FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append("Prepare on Mac: bash scripts/prepare_coreai_source_context.sh; choose your piano recording, then rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(SourceContextContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            let shapes = ["audio":[1,2,245760],"epsilon":[1,64,128],"chunk_mask":[1,128,64]]
            let outputShapes = ["posterior_mean":[1,64,128],"posterior_std":[1,64,128],
                                "source_latents":[1,128,64],"context_latents":[1,128,128]]
            try checkManifest(ref.schemaVersion, 1, "source_context.schema_version")
            try checkManifest(ref.profile, "source-context", "source_context.profile")
            try checkManifest(ref.assetFile, "source-context.aimodel", "source_context.asset_file")
            try checkManifest(ref.functionName, "main", "source_context.function_name")
            try checkManifest(ref.inputShapes, shapes, "source_context.input_shapes")
            try checkManifest(ref.outputShapes, outputShapes, "source_context.output_shapes")
            try checkManifest(ref.scalarType, "float32", "source_context.scalar_type")
            try checkManifest(ref.byteOrder, "little", "source_context.byte_order")
            try checkManifest(ref.sampleRate, 48000, "source_context.sample_rate")
            try checkManifest(ref.audioSeconds, 5.12, "source_context.audio_seconds")
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
                let values = try readFloats(folder: folder, name: "\(name).f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                if name == "chunk_mask" { try checkManifest(values.allSatisfy({ $0 == 1 }), true, "source_context.full_range_mask") }
                inputs[name] = NDArray(scalars: values, shape: shape)
            }
            var expected: [String: [Float]] = [:]
            for (name, shape) in outputShapes {
                expected[name] = try readFloats(folder: folder, name: "\(name)-reference.f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
            }
            lines.append("Recording: \(ref.recording.name); segment start seconds: \(ref.recording.startSeconds)")
            lines.append("Original recording SHA-256 (recorded by Mac preparation): \(ref.recording.sha256)")
            lines.append("SHA-256: VERIFIED all three input and four output-reference binaries; pretrained checkpoint verified by preparation")
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
                    guard let actual = result.values[name], let reference = expected[name] else { throw ProbeError.invalidOutputTensor }
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
            lines.append(repetitions == 3 ? "Real-recording audio-context parity: PASS (three runs, all four outputs)" : "Real-recording audio-context parity: PASS (one integrated call, all four outputs)")
        } catch {
            lines.append("Real-recording audio-context parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed audio-context parity runs: \(completed)/\(repetitions); total probe ms: \(ms(start.duration(to: clock.now)))")
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
    private static func bassConditioningStages() -> [BassConditioningStage] {
        let nt = 105, nl = 11
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

    private static func executeBassConditioning(progress: @MainActor @Sendable (String) -> Void) async
        -> (report: String, condition: [Float]?, mask: [Float]?) {
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var lines = [
            "Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
            "CoreAI architecture: \(AIModel.deviceArchitectureName)",
            "Test: bass prompt -> all 28 Qwen blocks -> complete ACE ConditionEncoder (8 lyric + 4 timbre blocks)",
            "Scope: text/target conditioning; fixed host-tokenized prompt; NO denoiser or bass waveform",
            "Requested compute policy: cpuOnly; accelerator placement NOT MEASURED",
            "27 sequential native assets; FP16 stored weights, FP32 calculations; detached CPU copies between assets",
            "Prompt tokenization on Mac; arbitrary prompt entry/tokenization on iPhone NOT tested",
            "No timbre reference recording: publisher silence latent, first 750 frames; single batch with all tokens valid",
            "Output: condition [1,117,2048], mask [1,117]; source audio context tested separately",
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
            guard let folder = Bundle.main.resourceURL?.appendingPathComponent("ModelCandidates/bass-conditioning"),
                  FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append("Prepare on Mac: bash scripts/prepare_coreai_bass_conditioning.sh; wait for Ready, then rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(BassConditioningContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion, 1, "conditioning.schema_version")
            try checkManifest(ref.profile, "bass-conditioning", "conditioning.profile")
            try checkManifest(ref.functionName, "main", "conditioning.function")
            try checkManifest(ref.batchSize, 1, "conditioning.batch")
            try checkManifest(ref.textTokens, 105, "conditioning.text_tokens")
            try checkManifest(ref.lyricTokens, 11, "conditioning.lyric_tokens")
            try checkManifest(ref.qwenLayerCount, 28, "conditioning.qwen_layers")
            try checkManifest(ref.lyricLayerCount, 8, "conditioning.lyric_layers")
            try checkManifest(ref.timbreLayerCount, 4, "conditioning.timbre_layers")
            try checkManifest(ref.stages, bassConditioningStages(), "conditioning.ordered_stages")
            try checkManifest(ref.qwenCheckpointDigest, "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd", "conditioning.qwen_digest")
            try checkManifest(ref.baseCheckpointDigest, "4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b", "conditioning.base_digest")
            try checkManifest(ref.silenceDigest, "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b", "conditioning.silence_digest")
            try checkManifest(ref.allCheckpointHashesVerified, true, "conditioning.checkpoints_verified")
            try checkManifest(ref.scalarType, "float32", "conditioning.scalar")
            try checkManifest(ref.byteOrder, "little", "conditioning.byte_order")
            try checkManifest(ref.weightStorageDtype, "float16", "conditioning.storage")
            try checkManifest(ref.computeDtype, "float32", "conditioning.compute")
            try checkManifest(ref.absoluteTolerance, 1e-4, "conditioning.abs_tolerance")
            try checkManifest(ref.relativeTolerance, 1e-3, "conditioning.rel_tolerance")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "conditioning.rmse_tolerance")
            try checkManifest(ref.tokenizationOnDevice, false, "conditioning.host_tokenization")
            try checkManifest(ref.referenceAudioMode, "publisher_silence_750", "conditioning.reference_mode")
            try checkManifest(ref.allValidSingleReferenceSpecialization, true, "conditioning.single_reference")
            try checkManifest(ref.hostPartitionVsOriginalFp32.parityPassed, true, "conditioning.partition_gold")
            try checkManifest(ref.hostStorage16VsOriginalFp32.parityPassed, true, "conditioning.storage_gold")
            try checkManifest(ref.prompts.instruction, "Generate the bass track based on the audio context:", "conditioning.instruction")
            try checkManifest(ref.prompts.audioSeconds, 5.12, "conditioning.clip_seconds")
            try checkManifest(ref.prompts.presetSha256, "4bacacb4f9fb1a9e66c98958869e2756878e6420910b2af065bea6522cb6ef63", "conditioning.preset")
            var state: [String: NDArray] = [:]
            let inputs = ["text_ids":[1,105],"lyric_ids":[1,11],"reference_audio":[1,750,64]]
            for (name, shape) in inputs {
                let values = try readFloats(folder: folder, name: "\(name).f32le", count: shape.reduce(1,*), hashes: ref.artifactSha256)
                if name.hasSuffix("_ids") {
                    guard values.allSatisfy({ $0 >= 0 && $0 < 151669 && $0.rounded(.towardZero) == $0 }) else { throw ProbeError.invalidFloatData }
                }
                state[name] = NDArray(scalars: values, shape: shape)
            }
            let milestones = ["text_embeddings":[1,105,1024],"condition":[1,117,2048],"condition_mask":[1,117]]
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
            lines.append("Bass text/ConditionEncoder dual parity: PASS (all 40 layers, six checks)")
        } catch {
            lines.append("Bass text/ConditionEncoder dual parity: NOT PASSED")
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

    private static func bassPreviewStages() -> [Contract.Stage] {
        let shared = ["hidden":[2,64,2048],"condition":[2,117,2048],"temb":[2,6,2048],
                      "rope_cos":[1,64,128],"rope_sin":[1,64,128],"self_mask":[1,1,64,64],
                      "local_mask":[1,1,64,64],"cross_mask":[1,1,64,117]]
        var frontOutputs = shared
        frontOutputs.removeValue(forKey: "condition")
        frontOutputs["condition_projected"] = [2,117,2048]
        frontOutputs["final_temb"] = [2,2048]
        var frontBindings = Dictionary(uniqueKeysWithValues: frontOutputs.keys.map { ($0, $0) })
        frontBindings["condition_projected"] = "condition"
        var result = [Contract.Stage(assetFile: "front.aimodel", layerIndices: [],
            inputShapes: ["noise":[1,128,64],"condition":[1,117,2048],"context":[1,128,128],"timestep":[1]],
            outputShapes: frontOutputs, outputBindings: frontBindings)]
        for i in stride(from: 0, to: 24, by: 2) {
            result.append(Contract.Stage(assetFile: String(format: "blocks-%02d-%02d.aimodel", i, i+1), layerIndices: [i,i+1],
                inputShapes: shared, outputShapes: ["hidden_out":[2,64,2048]], outputBindings: ["hidden_out":"hidden"]))
        }
        result.append(Contract.Stage(assetFile: "tail.aimodel", layerIndices: [],
            inputShapes: ["hidden":[2,64,2048],"final_temb":[2,2048]],
            outputShapes: ["velocity":[2,128,64]], outputBindings: ["velocity":"velocity"]))
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

    private static func streamingSHA256(_ url: URL) throws -> String {
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

    static func generateBassPreview(progress: @escaping @MainActor @Sendable (String) -> Void) async -> BassPreviewResult {
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        let priorReport = savedDenoiserReport()
        var retention = RetentionStats()
        var lines = ["Runtime: CoreAI", "OS: \(ProcessInfo.processInfo.operatingSystemVersionString)",
                     "Probe run: \(UUID().uuidString); started UTC: \(ISO8601DateFormatter().string(from: Date()))",
                     "CoreAI architecture: \(AIModel.deviceArchitectureName)",
                     "Test: actual piano -> VAE posterior -> full Qwen/ConditionEncoder -> 50 APG-guided Euler steps -> bass WAV",
                     "Audio: 5.12 s / 48000 Hz stereo; 128 latent frames / 117 condition tokens",
                     "Guidance 7; APG momentum -0.75, threshold 2.5, projection over latent time (dim 1); shift 1",
                     "APG projection FP32, compared on host against original FP64 projection; not plain CFG",
                     "Requested compute policy: source/Qwen cpuOnly; sampling/decoder policy below; accelerator placement NOT MEASURED",
                     "Fixed host PCM/tokenization/epsilon/noise fixtures; raw recording import, arbitrary prompts and on-device RNG NOT tested",
                     "Actual component outputs feed sampling; host references are comparisons only",
                     "Execution: single live Swift model/function scope per asset; detached CPU copies; native cache eviction NOT GUARANTEED",
                     "Raw-source cache repair: only blocks-18-19 after its saved BNNS graphLoadFailed; skipped for AOT; no automatic retry",
                     "Native API phase markers are persisted before calls; observer overhead is included in total pipeline time",
                     "Footprint guard 2000 MiB before each stage and after bind; NOT an OS peak-memory guarantee"]
        #if targetEnvironment(simulator)
        lines.append("Target: Simulator; not physical-device evidence")
        #else
        lines.append("Target: iOS device build")
        #endif
        var lastPhase = "preview manifest/reference validation"
        var completedSteps = 0, checks = 0, allParity = true
        var bassURL: URL?, mixURL: URL?
        do {
            guard let resources = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let folder = resources.appendingPathComponent("ModelCandidates/bass-preview-512")
            let sourceFolder = resources.appendingPathComponent("ModelCandidates/source-context")
            let conditionFolder = resources.appendingPathComponent("ModelCandidates/bass-conditioning")
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("reference.json").path) else {
                lines.append("Prepare on Mac: bash scripts/prepare_coreai_bass_preview.sh; wait for Ready then rebuild app.")
                throw ShardError.missingPreparedBundle
            }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let ref = try decoder.decode(BassPreviewContract.self, from: Data(contentsOf: folder.appendingPathComponent("reference.json")))
            try checkManifest(ref.schemaVersion, 1, "preview.schema")
            try checkManifest(ref.profile, "bass-preview-512", "preview.profile")
            try checkManifest(ref.functionName, "main", "preview.function")
            try checkManifest(ref.inferenceSteps, 50, "preview.steps")
            try checkManifest(ref.guidanceScale, 7, "preview.guidance")
            try checkManifest(ref.shift, 1, "preview.shift")
            try checkManifest(ref.method, "euler_ode_apg", "preview.method")
            try checkManifest(ref.apgMomentum, -0.75, "preview.momentum")
            try checkManifest(ref.apgNormThreshold, 2.5, "preview.threshold")
            try checkManifest(ref.apgProjectionDims, [1], "preview.projection_dims")
            try checkManifest(ref.apgProjectionPrecision, "float32_validated_against_original_float64", "preview.projection_precision")
            try checkManifest(ref.checkpoints, [1,10,25,50], "preview.checkpoints")
            try checkManifest(ref.latentFrames, 128, "preview.frames")
            try checkManifest(ref.conditionTokens, 117, "preview.condition_tokens")
            try checkManifest(ref.sampleRate, 48000, "preview.sample_rate")
            try checkManifest(ref.audioSeconds, 5.12, "preview.seconds")
            try checkManifest(ref.scalarType, "float32", "preview.dtype")
            try checkManifest(ref.byteOrder, "little", "preview.byte_order")
            try checkManifest(ref.checkpointDigest, "4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b", "preview.base_digest")
            try checkManifest(ref.vaeCheckpointDigest, Profile.encoderOneSecond.checkpointDigest, "preview.vae_digest")
            try checkManifest(ref.allCheckpointsVerified, true, "preview.checkpoints_verified")
            try checkManifest(ref.hostComparisons.map { $0.step }, Array(1...50), "preview.host_steps")
            try checkManifest(ref.apgSameInputVsPublisher.count, 50, "preview.host_apg_count")
            func boundedHostComparison(_ stats: BassPreviewContract.HostComparison) throws -> Bool {
                guard stats.finite, stats.maxAbsoluteError.isFinite, stats.maxAbsoluteError >= 0,
                      stats.rmse.isFinite, stats.rmse >= 0, stats.normalizedRmse.isFinite,
                      stats.normalizedRmse >= 0, stats.normalizedRmse <= 1e-3,
                      stats.elementwiseFractionOverBound.isFinite, stats.elementwiseFractionOverBound >= 0,
                      stats.elementwiseFractionOverBound <= 1 else { throw ShardError.invalidManifest }
                let strict = stats.elementwiseFractionOverBound == 0
                try checkManifest(stats.parityPassed, strict, "preview.host_comparison_flag")
                return strict
            }
            var hostTrajectoryPassed = true
            for step in ref.hostComparisons {
                let passed = try boundedHostComparison(step.latent)
                hostTrajectoryPassed = hostTrajectoryPassed && passed
            }
            for stats in ref.apgSameInputVsPublisher {
                try checkManifest(try boundedHostComparison(stats), true, "preview.host_apg_strict")
            }
            let hostAudioPassed = try boundedHostComparison(ref.hostAudioVsFp32)
            let hostStrict = hostTrajectoryPassed && hostAudioPassed
            try checkManifest(ref.hostDualTrajectoryParityPassed, hostTrajectoryPassed, "preview.host_trajectory")
            if let claimed = ref.hostNumericalGatePassed {
                try checkManifest(claimed, hostStrict, "preview.host_numerical_gate")
            } else if !hostStrict { throw ShardError.invalidManifest }
            let hostStatus = hostStrict ? "strict_pass" : "bounded_diagnostic"
            if let claimed = ref.hostExportStatus {
                try checkManifest(claimed, hostStatus, "preview.host_export_status")
            } else if !hostStrict { throw ShardError.invalidManifest }
            allParity = hostStrict
            lines.append("Host preparation: \(hostStatus); strict gate \(hostStrict ? "PASS" : "NOT PASSED"); original tolerances unchanged")
            if !hostStrict {
                lines.append("Bounded diagnostic bundle: finite, normalized RMSE <= 0.001; pointwise gate failed. Result cannot be labeled overall numerical PASS.")
            }
            try checkManifest(ref.hostDecoderFoldingExact, true, "preview.decoder_fold")
            try checkManifest(ref.absoluteTolerance, 1e-4, "preview.abs")
            try checkManifest(ref.relativeTolerance, 1e-3, "preview.rel")
            try checkManifest(ref.relativeRmseTolerance, 1e-3, "preview.nrmse")
            let expectedStages = bassPreviewStages()
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
            let preparedAOT = try await preparedBassPreviewAOT(source: folder, reportLines: lines + [lastPhase], progress: progress)
            if let preparedAOT {
                lines.append("Sampling/decoder assets: AOT .aimodelc; architecture \(AIModel.deviceArchitectureName); 16 compiled assets SHA-256 VERIFIED")
                lines.append("Sampling/decoder requested policy: default, matching compiler --preferred-compute none; physical accelerator NOT MEASURED")
                lines.append("AOT can still specialize on device; reduced compiler memory/time NOT YET MEASURED")
                lines.append("AOT source identity matched: \(preparedAOT.folder.lastPathComponent)")
            } else {
                lines.append("Sampling/decoder assets: source .aimodel; requested policy cpuOnly; device compilation required")
            }
            let noise = try readFloats(folder: folder, name: "initial-noise.f32le", count: 128*64, hashes: ref.artifactSha256)
            let times = try readFloats(folder: folder, name: "timesteps.f32le", count: 51, hashes: ref.artifactSha256)
            try checkManifest(times[0], Float(1), "preview.time_start")
            try checkManifest(times[50], Float(0), "preview.time_end")
            guard zip(times, times.dropFirst()).allSatisfy({ $0.0 > $0.1 }) else { throw ShardError.invalidManifest }
            // Verify every numerical reference before expensive device execution.
            var references: [String: [Float]] = [:]
            for step in ref.checkpoints {
                for name in ["latent"] {
                    for kind in ["reference","fp32-reference"] {
                        let file = "step-\(step)-\(name)-\(kind).f32le"
                        references[file] = try readFloats(folder: folder, name: file, count: 128*64, hashes: ref.artifactSha256)
                    }
                }
            }
            for kind in ["reference","fp32-reference"] {
                let file = "audio-\(kind).f32le"
                references[file] = try readFloats(folder: folder, name: file, count: 2*245760, hashes: ref.artifactSha256)
            }
            lines.append("SHA-256: VERIFIED component identities, initial noise, schedule and 10 output references")
            // Repair only the observed failing derived CPU cache entry, before
            // any model is opened. Bundled assets/weights/references are untouched.
            // The saved report is captured before component progress overwrites it.
            if case .none = preparedAOT, let priorReport,
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
            let source = await executeSourceContext(repetitions: 1) { text in progress(sourcePrefix + text) }
            lines.append("=== Source component before sampling ===\n" + source.report)
            guard let actualContext = source.context else { throw BassPreviewError.componentNotCompleted("source-context; see component report") }
            let conditionPrefix = lines.joined(separator: "\n") + "\n=== Integrated bass conditioning ===\n"
            lastPhase = "bass conditioning pipeline"
            let conditioning = await executeBassConditioning { text in progress(conditionPrefix + text) }
            lines.append("=== Bass conditioning before sampling ===\n" + conditioning.report)
            guard let actualCondition = conditioning.condition, let mask = conditioning.mask,
                  mask.count == 117, mask.allSatisfy({ $0 == 1 }) else { throw BassPreviewError.componentNotCompleted("bass conditioning; see component report") }
            let contextInput = NDArray(scalars: actualContext, shape: [1,128,128])
            let conditionInput = NDArray(scalars: actualCondition, shape: [1,117,2048])
            lines.append("=== Sampling and waveform generation ===")
            var latent = NDArray(scalars: noise, shape: [1,128,64])
            var momentum = NDArray(scalars: [Float](repeating: 0, count: 128*64), shape: [1,128,64])
            let apgStage = Contract.Stage(assetFile: "apg-euler.aimodel", layerIndices: [],
                inputShapes: ["velocity":[2,128,64],"latent":[1,128,64],"momentum":[1,128,64],"dt":[1]],
                outputShapes: ["latent_next":[1,128,64],"momentum_next":[1,128,64],"guided_velocity":[1,128,64]], outputBindings: nil)
            for index in 0..<50 {
                let stepStart = clock.now
                var state = ["noise":latent,"condition":conditionInput,"context":contextInput,
                             "timestep":NDArray(scalars: [times[index]], shape: [1])]
                for (stageIndex, stage) in ref.stages.enumerated() {
                    let phase = "Phase: sampling step \(index+1)/50, stage \(stageIndex+1)/14 \(stage.assetFile)"
                    lastPhase = phase
                    try requirePreviewMemoryBudget()
                    saveProgress(lines + [phase], profile: .denoiser)
                    await progress((lines + [phase]).joined(separator: "\n"))
                    var inputs: [String: NDArray] = [:]
                    for (name, shape) in stage.inputShapes {
                        guard let array = state[name], array.shape == shape else { throw ShardError.invalidManifest }
                        inputs[name] = array
                    }
                    let result = try await runSequentialPreviewStage(stage, folder: folder, inputs: inputs, aot: preparedAOT, stats: &retention,
                        phaseProgress: previewPhaseProgress(prefix: (lines + [phase]).joined(separator: "\n")))
                    if index == 0 {
                        lines.append("First sampling step stage \(stageIndex+1)/14 \(stage.assetFile): load ms \(result.loadMs); \(result.phases.report)")
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
                saveProgress(lines + [lastPhase], profile: .denoiser)
                await progress((lines + [lastPhase]).joined(separator: "\n"))
                let result = try await runSequentialPreviewStage(apgStage, folder: folder,
                    inputs: ["velocity":velocity,"latent":latent,"momentum":momentum,
                             "dt":NDArray(scalars: [times[index]-times[index+1]], shape: [1])], aot: preparedAOT, stats: &retention,
                    phaseProgress: previewPhaseProgress(prefix: (lines + [lastPhase]).joined(separator: "\n")))
                guard let next = result.values["latent_next"], let nextMomentum = result.values["momentum_next"],
                      result.values["guided_velocity"] != nil else { throw ProbeError.invalidOutputTensor }
                latent = next; momentum = nextMomentum
                if ref.checkpoints.contains(index+1) {
                    for (name, array) in [("latent",latent)] {
                        let actual = try copiedValues(array, shape: [1,128,64])
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
            }
            try checkManifest(retention.modelOpens, 750, "preview.sequential_opens")
            try checkManifest(retention.functionBinds, 750, "preview.sequential_binds")
            try checkManifest(retention.completedCalls, 750, "preview.sequential_calls")
            try checkManifest(retention.cacheHits, 0, "preview.sequential_hits")
            lines.append("Sampling single-stage scope: \(retention.modelOpens) opens, \(retention.functionBinds) binds, \(retention.cacheHits) hits, \(retention.completedCalls) calls")
            let values = try copiedValues(latent, shape: [1,128,64])
            var decoderInput = [Float](repeating: 0, count: 128*64)
            for t in 0..<128 { for c in 0..<64 { decoderInput[c*128+t] = values[t*64+c] } }
            lastPhase = "Phase: decoder.aimodel loading/binding/inference"
            try requirePreviewMemoryBudget()
            lines.append("Phase: loading full 5.12-second VAE decoder; previous Swift stage scope released, native eviction unverified")
            saveProgress(lines, profile: .denoiser); await progress(lines.joined(separator: "\n"))
            let decoderStage = Contract.Stage(assetFile: "decoder.aimodel", layerIndices: [],
                inputShapes: ["latent":[1,64,128]], outputShapes: ["audio":[1,2,245760]], outputBindings: nil)
            let decoded = try await runSequentialPreviewStage(decoderStage, folder: folder,
                inputs: ["latent":NDArray(scalars: decoderInput, shape: [1,64,128])], aot: preparedAOT, stats: &retention,
                phaseProgress: previewPhaseProgress(prefix: (lines + [lastPhase]).joined(separator: "\n")))
            guard let audioArray = decoded.values["audio"] else { throw ProbeError.invalidOutputTensor }
            let audio = try copiedValues(audioArray, shape: [1,2,245760])
            for kind in ["reference","fp32-reference"] {
                guard let expected = references["audio-\(kind).f32le"] else { throw ShardError.invalidManifest }
                let stats = try compare(actual: audio, expected: expected, absoluteTolerance: ref.absoluteTolerance,
                                        relativeTolerance: ref.relativeTolerance, relativeRmseTolerance: ref.relativeRmseTolerance)
                lines.append("Decoded audio vs \(kind): max abs error \(stats.maxError); normalized RMSE \(stats.normalizedRmse); \(stats.passed ? "PASS" : "NOT PASSED")")
                allParity = allParity && stats.passed; checks += 1
            }
            try checkManifest(checks, 10, "preview.output_checks")
            lastPhase = "writing bass and listening-mix WAV files"
            let sourceRef = try decoder.decode(SourceContextContract.self, from: Data(contentsOf: sourceFolder.appendingPathComponent("reference.json")))
            let piano = try readFloats(folder: sourceFolder, name: "audio.f32le", count: 2*245760, hashes: sourceRef.artifactSha256)
            guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { throw ShardError.missingPreparedBundle }
            let output = documents.appendingPathComponent("bass-preview-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let stem = output.appendingPathComponent("bass.wav"), mixed = output.appendingPathComponent("piano-with-bass.wav")
            try writeStereoFloatWAV(audio, frames: 245760, to: stem)
            var mix = zip(piano,audio).map { Float(0.78)*$0.0 + Float(0.55)*$0.1 }
            let mixPeak = mix.reduce(Float(0)) { max($0,abs($1)) }
            let gain = mixPeak > 0.98 ? Float(0.98)/mixPeak : Float(1)
            if gain < 1 { mix = mix.map { $0 * gain } }
            try writeStereoFloatWAV(mix, frames: 245760, to: mixed)
            bassURL = stem; mixURL = mixed
            lines.append("Integrated bass preview numerical parity: \(allParity ? "PASS" : "NOT PASSED") (10 comparisons; fixture scope)")
            lines.append("WAV generated on device: YES; bass.wav and piano-with-bass.wav, 5.12 s stereo Float32/48 kHz")
            lines.append("Raw bass peak: \(audio.reduce(Float(0)) { max($0,abs($1)) }); listening mix gain: \(gain)")
            if !allParity { lines.append("Audio is diagnostic: numerical gate did not pass; musical/device baseline NOT accepted") }
            lines.append("Musical quality, full-song behavior and total pipeline RAM: NOT VALIDATED")
            try lines.joined(separator: "\n").write(to: output.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
        } catch {
            lines.append("Integrated bass preview: NOT COMPLETED")
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
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed sampling steps: \(completedSteps)/50; output checks: \(checks)/10; total pipeline ms: \(ms(start.duration(to: clock.now)))")
        lines.append("Memory scope: sampled app process, not exact peak or total model/device RAM")
        saveProgress(lines, profile: .denoiser)
        if let bassURL { try? lines.joined(separator: "\n").write(to: bassURL.deletingLastPathComponent().appendingPathComponent("report.txt"), atomically: true, encoding: .utf8) }
        return BassPreviewResult(report: lines.joined(separator: "\n"), bassURL: bassURL, mixURL: mixURL)
    }

    private static func writeStereoFloatWAV(_ planar: [Float], frames: Int, to url: URL) throws {
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
            stats.completedCalls += 1
            return (detached, loadMs, runMs, phases)
        } catch {
            throw StageExecutionFailure(asset: stage.assetFile, operation: operation, reused: reused,
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
    static func latestSavedBassPreview() -> BassPreviewResult? {
        let manager = FileManager.default
        guard let documents = manager.urls(for: .documentDirectory, in: .userDomainMask).first,
              let entries = try? manager.contentsOfDirectory(at: documents,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles]) else { return nil }
        let root = documents.standardizedFileURL.resolvingSymlinksInPath().path + "/"
        var candidates: [(folder: URL, date: Date)] = []
        for folder in entries where folder.lastPathComponent.hasPrefix("bass-preview-") {
            guard let metadata = try? folder.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]),
                  metadata.isDirectory == true, metadata.isSymbolicLink != true,
                  folder.standardizedFileURL.resolvingSymlinksInPath().path.hasPrefix(root) else { continue }
            let files = ["bass.wav", "piano-with-bass.wav", "report.txt"].map { folder.appendingPathComponent($0) }
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
                bassURL: candidate.folder.appendingPathComponent("bass.wav"),
                mixURL: candidate.folder.appendingPathComponent("piano-with-bass.wav"))
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
