// Component gates and one complete sharded denoiser forward passed on device.
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
        let artifactSha256: [String: String]
        let absoluteTolerance: Double
        let relativeTolerance: Double
        let relativeRmseTolerance: Double
    }

    static func runDenoiserLoop(progress: @escaping @MainActor @Sendable (String) -> Void) async -> String {
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
        let clock = ContinuousClock(); let start = clock.now
        let memory = CoreAIProcessFootprintSampler()
        var completedSteps = 0; var completedStages = 0
        do {
            guard let resources = Bundle.main.resourceURL else { throw ShardError.missingPreparedBundle }
            let folder = resources.appendingPathComponent("ModelCandidates/denoiser-sharded")
            let fixture = resources.appendingPathComponent("ModelCandidates/denoiser-loop-4")
            let contract = try readShardedContract(folder: folder)
            guard let stages = contract.stages else { throw ShardError.invalidManifest }
            let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
            let loop = try decoder.decode(LoopContract.self, from: Data(contentsOf: fixture.appendingPathComponent("reference.json")))
            let schedule: [Float] = [1, 0.75, 0.5, 0.25, 0]
            guard loop.schemaVersion == 1, loop.profile == "denoiser-loop-4", loop.method == "euler_ode",
                  loop.timesteps == schedule, loop.steps.count == 4, loop.shift == 1, loop.guidanceScale == 1,
                  !loop.kvCacheEnabled, loop.inputShapes == Profile.denoiser.inputShapes, loop.outputShape == [1,25,64],
                  loop.weightStorageDtype == "float16", loop.computeDtype == "float32",
                  loop.scalarType == "float32", loop.byteOrder == "little", loop.checkpointFullSha256Verified,
                  loop.checkpointDigest == contract.checkpoint.digest,
                  loop.absoluteTolerance == contract.absoluteTolerance, loop.relativeTolerance == contract.relativeTolerance,
                  loop.relativeRmseTolerance == contract.relativeRmseTolerance,
                  Set(loop.sharedInputSha256.keys) == Set(["noise", "condition", "context"]) else {
                throw ShardError.invalidManifest
            }
            var expectedFiles = Set<String>()
            for (index, step) in loop.steps.enumerated() {
                let n = index + 1
                guard step.index == n, step.timestep == schedule[index], step.nextTimestep == schedule[index+1],
                      step.timestepR == step.timestep, step.velocityFile == "step-\(n)-velocity.f32le",
                      step.latentFile == "step-\(n)-latent.f32le", step.originalVelocityFile == "step-\(n)-velocity-fp32.f32le",
                      step.originalLatentFile == "step-\(n)-latent-fp32.f32le" else { throw ShardError.invalidManifest }
                expectedFiles.formUnion([step.velocityFile, step.latentFile, step.originalVelocityFile, step.originalLatentFile])
            }
            guard Set(loop.artifactSha256.keys) == expectedFiles else { throw ShardError.invalidManifest }
            var references: [String: [Float]] = [:]
            for name in expectedFiles {
                references[name] = try readFloats(folder: fixture, name: name, count: 1600, hashes: loop.artifactSha256)
            }
            var shared: [String: NDArray] = [:]
            var latent: [Float] = []
            for name in ["noise", "condition", "context"] {
                guard loop.sharedInputSha256[name] == contract.artifactSha256["\(name).f32le"],
                      let shape = Profile.denoiser.inputShapes[name] else { throw ShardError.invalidManifest }
                let values = try readFloats(folder: folder, name: "\(name).f32le", count: shape.reduce(1,*), hashes: contract.artifactSha256)
                if name == "noise" { latent = values }
                else { shared[name] = NDArray(scalars: values, shape: shape) }
            }
            lines.append("SHA-256: VERIFIED all 16 loop references and shared synthetic inputs; checkpoint contract matched")
            for step in loop.steps {
                let stepStart = clock.now
                var state = shared
                state["noise"] = NDArray(scalars: latent, shape: [1,25,64])
                state["timestep"] = NDArray(scalars: [step.timestep], shape: [1])
                state["timestep_r"] = NDArray(scalars: [step.timestepR], shape: [1])
                lines.append("Step \(step.index)/4: t=\(step.timestep), next=\(step.nextTimestep), timestep_r=\(step.timestepR)")
                for (index, stage) in stages.enumerated() {
                    guard let footprint = CoreAIProcessFootprintSampler.currentFootprint() else { throw ShardError.footprintUnavailable }
                    lines.append("Before step \(step.index) stage \(index+1)/14 footprint MiB: \(Double(footprint)/1048576)")
                    guard footprint < 2000 * 1048576 else { throw ShardError.memoryBudgetReached }
                    let phase = "Phase: step \(step.index)/4 stage \(index+1)/14 \(stage.assetFile) loading/running"
                    saveProgress(lines + [phase], profile: .denoiser)
                    await progress((lines + [phase]).joined(separator: "\n"))
                    var inputs: [String: NDArray] = [:]
                    for name in stage.inputShapes.keys { inputs[name] = state[name] }
                    let result = try await runDetachedStage(stage, folder: folder, inputs: inputs)
                    for (name, tensor) in result.values {
                        guard let target = stage.outputBindings?[name] else { throw ShardError.invalidManifest }
                        state[target] = tensor
                    }
                    completedStages += 1
                    lines.append("Step \(step.index) stage \(index+1)/14 \(stage.assetFile): load ms \(result.loadMs); run+detached-copy ms \(result.runMs)")
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
                saveProgress(lines, profile: .denoiser)
                await progress(lines.joined(separator: "\n"))
            }
            lines.append("Four-step Euler loop dual parity: PASS (feedback at all four steps)")
        } catch {
            lines.append("Four-step Euler loop dual parity: NOT PASSED")
            lines.append("Error: \(String(reflecting: error))")
        }
        let usage = memory.finish()
        if let baseline = usage.baseline { lines.append("Process baseline MiB: \(Double(baseline)/1048576)") }
        if let peak = usage.peak { lines.append("Sampled process maximum MiB (10 ms): \(Double(peak)/1048576)") }
        lines.append("Completed parity steps: \(completedSteps)/4; completed stages: \(completedStages)/56; total loop probe ms: \(ms(start.duration(to: clock.now)))")
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
        guard contract.schemaVersion == 1, contract.profile == "denoiser-sharded", contract.assetFile == "sharded-pipeline",
              contract.layerCount == 24, contract.outputName == "velocity", contract.outputShape == [1,25,64],
              contract.scalarType == "float32", contract.byteOrder == "little", contract.latentFrames == 25,
              contract.inputShapes == Profile.denoiser.inputShapes, contract.weightStorageDtype == "float16",
              contract.computeDtype == "float32", contract.originalFp32ReferenceFile == "velocity-fp32-reference.f32le",
              contract.checkpointFullSha256Verified, contract.checkpoint.digest == Profile.denoiser.checkpointDigest,
              contract.absoluteTolerance == 1e-4, contract.relativeTolerance == 1e-3, contract.relativeRmseTolerance == 1e-3,
              let stages = contract.stages, stages.count == 14, stages.map({ $0.assetFile }) == assets else {
            throw ShardError.invalidManifest
        }
        for (index, stage) in stages.enumerated() {
            if index == 0 {
                guard stage.layerIndices.isEmpty, stage.inputShapes == Profile.denoiser.inputShapes,
                      stage.outputShapes == frontOutputs, stage.outputBindings == frontBindings else { throw ShardError.invalidManifest }
            } else if index == 13 {
                guard stage.layerIndices.isEmpty, stage.inputShapes == ["hidden":[1,13,2048],"final_temb":[1,2048]],
                      stage.outputShapes == ["velocity":[1,25,64]], stage.outputBindings == ["velocity":"velocity"] else { throw ShardError.invalidManifest }
            } else {
                guard stage.layerIndices == [(index-1)*2,(index-1)*2+1], stage.inputShapes == pairInputs,
                      stage.outputShapes == ["hidden_out":[1,13,2048]], stage.outputBindings == ["hidden_out":"hidden"] else { throw ShardError.invalidManifest }
            }
        }
        return contract
    }

    // All model/function/output references stay inside this async scope. Fresh
    // CPU-owned NDArrays cross its boundary, not model-backed output buffers.
    private static func runDetachedStage(_ stage: Contract.Stage, folder: URL, inputs: [String: NDArray]) async throws
        -> (values: [String: NDArray], loadMs: Double, runMs: Double) {
        let clock = ContinuousClock(); let loadStart = clock.now
        let probe = try await CoreAIModelProbe(contentsOf: folder.appendingPathComponent(stage.assetFile), options: .cpuOnly)
        let loadMs = ms(loadStart.duration(to: clock.now)); let runStart = clock.now
        var outputs = try await probe.runStateless(functionNamed: "main", inputs: inputs)
        var detached: [String: NDArray] = [:]
        for (name, shape) in stage.outputShapes {
            guard let value = outputs.remove(name), let tensor = value.ndArray, tensor.scalarType == .float32,
                  tensor.shape == shape, tensor.interleaveLayout == nil else { throw ProbeError.invalidOutputTensor }
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
            detached[name] = NDArray(scalars: values, shape: shape)
        }
        return (detached, loadMs, ms(runStart.duration(to: clock.now)))
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
            if error > contract.absoluteTolerance + contract.relativeTolerance * abs(Double(reference)) { elementwisePass = false }
        }
        let rmse = sqrt(squaredError / Double(expected.count))
        let referenceRms = sqrt(squaredReference / Double(expected.count))
        let normalizedRmse = rmse / max(referenceRms, 1e-12)
        return (maxError, rmse, normalizedRmse, elementwisePass && normalizedRmse <= contract.relativeRmseTolerance)
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
