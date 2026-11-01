// API declarations checked against the user's Xcode 27 SDK interfaces.
// The user reported a successful tensor probe on iOS 27.0.1; see reviews/.
// Affine, codec and FP32 DiT profiles passed; FP16 weight-storage testing is next.
import CoreAI
import SwiftUI
import UIKit

@main
@MainActor
struct CoreAIDeviceProbeApp: App {
    @State private var report = "Belum dijalankan."
    @State private var runningModel = false

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Periksa runtime dan tensor CoreAI pada perangkat ini.")
                        Button("Jalankan pemeriksaan") {
                            do {
                                report = try CoreAIDeviceProbe.run()
                            } catch {
                                report = "Pemeriksaan gagal: \(error)"
                            }
                        }
                        .disabled(runningModel)
                        Button(runningModel ? "Memeriksa model…" : "Uji model native") {
                            runningModel = true
                            report = "Memuat aset native .aimodel melalui CoreAI…"
                            Task {
                                report = await CoreAINativeNumericalProbe.run()
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji decoder pretrained") {
                            runningModel = true
                            report = "Memuat decoder pretrained dan membandingkan hasil dengan PyTorch…"
                            Task {
                                report = await CoreAIDecoderProbe.run()
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji decoder 1 detik") {
                            runningModel = true
                            report = "Memuat profil decoder pretrained 1 detik…"
                            Task {
                                report = await CoreAIDecoderProbe.run(oneSecond: true)
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji encoder 1 detik") {
                            runningModel = true
                            report = "Memuat encoder pretrained dan memeriksa latent terhadap PyTorch…"
                            Task {
                                report = await CoreAIDecoderProbe.runEncoder()
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji blok generator") {
                            runningModel = true
                            report = "Memuat blok DiT pretrained dan membandingkan activations dengan PyTorch…"
                            Task {
                                report = await CoreAIDecoderProbe.runDiTBlock()
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji DiT konteks panjang") {
                            runningModel = true
                            report = "Memuat dua blok DiT pretrained dengan konteks 160 token…"
                            Task {
                                report = await CoreAIDecoderProbe.runDiTContext()
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji bobot FP16") {
                            runningModel = true
                            report = "Memuat bobot FP16 dengan hitungan FP32 dan memeriksa baseline…"
                            Task {
                                report = await CoreAIDecoderProbe.runDiTStorage16()
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Text(report)
                            .font(.system(.footnote, design: .monospaced))
                            .textSelection(.enabled)
                        ShareLink(item: report) {
                            Label("Bagikan laporan", systemImage: "square.and.arrow.up")
                        }
                    }
                    .padding()
                }
                .navigationTitle("CoreAI Device Probe")
            }
        }
    }
}

@MainActor
private enum CoreAIDeviceProbe {
    enum ProbeError: Error {
        case tensorMetadataMismatch
        case tensorRoundTripMismatch
    }

    static func run() throws -> String {
        let expected: [Float] = [0, 0.25, -0.5, 1]
        let tensor = NDArray(scalars: expected, shape: [1, 4])
        guard tensor.shape == [1, 4], tensor.scalarType == .float32 else {
            throw ProbeError.tensorMetadataMismatch
        }
        let actual = tensor.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            Array(UnsafeBufferPointer(start: pointer, count: expected.count))
        }
        guard actual == expected else {
            throw ProbeError.tensorRoundTripMismatch
        }

        #if targetEnvironment(simulator)
        let executionTarget = "Simulator; not physical-device evidence"
        #else
        let executionTarget = "iOS device build"
        #endif

        let kinds = ComputeUnitKind.availableKinds
            .map { String(describing: $0) }.sorted().joined(separator: ", ")
        return """
        Target: \(executionTarget)
        Device family: \(UIDevice.current.model)
        OS: \(ProcessInfo.processInfo.operatingSystemVersionString)
        CoreAI architecture: \(AIModel.deviceArchitectureName)
        Reported available compute kinds: \(kinds)
        NDArray Float32 round-trip: PASS
        Shape: \(tensor.shape)
        Values: \(actual)
        Model loaded: NO
        Model inference executed: NO
        Accelerator used by a model: NOT MEASURED
        """
    }
}
