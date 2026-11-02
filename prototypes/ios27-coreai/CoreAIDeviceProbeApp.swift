// API declarations checked against the user's Xcode 27 SDK interfaces.
// The user reported a successful tensor probe on iOS 27.0.1; see reviews/.
// Component gates passed; a complete 24-layer denoiser forward is next.
import CoreAI
import SwiftUI
import UIKit

@main
@MainActor
struct CoreAIDeviceProbeApp: App {
    @State private var report = CoreAIDecoderProbe.savedDenoiserReport() ?? "Belum dijalankan."
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
                        Button("Uji denoiser bertahap") {
                            runningModel = true
                            report = "Menjalankan denoiser per bagian…"
                            Task {
                                report = await CoreAIDecoderProbe.runShardedDenoiser { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji kestabilan 3x") {
                            runningModel = true
                            report = "Memeriksa tiga forward berurutan dalam aplikasi yang sama…"
                            Task {
                                report = await CoreAIDecoderProbe.runDenoiserStability { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji loop denoising 4 langkah") {
                            runningModel = true
                            report = "Menjalankan empat langkah Euler dengan feedback latent…"
                            Task {
                                report = await CoreAIDecoderProbe.runDenoiserLoop { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Profil waktu loop 4 langkah") {
                            runningModel = true
                            report = "Mengukur pembukaan model, binding fungsi, eksekusi dan copy output…"
                            Task {
                                report = await CoreAIDecoderProbe.runDenoiserLoop(profilePhases: true) { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji reuse satu pasangan layer") {
                            runningModel = true
                            report = "Membandingkan run pertama dan tiga run ulang pada model/fungsi yang sama…"
                            Task {
                                report = await CoreAIDecoderProbe.runPairReuse { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji loop 24 layer dengan reuse") {
                            runningModel = true
                            report = "Menguji empat langkah dengan 14 model/fungsi yang dipertahankan…"
                            Task {
                                report = await CoreAIDecoderProbe.runDenoiserLoop(profilePhases: true, retainStages: true) { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji audio-context piano") {
                            runningModel = true
                            report = "Mengubah audio piano menjadi posterior sample dan konteks lego…"
                            Task {
                                report = await CoreAIDecoderProbe.runSourceContext { text in self.report = text }
                                runningModel = false
                            }
                        }
                        .disabled(runningModel)
                        Button("Uji conditioning bass") {
                            runningModel = true
                            report = "Menghitung embedding Qwen dan conditioning instrumen bass…"
                            Task {
                                report = await CoreAIDecoderProbe.runBassConditioning { text in self.report = text }
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
