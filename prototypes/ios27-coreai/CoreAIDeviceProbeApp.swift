// API declarations checked against the user's Xcode 27 SDK interfaces.
// The user reported a successful tensor probe on iOS 27.0.1; see reviews/.
// One complete source CPU-request bass preview and playback confirmed by user.
import AVFoundation
import CoreAI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

@main
@MainActor
struct CoreAIDeviceProbeApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @State private var report: String
    @State private var runningModel = false
    @State private var generatedBass: URL?
    @State private var generatedMix: URL?
    @State private var audioPlayer: AVAudioPlayer?
    @State private var restoredPreview = false
    @State private var showingAudioImporter = false
    @State private var importedPiano: CoreAIImportedPiano?

    init() {
        let saved = CoreAIDecoderProbe.latestSavedBassPreview()
        _report = State(initialValue: CoreAIDecoderProbe.savedDenoiserReport() ?? saved?.report ?? "Belum dijalankan.")
        _generatedBass = State(initialValue: saved?.bassURL)
        _generatedMix = State(initialValue: saved?.mixURL)
        _restoredPreview = State(initialValue: saved != nil)
    }

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if hasResource(window.previewBundle + "/reference.json") {
                            Text("Buat bass untuk bagian awal rekaman piano Anda.")
                            Text("Pilih audio mono atau stereo minimal \(window.durationLabel) detik. File asli disimpan; generasi memakai \(window.durationLabel) detik awal.")
                                .font(.footnote)
                            Button("Pilih rekaman piano") { showingAudioImporter = true }
                                .disabled(runningModel)
                            if let importedPiano {
                                Text(importedPiano.metadata.name)
                                    .font(.callout)
                                Button("Dengarkan bagian piano") { playPreview(importedPiano.previewURL) }
                                    .disabled(runningModel)
                                Button("Generate bass dari rekaman") { startGeneration(importedPiano: importedPiano) }
                                    .disabled(runningModel)
                            }
                            Divider()
                        }
                        Text("Pemeriksaan dan baseline")
                            .font(.headline)
                        Button("Jalankan pemeriksaan") {
                            do {
                                report = try CoreAIDeviceProbe.run()
                            } catch {
                                report = "Pemeriksaan gagal: \(error)"
                            }
                        }
                        .disabled(runningModel)
                        if hasResource("affine-native-reference.json") {
                            Button(runningModel ? "Memeriksa model…" : "Uji model native") {
                                runningModel = true
                                report = "Memuat aset native .aimodel melalui CoreAI…"
                                Task {
                                    report = await CoreAINativeNumericalProbe.run()
                                    runningModel = false
                                }
                            }
                            .disabled(runningModel)
                        }
                        if includesLegacyDiagnostics {
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
                        }
                        if hasResource(window.sourceBundle + "/reference.json") {
                            Button("Uji audio-context piano") {
                                runningModel = true
                                report = "Mengubah audio piano menjadi posterior sample dan konteks lego…"
                                Task {
                                    report = await CoreAIDecoderProbe.runSourceContext { text in self.report = text }
                                    runningModel = false
                                }
                            }
                            .disabled(runningModel)
                        }
                        if hasResource(window.conditionBundle + "/reference.json") {
                            Button("Uji conditioning bass") {
                                runningModel = true
                                report = "Menghitung embedding Qwen dan conditioning instrumen bass…"
                                Task {
                                    report = await CoreAIDecoderProbe.runBassConditioning { text in self.report = text }
                                    runningModel = false
                                }
                            }
                            .disabled(runningModel)
                        }
                        if hasResource(window.previewBundle + "/reference.json") {
                            Button("Uji baseline piano \(window.durationLabel) detik") { startGeneration() }
                            .disabled(runningModel)
                        }
                        if let generatedBass {
                            let generationReport = generatedBass.deletingLastPathComponent().appendingPathComponent("report.txt")
                            if restoredPreview {
                                Text("Audio dari generasi terakhir yang tersimpan.")
                                    .font(.footnote)
                            }
                            Button("Lihat laporan generasi") {
                                do {
                                    report = try String(contentsOf: generationReport, encoding: .utf8)
                                } catch {
                                    report = "Laporan generasi tidak dapat dibaca: \(error)"
                                }
                            }
                            .disabled(runningModel)
                            ShareLink(item: generationReport) { Label("Bagikan laporan generasi", systemImage: "square.and.arrow.up") }
                                .disabled(runningModel)
                            Button("Dengarkan bass") { playPreview(generatedBass) }
                                .disabled(runningModel)
                            ShareLink(item: generatedBass) { Label("Bagikan WAV bass", systemImage: "square.and.arrow.up") }
                                .disabled(runningModel)
                            let pianoClip = generatedBass.deletingLastPathComponent().appendingPathComponent("piano.wav")
                            if FileManager.default.fileExists(atPath: pianoClip.path) {
                                ShareLink(item: pianoClip) { Label("Bagikan WAV piano input", systemImage: "square.and.arrow.up") }
                                    .disabled(runningModel)
                            }
                            let inputMetadata = generatedBass.deletingLastPathComponent().appendingPathComponent("input.json")
                            if FileManager.default.fileExists(atPath: inputMetadata.path) {
                                ShareLink(item: inputMetadata) { Label("Bagikan info rekaman", systemImage: "square.and.arrow.up") }
                                    .disabled(runningModel)
                            }
                        }
                        if let generatedMix {
                            Button("Dengarkan bersama piano") { playPreview(generatedMix) }
                                .disabled(runningModel)
                            ShareLink(item: generatedMix) { Label("Bagikan WAV piano + bass", systemImage: "square.and.arrow.up") }
                                .disabled(runningModel)
                        }
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
                .onChange(of: runningModel) { _, running in
                    UIApplication.shared.isIdleTimerDisabled = running && scenePhase == .active
                }
                .onChange(of: scenePhase) { _, phase in
                    UIApplication.shared.isIdleTimerDisabled = runningModel && phase == .active
                }
                .fileImporter(isPresented: $showingAudioImporter, allowedContentTypes: [.audio], allowsMultipleSelection: false) { result in
                    importRecording(result)
                }
            }
        }
    }

    private func importRecording(_ selection: Result<[URL], Error>) {
        guard !runningModel else { return }
        switch selection {
        case .failure(let error):
            report = "Pemilihan audio tidak berhasil: \(error.localizedDescription)"
        case .success(let urls):
            guard let selectedURL = urls.first else { return }
            audioPlayer?.stop()
            importedPiano = nil
            runningModel = true
            report = "Menyalin rekaman dan menyiapkan \(window.durationLabel) detik awal di iPhone…"
            let selectedWindow = window
            Task {
                do {
                    let imported = try await Task.detached(priority: .userInitiated) {
                        try CoreAIAudioImport.importPiano(from: selectedURL, window: selectedWindow)
                    }.value
                    importedPiano = imported
                    report = "Rekaman siap: \(imported.metadata.name)\nBagian awal \(window.durationLabel) detik, stereo 48 kHz. File asli disimpan tanpa diubah.\nDengarkan bagian piano, lalu tekan Generate bass dari rekaman."
                } catch {
                    report = "Impor gagal: \(error.localizedDescription)"
                }
                runningModel = false
            }
        }
    }

    private func startGeneration(importedPiano: CoreAIImportedPiano? = nil) {
        guard !runningModel else { return }
        audioPlayer?.stop()
        generatedBass = nil; generatedMix = nil
        restoredPreview = false
        runningModel = true
        report = importedPiano == nil
            ? "Menjalankan uji baseline piano ke WAV bass pendek…"
            : "Membuat bass untuk rekaman pilihan Anda…"
        let selectedWindow = window
        Task {
            let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
            let result = await Task.detached(priority: .userInitiated) {
                await CoreAIDecoderProbe.generateBassPreview(window: selectedWindow, importedPiano: importedPiano, progress: update)
            }.value
            report = result.report
            generatedBass = result.bassURL
            generatedMix = result.mixURL
            runningModel = false
        }
    }

    private let window = CoreAIPreviewWindow.bundled

    private func hasResource(_ name: String) -> Bool {
        guard let root = Bundle.main.resourceURL else { return false }
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("ModelCandidates/" + name).path)
    }

    private var includesLegacyDiagnostics: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return manifest["selected_profile"] as? String == "all"
    }

    private func playPreview(_ url: URL) {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
        } catch {
            report += "\nAudio playback error: \(error)"
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
