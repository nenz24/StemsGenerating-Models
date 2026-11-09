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
    @State private var workStartedInProcess = false
    @State private var preflightAttempted = false
    @State private var generatedBass: URL?
    @State private var generatedMix: URL?
    @State private var audioPlayer: AVAudioPlayer?
    @State private var restoredPreview = false
    @State private var showingAudioImporter = false
    @State private var importedPiano: CoreAIImportedPiano?

    init() {
        let active = CoreAIPreviewWindow.bundled
        let decoderOnly = CoreAIPreviewWindow.drumReference30DecoderOnlyEnabled
        let drum30 = CoreAIPreviewWindow.drumReference30ComponentsEnabled || decoderOnly
        let saved = drum30 ? nil : CoreAIDecoderProbe.latestSavedBassPreview(window: active)
        let prior = CoreAIDecoderProbe.savedDenoiserReport()
        let matchingPrior: String? = prior.flatMap { text in
            text.contains("Selected instrument: \(active.stemID);") || (active.stemID == "bass" && !text.contains("Selected instrument:")) ? text : nil
        }
        _report = State(initialValue: drum30
            ? (CoreAIDecoderProbe.savedDrumReference30Report(decoderOnly: decoderOnly) ?? "Pemeriksaan drum 30 detik belum dijalankan.")
            : (matchingPrior ?? saved?.report ?? "Preset \(active.instrumentLabel): belum dijalankan."))
        _generatedBass = State(initialValue: saved?.bassURL)
        _generatedMix = State(initialValue: saved?.mixURL)
        _restoredPreview = State(initialValue: saved != nil)
    }

    var body: some Scene {
        WindowGroup {
            NavigationStack {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if CoreAIPreviewWindow.drumReference30DecoderOnlyEnabled {
                            Text("Pemeriksaan decoder 30 detik").font(.headline)
                            Text("Memeriksa decoder sendiri memakai acuan yang sudah diterima. Setelah selesai, bagikan laporannya.")
                            Button("Uji decoder 30 detik saja") {
                                audioPlayer?.stop()
                                runningModel = true
                                report = "Memverifikasi decoder dan input acuan…"
                                Task {
                                    let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
                                    report = await Task.detached(priority: .userInitiated) {
                                        await CoreAIDecoderProbe.runDrumReference30DecoderOnly(progress: update)
                                    }.value
                                    runningModel = false
                                }
                            }
                            .disabled(runningModel || workStartedInProcess)
                            Divider()
                        }
                        if CoreAIPreviewWindow.drumReference30ComponentsEnabled {
                            Text("Pemeriksaan drum 30 detik").font(.headline)
                            Text("Memeriksa komponen memakai acuan yang sudah diterima. Setelah selesai, bagikan laporannya.")
                            Button("Uji komponen drum 30 detik") {
                                audioPlayer?.stop()
                                runningModel = true
                                report = "Memverifikasi asset dan input drum 30 detik…"
                                Task {
                                    let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
                                    report = await Task.detached(priority: .userInitiated) {
                                        await CoreAIDecoderProbe.runDrumReference30Components(progress: update)
                                    }.value
                                    runningModel = false
                                }
                            }
                            .disabled(runningModel || workStartedInProcess)
                            Divider()
                        }
                        if hasResource(window.previewBundle + "/reference.json") {
                            Text("Buat \(window.instrumentLabel) untuk bagian awal rekaman piano Anda.")
                            if window.stemID.hasPrefix("guitar-") {
                                Text("Preview peran gitar; referensi warna suara Colab belum disertakan.")
                                    .font(.footnote)
                            }
                            if CoreAIPreviewWindow.trajectoryDiagnosticEnabled {
                                Text("Preview diagnostik. Hasilnya perlu diperiksa melalui laporan dan uji dengar.")
                                    .font(.footnote)
                            }
                            if window.isLong {
                                Text("Pemeriksaan terpisah: 16 panggilan sintetis tanpa piano/WAV. Jalankan pada proses app baru, sebelum impor rekaman.")
                                    .font(.footnote)
                                Button("Periksa aset sebelum generasi") { startNativePreflight() }
                                    .disabled(runningModel || workStartedInProcess || preflightAttempted)
                                if preflightAttempted {
                                    Text("Simpan laporan pemeriksaan. Hentikan app dari Xcode lalu buka lagi dari ikon sebelum Generate; bila gagal, jangan lanjutkan.")
                                        .font(.footnote)
                                }
                            }
                            Text("Pilih audio mono atau stereo minimal \(window.durationLabel) detik. File asli disimpan; generasi memakai \(window.durationLabel) detik awal.")
                                .font(.footnote)
                            Button("Pilih rekaman piano") { showingAudioImporter = true }
                                .disabled(runningModel)
                            if let importedPiano {
                                Text(importedPiano.metadata.name)
                                    .font(.callout)
                                Button("Dengarkan bagian piano") { playPreview(importedPiano.previewURL) }
                                    .disabled(runningModel)
                                Button("Generate \(window.instrumentLabel) dari rekaman") { startGeneration(importedPiano: importedPiano) }
                                    .disabled(runningModel || preflightAttempted)
                                if window.isLong {
                                    Text("Diagnosis singkat memakai input dan feedback aktual. Hanya tiga langkah pertama, tanpa WAV baru.")
                                        .font(.footnote)
                                    Button("Uji 3 langkah aktual (tanpa WAV)") { startSamplingDiagnostic(importedPiano) }
                                        .disabled(runningModel || preflightAttempted)
                                }
                            }
                            Divider()
                        }
                        if !CoreAIPreviewWindow.drumReference30ComponentsEnabled && !CoreAIPreviewWindow.drumReference30DecoderOnlyEnabled {
                            Text("Preset aktif: \(window.instrumentLabel)")
                                .font(.headline)
                            Text("Pemeriksaan dan baseline")
                                .font(.headline)
                        }
                        if hasResource("dit-interleaving-1024/reference.json") {
                            Text("Uji BF16 pada input tetap. Mulai dari 2 aset; ukuran lebih besar memerlukan bundle yang sesuai. Tidak menghasilkan audio.")
                                .font(.footnote)
                            ForEach(interleavingPoolSizes, id: \.self) { count in
                                ForEach([false,true], id: \.self) { reverse in
                                    Button("Uji \(count) aset BF16 — \(reverse ? "bergantian dahulu" : "berkelompok dahulu")") {
                                        audioPlayer?.stop()
                                        runningModel = true
                                        report = "Memeriksa bundle BF16…"
                                        Task {
                                            let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
                                            report = await Task.detached(priority: .userInitiated) {
                                                await CoreAIDecoderProbe.runInterleavingProbe(count: count, reverse: reverse, progress: update)
                                            }.value
                                            runningModel = false
                                        }
                                    }
                                    .disabled(runningModel)
                                }
                            }
                        }
                        if hasResource("dit-precision-1024/reference.json") {
                            Text("Uji komponen singkat untuk membandingkan biaya bobot BF16 dan FP32. Tidak menghasilkan audio baru.")
                                .font(.footnote)
                            Button("Uji biaya bobot BF16 vs FP32") {
                                audioPlayer?.stop()
                                runningModel = true
                                report = "Memeriksa aset dan referensi uji bobot…"
                                Task {
                                    let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
                                    report = await Task.detached(priority: .userInitiated) {
                                        await CoreAIDecoderProbe.runPrecisionProbe(progress: update)
                                    }.value
                                    runningModel = false
                                }
                            }
                            .disabled(runningModel)
                        }
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
                            Button("Uji conditioning \(window.instrumentLabel)") {
                                runningModel = true
                                report = "Menghitung embedding Qwen dan conditioning instrumen \(window.instrumentLabel)…"
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
                            Button("Dengarkan \(window.instrumentLabel)") { playPreview(generatedBass) }
                                .disabled(runningModel)
                            Button("Bandingkan \(window.instrumentLabel) dengan headroom") { playWithHeadroom(generatedBass) }
                                .disabled(runningModel)
                            ShareLink(item: generatedBass) { Label("Bagikan WAV \(window.instrumentLabel)", systemImage: "square.and.arrow.up") }
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
                            Button("Bandingkan mix dengan headroom") { playWithHeadroom(generatedMix) }
                                .disabled(runningModel)
                            Text("Pembanding headroom hanya menurunkan level salinan playback bila perlu. WAV asli dan mix tersimpan tetap.")
                                .font(.footnote)
                            ShareLink(item: generatedMix) { Label("Bagikan WAV piano + \(window.instrumentLabel)", systemImage: "square.and.arrow.up") }
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
                    if running { workStartedInProcess = true }
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
                    report = "Rekaman siap: \(imported.metadata.name)\nBagian awal \(window.durationLabel) detik, stereo 48 kHz. File asli disimpan tanpa diubah.\nDengarkan bagian piano, lalu tekan Generate \(window.instrumentLabel) dari rekaman."
                } catch {
                    report = "Impor gagal: \(error.localizedDescription)"
                }
                runningModel = false
            }
        }
    }

    private func startNativePreflight() {
        guard !runningModel, !workStartedInProcess, !preflightAttempted else { return }
        preflightAttempted = true
        workStartedInProcess = true
        audioPlayer?.stop()
        runningModel = true
        report = "Memeriksa aset terpisah; tidak menghasilkan WAV…"
        let selectedWindow = window
        Task {
            let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
            report = await Task.detached(priority: .userInitiated) {
                await CoreAIDecoderProbe.runNativeAssetPreflight(window: selectedWindow, progress: update)
            }.value
            runningModel = false
        }
    }

    private func startSamplingDiagnostic(_ piano: CoreAIImportedPiano) {
        guard !runningModel, !preflightAttempted, window.isLong else { return }
        audioPlayer?.stop()
        runningModel = true
        report = "Menyiapkan diagnosis tiga langkah aktual; audio lama tetap disimpan…"
        let selectedWindow = window
        Task {
            let update: @MainActor @Sendable (String) -> Void = { text in self.report = text }
            let result = await Task.detached(priority: .userInitiated) {
                await CoreAIDecoderProbe.generateBassPreview(window: selectedWindow, importedPiano: piano,
                    diagnosticThreeSteps: true, progress: update)
            }.value
            report = result.report
            runningModel = false
        }
    }

    private func startGeneration(importedPiano: CoreAIImportedPiano? = nil) {
        guard !runningModel, !preflightAttempted else { return }
        audioPlayer?.stop()
        generatedBass = nil; generatedMix = nil
        restoredPreview = false
        runningModel = true
        report = importedPiano == nil
            ? "Menjalankan uji baseline piano ke WAV \(window.instrumentLabel) pendek…"
            : "Membuat \(window.instrumentLabel) untuk rekaman pilihan Anda…"
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

    private var interleavingPoolSizes: [Int] {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/dit-interleaving-1024/reference.json")),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let count = manifest["pair_count"] as? Int else { return [] }
        return [2,4,8,12].filter { $0 <= count }
    }

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

    private func playWithHeadroom(_ url: URL) {
        guard !runningModel else { return }
        audioPlayer?.stop()
        runningModel = true
        let selectedWindow = window
        Task {
            do {
                let preview = try await Task.detached(priority: .userInitiated) {
                    try CoreAIAudioImport.headroomPreview(from: url, window: selectedWindow)
                }.value
                playPreview(preview.url)
                report += "\nPlayback headroom ONLY: source peak \(preview.sourcePeak); linear gain \(preview.gain); source SHA-256 \(preview.sourceSHA256). Target sample peak at most -3 dBFS; no limiter/EQ/model change or original-file modification."
            } catch { report += "\nPembanding headroom gagal: \(error)" }
            runningModel = false
        }
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
