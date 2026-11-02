// Bounded on-device file decoding. No model, server or FFmpeg is used here.
import AVFoundation
import CryptoKit
import Foundation

struct CoreAIImportedPiano: Sendable {
    struct Metadata: Codable, Sendable {
        let schemaVersion: Int
        let importID: String
        let name: String
        let originalSHA256: String
        let originalSampleRate: Double
        let originalChannels: UInt32
        let originalDurationSeconds: Double
        let segmentStartSeconds: Double
        let sampleRate: Int
        let channels: Int
        let frames: Int
        let seconds: Double
        let planarPCMSHA256: String
        let previewWAVSHA256: String
        let monoDuplicatedToStereo: Bool
        let normalizationApplied: Bool
    }

    let metadata: Metadata
    let originalURL: URL
    let previewURL: URL
    let planarPCM: [Float]

    func validatePCM() throws {
        guard metadata.sampleRate == 48000, metadata.channels == 2,
              metadata.frames == 245760, metadata.seconds == 5.12,
              metadata.segmentStartSeconds == 0, !metadata.normalizationApplied,
              planarPCM.count == 2 * 245760, planarPCM.allSatisfy({ $0.isFinite }),
              CoreAIAudioImport.pcmSHA256(planarPCM) == metadata.planarPCMSHA256 else {
            throw CoreAIAudioImport.ImportError.invalidPCM
        }
    }
}

enum CoreAIAudioImport {
    static let frames = 245760
    static let sampleRate = 48000.0

    enum ImportError: LocalizedError {
        case invalidFile, unsupportedChannels, invalidFormat, tooShort, invalidPCM, conversionFailed
        var errorDescription: String? {
            switch self {
            case .invalidFile: return "File audio tidak dapat dibaca. Pilih file audio dari Files."
            case .unsupportedChannels: return "Tahap ini mendukung rekaman mono atau stereo."
            case .invalidFormat: return "Format audio ini belum dapat didekode. Coba WAV, M4A, MP3, atau AIFF."
            case .tooShort: return "Rekaman harus menyediakan minimal 5,12 detik audio."
            case .invalidPCM: return "Data audio tidak valid atau mengandung nilai nonfinite."
            case .conversionFailed: return "Konversi audio ke 48 kHz tidak berhasil."
            }
        }
    }

    // Only immutable Swift values/URLs leave this synchronous helper. AVAudioFile,
    // converter and buffers stay on the calling worker, outside the UI actor.
    static func importPiano(from selectedURL: URL) throws -> CoreAIImportedPiano {
        guard selectedURL.isFileURL else { throw ImportError.invalidFile }
        let access = selectedURL.startAccessingSecurityScopedResource()
        defer { if access { selectedURL.stopAccessingSecurityScopedResource() } }
        let manager = FileManager.default
        guard let documents = manager.urls(for: .documentDirectory, in: .userDomainMask).first else { throw ImportError.invalidFile }
        let identifier = UUID().uuidString
        let folder = documents.appendingPathComponent("piano-import-" + identifier, isDirectory: true)
        try manager.createDirectory(at: folder, withIntermediateDirectories: true)
        do {
            let ext = selectedURL.pathExtension.lowercased()
            let safeExtension = ext.range(of: "^[a-z0-9]{1,12}$", options: .regularExpression) != nil ? ext : "audio"
            let original = folder.appendingPathComponent("original." + safeExtension)
            let copyFailure = ImportFailure()
            var coordinatorFailure: NSError?
            let coordinator = NSFileCoordinator(filePresenter: nil)
            coordinator.coordinate(readingItemAt: selectedURL, options: [], error: &coordinatorFailure) { readableURL in
                do { try FileManager.default.copyItem(at: readableURL, to: original) }
                catch { copyFailure.set(error) }
            }
            if let error = coordinatorFailure { throw error }
            if let error = copyFailure.get() { throw error }
            let file = try AVAudioFile(forReading: original, commonFormat: .pcmFormatFloat32, interleaved: false)
            let inputFormat = file.processingFormat
            guard inputFormat.sampleRate.isFinite, inputFormat.sampleRate > 0,
                  inputFormat.commonFormat == .pcmFormatFloat32, !inputFormat.isInterleaved else { throw ImportError.invalidFormat }
            guard inputFormat.channelCount == 1 || inputFormat.channelCount == 2 else { throw ImportError.unsupportedChannels }
            let duration = Double(file.length) / inputFormat.sampleRate
            guard duration >= Double(frames)/sampleRate else { throw ImportError.tooShort }
            let pcm = try decodeWindow(file: file)
            guard pcm.count == 2 * frames, pcm.allSatisfy({ $0.isFinite }) else { throw ImportError.invalidPCM }
            let preview = folder.appendingPathComponent("piano-preview.wav")
            try CoreAIDecoderProbe.writeStereoFloatWAV(pcm, frames: frames, to: preview)
            let metadata = CoreAIImportedPiano.Metadata(schemaVersion: 1, importID: identifier, name: selectedURL.lastPathComponent,
                originalSHA256: try CoreAIDecoderProbe.streamingSHA256(original),
                originalSampleRate: inputFormat.sampleRate, originalChannels: inputFormat.channelCount,
                originalDurationSeconds: duration, segmentStartSeconds: 0,
                sampleRate: 48000, channels: 2, frames: frames, seconds: 5.12,
                planarPCMSHA256: pcmSHA256(pcm), previewWAVSHA256: try CoreAIDecoderProbe.streamingSHA256(preview),
                monoDuplicatedToStereo: inputFormat.channelCount == 1, normalizationApplied: false)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(metadata).write(to: folder.appendingPathComponent("import.json"), options: .atomic)
            return CoreAIImportedPiano(metadata: metadata, originalURL: original, previewURL: preview, planarPCM: pcm)
        } catch {
            // Only our newly created failed import is removed, never the provider file.
            try? manager.removeItem(at: folder)
            throw error
        }
    }

    static func pcmSHA256(_ samples: [Float]) -> String {
        // iPhone arm64 uses little endian, matching the planar Float32 contract.
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func decodeWindow(file: AVAudioFile) throws -> [Float] {
        let format = file.processingFormat
        guard let input = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else { throw ImportError.invalidFormat }
        var left: [Float] = []; left.reserveCapacity(frames)
        var right: [Float] = []; right.reserveCapacity(frames)
        func append(_ buffer: AVAudioPCMBuffer) throws {
            guard let channels = buffer.floatChannelData else { throw ImportError.invalidPCM }
            let count = min(Int(buffer.frameLength), frames - left.count)
            guard count > 0 else { return }
            left.append(contentsOf: UnsafeBufferPointer(start: channels[0], count: count))
            right.append(contentsOf: UnsafeBufferPointer(start: channels[format.channelCount == 1 ? 0 : 1], count: count))
        }
        if format.sampleRate == sampleRate {
            while left.count < frames {
                try file.read(into: input, frameCount: AVAudioFrameCount(min(4096, frames - left.count)))
                guard input.frameLength > 0 else { throw ImportError.tooShort }
                try append(input)
            }
        } else {
            guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                                            channels: format.channelCount, interleaved: false),
                  let converter = AVAudioConverter(from: format, to: target),
                  let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: 4096) else { throw ImportError.invalidFormat }
            let feeder = AudioInputFeeder(file: file, buffer: input)
            var emptyRounds = 0
            while left.count < frames {
                var failure: NSError?
                output.frameLength = 0
                let status = converter.convert(to: output, error: &failure) { requested, inputStatus in
                    feeder.read(requested, status: inputStatus)
                }
                if let error = failure { throw error }
                if let error = feeder.failure.get() { throw error }
                guard status != .error else { throw ImportError.conversionFailed }
                try append(output)
                if left.count == frames { break }
                switch status {
                case .error: throw ImportError.conversionFailed
                case .endOfStream: throw ImportError.tooShort
                case .haveData, .inputRanDry:
                    emptyRounds = output.frameLength == 0 ? emptyRounds + 1 : 0
                    guard emptyRounds < 16 else { throw ImportError.conversionFailed }
                @unknown default: throw ImportError.conversionFailed
                }
            }
        }
        return left + right
    }
}

// AVAudioConverter synchronously consumes the feeder buffer before requesting the
// next one. The feeder is owned by one serial conversion; no AVFAudio object is
// transferred to another task. Locking protects callback state and error access.
private final class AudioInputFeeder: @unchecked Sendable {
    private let file: AVAudioFile
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var ended = false
    let failure = ImportFailure()
    init(file: AVAudioFile, buffer: AVAudioPCMBuffer) { self.file = file; self.buffer = buffer }
    func read(_ requested: AVAudioPacketCount, status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.lock(); defer { lock.unlock() }
        guard !ended else { status.pointee = .endOfStream; return nil }
        do {
            let count = min(requested, buffer.frameCapacity)
            guard count > 0 else { throw CoreAIAudioImport.ImportError.conversionFailed }
            try file.read(into: buffer, frameCount: count)
            guard buffer.frameLength > 0 else { ended = true; status.pointee = .endOfStream; return nil }
            status.pointee = .haveData
            return buffer
        } catch {
            ended = true; failure.set(error); status.pointee = .endOfStream; return nil
        }
    }
}

private final class ImportFailure: @unchecked Sendable {
    private let lock = NSLock()
    private var value: (any Error)?
    func set(_ error: any Error) { lock.lock(); defer { lock.unlock() }; value = error }
    func get() -> (any Error)? { lock.lock(); defer { lock.unlock() }; return value }
}
