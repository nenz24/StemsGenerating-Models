import Foundation

// Independent allowlist: publisher-tokenized preset lengths are pinned in host tests.
enum CoreAIPreviewWindow: String, Sendable {
    case short512, long1024, drums1024, guitarStrum1024, guitarPick1024
    var isLong: Bool { self != .short512 }
    var stemID: String {
        switch self { case .drums1024: return "drums"; case .guitarStrum1024: return "guitar-strum"; case .guitarPick1024: return "guitar-pick"; default: return "bass" }
    }
    var instrumentLabel: String {
        switch self { case .drums1024: return "drum"; case .guitarStrum1024: return "gitar strumming"; case .guitarPick1024: return "gitar picking"; default: return "bass" }
    }
    var instruction: String { "Generate the \(stemID.hasPrefix("guitar-") ? "guitar" : stemID) track based on the audio context:" }
    var presetSHA256: String {
        switch self {
        case .drums1024: return "bcfd73641fddbda1fb0975febfe28cccb3a3121ec48cec1d877933e64a6d70ca"
        case .guitarStrum1024: return "885c989481abca270fcea2baa77cd843c73efcde4ad3e2f6d42ad9eeee05d104"
        case .guitarPick1024: return "7aaff6ec3d7fb3a8c439d4b6d9e2baca3edc65835e3b59e64891d9b6694303bb"
        default: return "4bacacb4f9fb1a9e66c98958869e2756878e6420910b2af065bea6522cb6ef63"
        }
    }
    var weightStorageDtype: String { isLong ? "bfloat16" : "float16" }
    var apgProjectionPrecision: String { "float32_validated_against_original_float64" }
    var latentFrames: Int { isLong ? 256 : 128 }
    var samples: Int { latentFrames * 1920 }
    var seconds: Double { Double(samples) / 48000 }
    var hiddenTokens: Int { latentFrames / 2 }
    var textTokens: Int { self == .short512 ? 105 : ((self == .drums1024 || self == .guitarPick1024) ? 111 : 106) }
    var conditionTokens: Int { textTokens + 12 }
    var durationLabel: String { isLong ? "10,24" : "5,12" }
    var sourceBundle: String { isLong ? "source-context-1024" : "source-context" }
    var conditionBundle: String { isLong ? stemID + "-conditioning-1024" : "bass-conditioning" }
    var previewBundle: String { stemID + (isLong ? "-preview-1024" : "-preview-512") }

    static var trajectoryDiagnosticEnabled: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["diagnostic_trajectory"] as? Bool == true
    }

    static var drumReference30ComponentsEnabled: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["selected_profile"] as? String == "drums-reference30-components"
    }

    static var drumReference30DecoderOnlyEnabled: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["selected_profile"] as? String == "drums-reference30-decoder"
    }

    static var drumReference30WindowEnabled: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["selected_profile"] as? String == "drums-reference30-window-decoder"
    }

    static var drumReference30GenerationEnabled: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return object["selected_profile"] as? String == "drums-reference30-generation"
    }

    static var sourceWindowEncoderEnabled: Bool {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String:Any] else { return false }
        return object["selected_profile"] as? String == "source-reference30-window-encoder"
    }

    static var bundled: Self {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let profile = object["selected_profile"] as? String else { return .short512 }
        switch profile {
        case "bass-source-1024": return .long1024
        case "drums-source-1024": return .drums1024
        case "guitar-strum-source-1024": return .guitarStrum1024
        case "guitar-pick-source-1024": return .guitarPick1024
        default: return .short512
        }
    }
}
