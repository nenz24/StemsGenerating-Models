import Foundation

// Independent allowlist, not dimensions accepted from arbitrary manifest data.
enum CoreAIPreviewWindow: String, Sendable {
    case short512, long1024
    var weightStorageDtype: String { self == .short512 ? "float16" : "bfloat16" }
    var apgProjectionPrecision: String { "float32_validated_against_original_float64" }
    var latentFrames: Int { self == .short512 ? 128 : 256 }
    var samples: Int { latentFrames * 1920 }
    var seconds: Double { Double(samples) / 48000 }
    var hiddenTokens: Int { latentFrames / 2 }
    var textTokens: Int { self == .short512 ? 105 : 106 }
    var conditionTokens: Int { textTokens + 12 }
    var durationLabel: String { self == .short512 ? "5,12" : "10,24" }
    var sourceBundle: String { self == .short512 ? "source-context" : "source-context-1024" }
    var conditionBundle: String { self == .short512 ? "bass-conditioning" : "bass-conditioning-1024" }
    var previewBundle: String { self == .short512 ? "bass-preview-512" : "bass-preview-1024" }

    static var bundled: Self {
        guard let root = Bundle.main.resourceURL,
              let data = try? Data(contentsOf: root.appendingPathComponent("ModelCandidates/app-resource-manifest.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["selected_profile"] as? String == "bass-source-1024" else { return .short512 }
        return .long1024
    }
}
