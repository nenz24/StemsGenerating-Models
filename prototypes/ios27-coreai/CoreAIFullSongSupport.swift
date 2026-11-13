import Foundation
import CryptoKit

enum FullSongFailure: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let text) = self { return text }; return nil }
}

struct FullSongExecution: Decodable, Sendable {
    let schemaVersion: Int; let profile: String; let scope: String
    let jobBundle: String; let jobReferenceSha256: String
    let samplerBundle: String; let samplerReferenceSha256: String
    let decoderBundle: String; let decoderReferenceSha256: String
    let nativeImplementation: String; let sourceFrontend: String; let fullSongStrictParity: String; let baselineModified: Bool
}

struct FullSongJob: Decodable, Sendable {
    struct Window: Decodable, Sendable {
        let index: Int; let latentStart: Int; let latentEnd: Int
        let sampleStart48: Int; let sampleEnd48: Int; let rightPaddingSamples48: Int
        let sourceValidFrames: Int; let rightPaddingLatentFrames: Int; let overlapWithPreviousFrames: Int
    }
    struct Run: Decodable, Sendable {
        let instrument: String; let windowIndex: Int; let latentStart: Int; let latentEnd: Int
        let conditionSha256: String; let initialNoiseSha256: String
        let sourceContextSha256: String; let sourceAudioSha256: String; let sourceEpsilonSha256: String
    }
    let schemaVersion: Int; let profile: String; let status: String; let frames32: Int; let targetSamples48: Int
    let latentFrames: Int; let windowFrames: Int; let conditionTokens: Int; let instruments: [String]
    let windows: [Window]; let runs: [Run]; let presetSha256: [String:String]
    let artifactSha256: [String:String]; let tensorShapes: [String:[Int]]
    let samplerWeightStorageDtype: String; let computeDtype: String; let vaeDtype: String
    let absoluteTolerance: Double; let relativeTolerance: Double; let relativeRmseTolerance: Double
    let inferenceSteps: Int; let guidanceScale: Double; let apgMomentum: Double; let apgNormThreshold: Double; let apgProjectionDims: [Int]
    let baselineModified: Bool; let nativeExports: Int; let profileActivated: Bool; let readyForNativeExecution: Bool
    let sourceGainAlreadyApplied: Double; let additionalSourceGain: Double

    func validate() throws {
        let presets = ["bass":"73177b9d57ef961a66f3df375d24d2a869ad4f27fd6f007b6641d3d58c8614a1",
                       "drums":"bcfd73641fddbda1fb0975febfe28cccb3a3121ec48cec1d877933e64a6d70ca",
                       "guitar":"2f49a2db3ee92b2c52418794d38e4e29312dec5219c3f4ab4c7cf571c12df389"]
        guard schemaVersion == 1, profile == "full-song-prepared-inputs-v1", status == "prepared_inactive_inputs",
              frames32 > 0, frames32 < Int.max/3, targetSamples48 == (frames32*3+1)/2,
              targetSamples48 <= (Int(UInt32.max)-50)/8, latentFrames == (targetSamples48+1919)/1920,
              latentFrames >= 750, windowFrames == 750, conditionTokens == 127,
              instruments == ["bass","drums","guitar"], presetSha256 == presets,
              samplerWeightStorageDtype == "bfloat16", computeDtype == "float32", vaeDtype == "float32",
              absoluteTolerance == 1e-4, relativeTolerance == 1e-3, relativeRmseTolerance == 1e-3,
              inferenceSteps == 50, guidanceScale == 7, apgMomentum == -0.75, apgNormThreshold == 2.5, apgProjectionDims == [1],
              !baselineModified, nativeExports == 0, !profileActivated, !readyForNativeExecution,
              sourceGainAlreadyApplied == 0.7, additionalSourceGain == 1 else { throw FullSongFailure.invalid("Full-song geometry/settings differ") }
        let span = latentFrames-750, count = (span+549)/550+1
        guard windows.count == count, runs.count == count*3 else { throw FullSongFailure.invalid("Incomplete full-song schedule") }
        for (i,w) in windows.enumerated() {
            let start = count == 1 ? 0 : i*span/(count-1), end = start+750
            guard w.index == i, w.latentStart == start, w.latentEnd == end, w.sampleStart48 == start*1920,
                  w.sampleEnd48 == min(end*1920,targetSamples48), w.rightPaddingSamples48 == max(0,end*1920-targetSamples48),
                  w.sourceValidFrames == 750, w.rightPaddingLatentFrames == 0,
                  w.overlapWithPreviousFrames == (i == 0 ? 0 : windows[i-1].latentEnd-start) else { throw FullSongFailure.invalid("Invalid full-song window") }
        }
        var shapes = ["piano48.f32le":[1,2,latentFrames*1920],"source-context.f32le":[1,latentFrames,128],
                      "source-epsilon.f32le":[1,64,latentFrames],"timesteps.f32le":[51]]
        for role in instruments {
            shapes[role+"/noise-global.f32le"] = [1,latentFrames,64]
            shapes[role+"/condition.f32le"] = [1,127,2048]
            shapes[role+"/condition-mask.f32le"] = [1,127]
        }
        let extra: Set<String> = ["provenance/source-report.json","provenance/context-report.json","provenance/conditioning-report.json"]
        guard tensorShapes == shapes, Set(artifactSha256.keys) == Set(shapes.keys).union(extra),
              artifactSha256.values.allSatisfy(FullSongIO.isSHA) else { throw FullSongFailure.invalid("Full-song artifact set differs") }
        for (roleIndex,role) in instruments.enumerated() {
            for (i,w) in windows.enumerated() {
                let r = runs[roleIndex*count+i]
                guard r.instrument == role, r.windowIndex == i, r.latentStart == w.latentStart, r.latentEnd == w.latentEnd,
                      r.conditionSha256 == artifactSha256[role+"/condition.f32le"],
                      [r.conditionSha256,r.initialNoiseSha256,r.sourceContextSha256,r.sourceAudioSha256,r.sourceEpsilonSha256].allSatisfy(FullSongIO.isSHA)
                else { throw FullSongFailure.invalid("Full-song run input identity differs") }
            }
        }
    }
}

enum FullSongIO {
    static func isSHA(_ value: String) -> Bool { value.range(of:"^[a-f0-9]{64}$",options:.regularExpression) != nil }
    static func hash(_ values: [Float]) -> String {
        values.withUnsafeBufferPointer { SHA256.hash(data:Data(buffer:$0)).map { String(format:"%02x",$0) }.joined() }
    }
    static func regular(_ url: URL, bytes: Int? = nil) throws {
        let attributes = try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
        guard attributes.isRegularFile == true, attributes.isSymbolicLink != true,
              bytes == nil || attributes.fileSize == bytes else { throw FullSongFailure.invalid("Missing/linked/wrong-size file: "+url.lastPathComponent) }
    }
    static func read(_ url: URL, offset: Int, count: Int) throws -> [Float] {
        guard offset >= 0, count >= 0 else { throw FullSongFailure.invalid("Invalid Float32 range") }
        try regular(url)
        let file = try FileHandle(forReadingFrom:url); defer { try? file.close() }
        try file.seek(toOffset:UInt64(offset)*4)
        var data = Data(); data.reserveCapacity(count*4)
        while data.count < count*4 {
            guard let block = try file.read(upToCount:min(262144,count*4-data.count)), !block.isEmpty else { throw FullSongFailure.invalid("Incomplete Float32 range") }
            data.append(block)
        }
        let values: [Float] = data.withUnsafeBytes { bytes in
            (0..<count).map { Float(bitPattern:UInt32(littleEndian:bytes.loadUnaligned(fromByteOffset:$0*4,as:UInt32.self))) }
        }
        guard values.allSatisfy({ $0.isFinite }) else { throw FullSongFailure.invalid("Nonfinite full-song input") }
        return values
    }
    static func stereo(_ url: URL, plane: Int, start: Int, count: Int) throws -> [Float] {
        var value = try read(url,offset:start,count:count)
        value.append(contentsOf:try read(url,offset:plane+start,count:count))
        return value
    }
    static func appendFloats(_ values: [Float], to file: FileHandle) throws {
        guard values.allSatisfy({ $0.isFinite }) else { throw FullSongFailure.invalid("Nonfinite output") }
        try values.withUnsafeBufferPointer { try file.write(contentsOf:Data(buffer:$0)) }
    }
    static func preserveOrphan(_ url: URL) throws {
        if FileManager.default.fileExists(atPath:url.path) {
            try FileManager.default.moveItem(at:url,to:url.deletingLastPathComponent().appendingPathComponent(url.lastPathComponent+".preserved-"+UUID().uuidString))
        }
    }
}

final class FullSongOverlap {
    let job: FullSongJob
    var sum: [Double]; var weight: [Double]; var completed = Set<Int>()
    init(_ job: FullSongJob) {
        self.job = job
        sum = [Double](repeating:0,count:2*job.targetSamples48)
        weight = [Double](repeating:0,count:job.targetSamples48)
    }
    func add(_ values: [Float], index: Int) throws {
        guard index >= 0, index < job.windows.count, !completed.contains(index), values.count == 2880000,
              values.allSatisfy({ $0.isFinite }) else { throw FullSongFailure.invalid("Invalid/duplicate full-song window output") }
        let w = job.windows[index], n = w.sampleEnd48-w.sampleStart48
        let left = index == 0 ? 0 : min(w.sampleEnd48,job.windows[index-1].sampleEnd48)-w.sampleStart48
        let right = index+1 == job.windows.count ? 0 : w.sampleEnd48-max(w.sampleStart48,job.windows[index+1].sampleStart48)
        for j in 0..<n {
            var factor = 1.0
            if left > 0 && j < left { let s = sin(0.5*Double.pi*(Double(j)+0.5)/Double(left)); factor *= s*s }
            if right > 0 && j >= n-right { let c = cos(0.5*Double.pi*(Double(j-(n-right))+0.5)/Double(right)); factor *= c*c }
            guard factor.isFinite, factor > 0 else { throw FullSongFailure.invalid("Invalid overlap weight") }
            let position = w.sampleStart48+j
            weight[position] += factor
            sum[position] += Double(values[j])*factor
            sum[job.targetSamples48+position] += Double(values[1440000+j])*factor
        }
        completed.insert(index)
    }
    func finish(to url: URL) throws {
        guard completed.count == job.windows.count, weight.allSatisfy({ $0.isFinite && $0 > 0 }) else { throw FullSongFailure.invalid("Uncovered or missing full-song windows") }
        try FullSongIO.preserveOrphan(url)
        let temporary = url.appendingPathExtension("partial-"+UUID().uuidString)
        guard FileManager.default.createFile(atPath:temporary.path,contents:nil) else { throw FullSongFailure.invalid("Cannot create assembled audio") }
        let file = try FileHandle(forWritingTo:temporary)
        do {
            for c in 0..<2 {
                for a in stride(from:0,to:job.targetSamples48,by:65536) {
                    let b = min(a+65536,job.targetSamples48)
                    try FullSongIO.appendFloats((a..<b).map { Float(sum[c*job.targetSamples48+$0]/weight[$0]) },to:file)
                }
            }
            try file.close()
            try FullSongIO.regular(temporary,bytes:job.targetSamples48*8)
            try FileManager.default.moveItem(at:temporary,to:url)
        } catch { try? file.close(); throw error }
    }
}

enum FullSongListening {
    static func header(frames: Int) -> Data {
        var d = Data()
        func number<T:FixedWidthInteger>(_ value: T) { var v = value.littleEndian; withUnsafeBytes(of:&v) { d.append(contentsOf:$0) } }
        d.append(contentsOf:"RIFF".utf8); number(UInt32(50+frames*8)); d.append(contentsOf:"WAVEfmt ".utf8)
        number(UInt32(18)); number(UInt16(3)); number(UInt16(2)); number(UInt32(48000)); number(UInt32(384000)); number(UInt16(8)); number(UInt16(32)); number(UInt16(0))
        d.append(contentsOf:"fact".utf8); number(UInt32(4)); number(UInt32(frames)); d.append(contentsOf:"data".utf8); number(UInt32(frames*8))
        return d
    }
    static func write(job: FullSongJob, input: URL, output: URL, progress: (String) -> Void) throws -> Float {
        let names = ["piano","bass","drums","guitar"], frames = job.targetSamples48
        func block(_ a: Int,_ n: Int) throws -> [[Float]] {
            try names.map { role in
                try FullSongIO.stereo(role == "piano" ? input.appendingPathComponent("piano48.f32le") : output.appendingPathComponent("raw-"+role+".f32le"),
                    plane:role == "piano" ? job.latentFrames*1920 : frames,start:a,count:n)
            }
        }
        var peak: Float = 0
        for a in stride(from:0,to:frames,by:65536) {
            let n = min(65536,frames-a), parts = try block(a,n)
            for j in 0..<(2*n) {
                for part in parts { peak = max(peak,abs(part[j])) }
                peak = max(peak,abs((parts[0][j]+parts[1][j]+parts[2][j]+parts[3][j])*0.25))
            }
        }
        let gain = min(Float(0.36943485303214846),Float(pow(10.0,-3.0/20.0))/max(peak,1e-12))
        let outputs = names+["piano-with-band"], token = UUID().uuidString
        var handles: [FileHandle] = [], temporary: [URL] = []
        defer { for handle in handles { try? handle.close() } }
        for name in outputs {
            let path = output.appendingPathComponent(name+".partial-"+token+".wav")
            guard FileManager.default.createFile(atPath:path.path,contents:header(frames:frames)) else { throw FullSongFailure.invalid("Cannot create full-song WAV") }
            let handle = try FileHandle(forWritingTo:path); try handle.seekToEnd(); handles.append(handle); temporary.append(path)
        }
        for a in stride(from:0,to:frames,by:65536) {
            let n = min(65536,frames-a), parts = try block(a,n)
            for k in 0..<outputs.count {
                var interleaved = [Float](); interleaved.reserveCapacity(2*n)
                for j in 0..<n {
                    for c in 0..<2 {
                        let position = c*n+j
                        let value = k < 4 ? parts[k][position] : (parts[0][position]+parts[1][position]+parts[2][position]+parts[3][position])*0.25
                        interleaved.append(value*gain)
                    }
                }
                try FullSongIO.appendFloats(interleaved,to:handles[k])
            }
            if a % (65536*32) == 0 { progress("Writing full-song WAV samples \(a)/\(frames)") }
        }
        for k in outputs.indices {
            try handles[k].close()
            let target = output.appendingPathComponent(outputs[k]+".wav")
            try FullSongIO.preserveOrphan(target)
            try FullSongIO.regular(temporary[k],bytes:58+frames*8)
            try FileManager.default.moveItem(at:temporary[k],to:target)
        }
        return gain
    }
}
