#!/usr/bin/env bash
# Read public iPhone SDK interfaces and type-check one import; no model inference.
set -euo pipefail

for probe_tool in xcrun xcodebuild ditto; do
    if ! command -v "$probe_tool" >/dev/null 2>&1; then
        printf 'Run this probe on the Mac with Xcode installed; missing tool: %s\n' "$probe_tool" >&2
        exit 2
    fi
done

output_base="${1:-$HOME/Desktop}"
mkdir -p "$output_base"
report_dir="$(mktemp -d "$output_base/coreai-sdk-report.XXXXXX")"
probe_cache="$(mktemp -d "${TMPDIR:-/tmp}/coreai-module-cache.XXXXXX")"
ios_sdk_path="$(xcrun --sdk iphoneos --show-sdk-path)"
xcodebuild -version > "$report_dir/xcode-version.txt"
xcodebuild -showsdks > "$report_dir/available-sdks.txt"
printf '%s\n' "$ios_sdk_path" > "$report_dir/iphoneos-sdk-path.txt"
printf 'import CoreAI\n' > "$report_dir/ImportProbe.swift"

if xcrun --sdk iphoneos swiftc -sdk "$ios_sdk_path" \
    -target arm64-apple-ios27.0 -module-cache-path "$probe_cache" \
    -typecheck "$report_dir/ImportProbe.swift" > "$report_dir/import-log.txt" 2>&1; then
    printf 'CoreAI import: succeeded (type-check only)\n' > "$report_dir/import-status.txt"
else
    probe_exit=$?
    printf 'CoreAI import: failed; compiler exit %s; inspect import-log.txt\n' "$probe_exit" > "$report_dir/import-status.txt"
fi

# Ask the compiler to resolve dependencies instead of guessing additional SDK
# directories. Preserve diagnostics even when scanning fails after import works.
if xcrun --sdk iphoneos swiftc -scan-dependencies -module-name CoreAIProbe \
    -sdk "$ios_sdk_path" -target arm64-apple-ios27.0 \
    -module-cache-path "$probe_cache" "$report_dir/ImportProbe.swift" \
    > "$report_dir/dependencies.json" 2> "$report_dir/dependency-scan-log.txt"; then
    printf 'Dependency scan: succeeded; model support still unverified\n' > "$report_dir/dependency-scan-status.txt"
else
    scan_exit=$?
    printf 'Dependency scan: failed; compiler exit %s; inspect dependency-scan-log.txt\n' "$scan_exit" > "$report_dir/dependency-scan-status.txt"
fi

copy_public_interfaces() {
    local interface_root="$1" interface_label="$2" interface_file relative_file
    if [ ! -d "$interface_root" ]; then
        printf 'Not found: %s\n' "$interface_root" >> "$report_dir/interface-locations.txt"
        return
    fi
    printf 'Found: %s\n' "$interface_root" >> "$report_dir/interface-locations.txt"
    while IFS= read -r -d '' interface_file; do
        relative_file="${interface_file#"$interface_root"/}"
        mkdir -p "$report_dir/interfaces/$interface_label/$(dirname "$relative_file")"
        cp "$interface_file" "$report_dir/interfaces/$interface_label/$relative_file"
    done < <(find -H "$interface_root" -type f \( -name '*.swiftinterface' -o -name '*.h' -o -name '*.modulemap' \) ! -name '*.private.swiftinterface' ! -name '*.package.swiftinterface' -print0)
}

# These six modules and the SubFrameworks location are established by the
# user's compiler dependency report; their names alone do not prove capability.
for module_name in CoreAI CoreAIDelegates CoreAIRuntime CoreAICompiler CoreAICommon CoreAIAsset; do
    mkdir -p "$report_dir/interfaces/$module_name"
    for framework_location in Frameworks SubFrameworks; do
        framework_dir="$ios_sdk_path/System/Library/$framework_location/$module_name.framework"
        copy_public_interfaces "$framework_dir/Headers" "$module_name/$framework_location/headers"
        copy_public_interfaces "$framework_dir/Modules" "$module_name/$framework_location/modules"
    done
    copy_public_interfaces "$ios_sdk_path/usr/lib/swift/$module_name.swiftmodule" "$module_name/sdk-swift-overlay"
    source_count="$(find "$report_dir/interfaces/$module_name" -type f | wc -l | tr -d ' ')"
    printf '%s public source files: %s\n' "$module_name" "$source_count" >> "$report_dir/interface-counts.txt"
done
cat > "$report_dir/README.txt" <<'REPORT'
Target: arm64 iPhone / iOS 27.0. Public SDK interface and import probe only.
Import success does not establish custom audio model support or accelerator support.
No checkpoint was loaded; no device execution, conversion or benchmark was performed.
No credentials, device identifiers, full SDK or private framework interfaces are collected.
REPORT

ditto -c -k --keepParent "$report_dir" "$report_dir.zip"
cat "$report_dir/import-status.txt"
cat "$report_dir/dependency-scan-status.txt"
cat "$report_dir/interface-counts.txt"
printf 'Report ZIP: %s.zip\n' "$report_dir"
