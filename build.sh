#!/usr/bin/env bash
#
# Builds Boids3D.app from the command line — no Xcode project needed.
# Requires the Xcode Command Line Tools:  xcode-select --install
#
# Usage:
#   ./build.sh           build for the host architecture
#   ./build.sh universal build a universal (arm64 + x86_64) binary
#   ./build.sh run       build, then launch
#   ./build.sh clean     remove the build directory
#
set -euo pipefail

APP="Boids3D"
BUNDLE_ID="com.makarov.boids3d"
DEPLOY="12.0"
BUILD="build"

# Sources are taken from the current directory; falls back to ./BoidsMetal.
if [ -f "Shaders.metal" ]; then
    SRC="."
elif [ -f "BoidsMetal/Shaders.metal" ]; then
    SRC="BoidsMetal"
else
    echo "error: Shaders.metal not found in '.' or './BoidsMetal'. Run this from the source directory." >&2
    exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
    echo "error: Xcode Command Line Tools not found. Run: xcode-select --install" >&2
    exit 1
fi

echo "==> Sources: $SRC"

case "${1:-build}" in
clean)
    rm -rf "$BUILD"; echo "cleaned."; exit 0;;
universal)
    TARGETS=(arm64-apple-macos$DEPLOY x86_64-apple-macos$DEPLOY); RUN=0;;
run)
    TARGETS=("$(uname -m)-apple-macos$DEPLOY"); RUN=1;;
*)
    TARGETS=("$(uname -m)-apple-macos$DEPLOY"); RUN=0;;
esac

APPDIR="$BUILD/$APP.app"
MACOS="$APPDIR/Contents/MacOS"
RES="$APPDIR/Contents/Resources"
rm -rf "$APPDIR"
mkdir -p "$MACOS" "$RES" "$BUILD"

echo "==> Compiling Metal shaders"
xcrun -sdk macosx metal -ffast-math -c "$SRC/Shaders.metal" -o "$BUILD/Shaders.air"
xcrun -sdk macosx metallib "$BUILD/Shaders.air" -o "$RES/default.metallib"

echo "==> Compiling Swift  [${TARGETS[*]}]"
TARGET_FLAGS=()
for t in "${TARGETS[@]}"; do TARGET_FLAGS+=(-target "$t"); done

# swiftc can't emit a single binary for two -target flags directly, so build
# one slice per arch and lipo them together when more than one is requested.
SLICES=()
for t in "${TARGETS[@]}"; do
    slice="$BUILD/$APP.${t%%-*}"
    xcrun -sdk macosx swiftc \
        -O -parse-as-library \
        -target "$t" \
        -framework Metal -framework MetalKit -framework SwiftUI -framework AppKit \
        "$SRC"/*.swift \
        -o "$slice"
    SLICES+=("$slice")
done

if [ "${#SLICES[@]}" -gt 1 ]; then
    xcrun lipo -create "${SLICES[@]}" -output "$MACOS/$APP"
else
    cp "${SLICES[0]}" "$MACOS/$APP"
fi

echo "==> Writing Info.plist"
cat > "$APPDIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>        <string>$APP</string>
    <key>CFBundleIdentifier</key>        <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>              <string>$APP</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key>           <string>1</string>
    <key>LSMinimumSystemVersion</key>    <string>$DEPLOY</string>
    <key>NSPrincipalClass</key>          <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>   <true/>
</dict>
</plist>
PLIST

echo "==> Ad-hoc code signing"
codesign --force --sign - "$APPDIR"

echo ""
echo "Built: $APPDIR"
echo "Run:   open $APPDIR"

if [ "${RUN:-0}" = "1" ]; then
    echo "==> Launching"
    open "$APPDIR"
fi
