#!/bin/sh
# Builds the layer and packs a release archive into dist/.
# On Linux: sh scripts/package.sh
# Elsewhere, with cargo-zigbuild: BUILD="cargo zigbuild" TARGET=x86_64-unknown-linux-gnu.2.28 sh scripts/package.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${TARGET:-x86_64-unknown-linux-gnu}"
BUILD="${BUILD:-cargo build}"
(cd "$ROOT" && $BUILD --release --locked --target "$TARGET" -p haqengine-layer)
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -n 1)"
name="haqengine-$version-x86_64-linux"
stage="$ROOT/dist/$name"
rm -rf "$stage" "$stage.tar.gz"
mkdir -p "$stage"
cp "$ROOT/target/${TARGET%%.*}/release/libhaqengine.so" "$stage/"
cp "$ROOT/scripts/haqengine.json.in" "$ROOT/scripts/install.sh" "$ROOT/scripts/uninstall.sh" "$stage/"
cp "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.txt" "$stage/"
cp "$ROOT/third_party/fidelityfx/LICENSE.txt" "$stage/LICENSE-FidelityFX.txt"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$ROOT/dist" -czf "$stage.tar.gz" "$name"
echo "$stage.tar.gz"
