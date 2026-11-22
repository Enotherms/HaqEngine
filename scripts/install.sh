#!/bin/sh
# Installs HaqEngine for the current user. Run on the Steam Deck from the unpacked release folder.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
lib="$HOME/.local/lib"
layers="$HOME/.local/share/vulkan/implicit_layer.d"
mkdir -p "$lib" "$layers"
cp "$here/libhaqengine.so" "$lib/libhaqengine.so.new"
mv -f "$lib/libhaqengine.so.new" "$lib/libhaqengine.so"
sed "s|@LIBRARY_PATH@|$lib/libhaqengine.so|" "$here/haqengine.json.in" > "$layers/haqengine.json.new"
mv -f "$layers/haqengine.json.new" "$layers/haqengine.json"
echo "HaqEngine installed. Steam launch options: HAQENGINE=1 HAQENGINE_MULTIPLIER=2 %command%"
