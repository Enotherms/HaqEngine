#!/bin/sh
# Removes HaqEngine from the Steam Deck: the layer manifest and the library, nothing else.
# Run on the Deck. The log in ~/.local/state/haqengine is kept; delete it by hand if wanted.
set -eu
rm -f "$HOME/.local/share/vulkan/implicit_layer.d/haqengine.json"
rm -f "$HOME/.local/lib/libhaqengine.so"
echo "HaqEngine removed."
