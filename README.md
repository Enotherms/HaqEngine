# HaqEngine

Frame generation for the Steam Deck, written in Rust as a Vulkan layer. HaqEngine sits between a game and the screen, estimates motion between two real frames on the GPU, and shows one to three generated frames in between, so a game capped at 30 fps can be displayed at 60 or 90.

It is tuned for the Deck's GPU: the default Battery mode costs about 2 ms of GPU time per real frame at 1280×800 and needs no data from the game (no motion vectors, no depth).

## Status

Version 0.1, experimental. Tested on the Steam Deck OLED (RADV VANGOGH) in SteamOS gaming mode with Wuthering Waves, Monster Hunter Wilds and Arknights: Endfield through Proton. Other Vulkan games, including DirectX games running through DXVK or vkd3d-proton, should work but are untested.

## Install

On the Deck, in desktop mode or over SSH:

```sh
tar -xzf haqengine-0.1.0-x86_64-linux.tar.gz
sh haqengine-0.1.0-x86_64-linux/install.sh
```

This copies `libhaqengine.so` to `~/.local/lib` and writes the layer manifest to `~/.local/share/vulkan/implicit_layer.d/haqengine.json`. Nothing outside your home folder is touched. `uninstall.sh` removes both files.

The layer stays inactive until a game is started with `HAQENGINE=1`.

## Use

Set the game's Steam launch options:

```
ENABLE_GAMESCOPE_WSI=0 HAQENGINE=1 HAQENGINE_MULTIPLIER=2 %command%
```

`ENABLE_GAMESCOPE_WSI=0` is required: with Gamescope's WSI layer on, Gamescope shows only the newest image per refresh and drops the generated frames.

For smooth motion the game must hold a steady frame cap, and the display refresh rate must equal the cap times the multiplier:

| Game cap | Multiplier | Refresh rate |
|---|---|---|
| 30 | 2 | 60 Hz |
| 30 | 3 | 90 Hz |
| 25 | 2 | 50 Hz |
| 20 | 2 | 40 Hz |

If a game cannot hold its cap, lower the cap rather than raising the multiplier. Motion looks wavy when the refresh rate does not match: check the rate the display actually runs at, because the Quick Access slider can show a value that was not applied.

### Settings

| Variable | Values | Default |
|---|---|---|
| `HAQENGINE` | `1` turns the layer on | off |
| `DISABLE_HAQENGINE` | `1` turns it off even if `HAQENGINE` is set | |
| `HAQENGINE_MULTIPLIER` | `1` (pass-through) to `4` | `1` |
| `HAQENGINE_MODE` | `battery`, `quality` (experimental, based on AMD FSR 3 optical flow), `copy` (repeats frames, for testing) | `battery` |
| `HAQENGINE_PROFILE` | name of a `[game."…"]` section in the config file | matched by process name |
| `HAQENGINE_PACING` | `0` turns off the layer's present pacing | on |
| `HAQENGINE_LOG` | a file path, or `off` | `~/.local/state/haqengine/haqengine.log` |
| `HAQENGINE_TIMING` | `1` logs GPU time per phase | off |
| `HAQENGINE_MEASURE` | `1` measures frame rate without generating | off |

### Config file

`~/.config/haqengine/conf.toml` sets a default and per-game overrides. Environment variables win over the file. Changes are picked up while the game runs.

```toml
[default]
mode = "battery"
multiplier = 2

[game."Client-Win64-Shipping"]
multiplier = 3
```

A game section applies when its name is the value of `HAQENGINE_PROFILE`, or else when it starts with the game's process name.

### Log

Every 10 seconds the log shows, per swapchain, the real and output frame rates, the spread of real frame intervals, the added delay, HaqEngine's GPU time, and how often it fell back to a real frame (scene cuts, still screens). A large interval spread means the game itself is delivering frames unevenly.

## Known limits

- Fast-moving thin details (hair, cloth edges) can flicker briefly on quick camera turns where an object uncovers the background.
- Frames the game delivers late (shader compilation, loading) cannot be filled: generation needs the next real frame.
- Games with anti-cheat may object to any Vulkan layer. Use at your own risk.

## Build

Requirements: Rust 1.85 or newer, and `glslangValidator` (package `glslang`) on the `PATH` or in `GLSLANG_VALIDATOR`.

```sh
cargo test --workspace
sh scripts/package.sh
```

`package.sh` builds the layer for `x86_64-unknown-linux-gnu` and writes `dist/haqengine-<version>-x86_64-linux.tar.gz`. To build on macOS, install `cargo-zigbuild` and run:

```sh
BUILD="cargo zigbuild" TARGET=x86_64-unknown-linux-gnu.2.28 sh scripts/package.sh
```

### Layout

| Path | Contents |
|---|---|
| `crates/haqengine-core` | Platform-free logic: settings, config, pacing, statistics, and CPU reference versions of every GPU pass |
| `crates/haqengine-layer` | The Vulkan layer (`libhaqengine.so`): loader interface, swapchain handling, frame generation |
| `crates/haqengine-shaders` | GLSL compute shaders, compiled to SPIR-V at build time |
| `crates/haqengine-gpu-tests` | A headless test runner that checks each GPU pass against the CPU reference on a real GPU |
| `third_party/fidelityfx` | Unmodified AMD FidelityFX SDK shaders used by Quality mode |
| `scripts` | Packaging, install and uninstall scripts |

The GPU tests need a Vulkan device, so they run on the Deck: build `haqengine-gpu-tests`, copy it over and run it (`--filter <name>` selects cases).

## Licence

BSD 3-Clause, see `LICENSE`. The FidelityFX shaders in `third_party/fidelityfx` are MIT-licensed by AMD, see `THIRD_PARTY_NOTICES.txt`.
