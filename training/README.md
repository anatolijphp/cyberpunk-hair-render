# Training the hair resolve network

These tools collect training data in the game and train the network behind
`[Hair] Resolve = neural` (`hairfix_resolve.bin`). You don't need them to use the mod: the release
ships a trained model. They're for retraining it, for example on more characters, hairstyles or
lighting. Run the commands below from the repository root.

## What the network learns

Before FSR, the mod blends the game's dithered hair into a smooth image (the strand resolve). The
network refines that image at render resolution, before FSR. It sees 11 planes per pixel:

- the mod's resolve, tone-mapped
- the hair coverage and the hair-surface mask
- the local line direction of the resolve: a structure tensor, so it works along strands
- the game's raw, dithered frame

It predicts a colour correction, applied only on hair. The target is the same view rendered at
Native AA, sampled at each render pixel's centre plus that frame's FSR jitter. That's what the game
would render without the dither, and FSR then rebuilds detail from these sharp jittered frames.

The shipped model: 12 channels, 53,571 weights, trained on 32 views of one character
(4 photo mode scenes × 8 rotations) at 853×480 → 2560×1440 (FSR 3 Ultra Performance). On the
held-out scene it scores 23.7 dB hair PSNR against Native AA. The plain strand resolve scores 22.5 dB.
It was trained on captures from v0.3.0, which stored the hair coverage in 8 bits. Captures from
v0.3.1 on hold 16-bit coverage. Old captures can't be converted, but `prepare.py` and `train.py`
read both, so old and new scenes can be trained together.

## Requirements

- [Cyber Engine Tweaks](https://github.com/maximegmd/CyberEngineTweaks) (photo mode automation).
- Python 3.12 with `numpy`, `pillow` and `torch`. On Windows with an AMD or Intel GPU add
  `torch-directml`; otherwise training runs on CUDA if your torch build has it, else on the CPU (slow).
- Disk space: a low-quality capture is ~150 MB, a Native AA one ~550 MB, so one scene of 8 views
  in both qualities is ~5.5 GB. Prepared scenes are ~40 MB each.
- For the optional engine check: the build toolchain from the main README.

## 1. Capture

1. In `bin/x64/hairfix.ini` set `[Debug] CaptureKey = 0x77` (F8) and restart the game.
2. Copy `training/cet/hairfix_capture` to
   `bin/x64/plugins/cyber_engine_tweaks/mods/hairfix_capture`.
3. Pick a scene with your character's hair clearly visible against a calm background. Set FSR 3 to
   **Ultra Performance** (or the quality you want to train for).
4. Enter photo mode and press **F8**. The script:
   - sets the camera composition to preset 3;
   - turns the character through 0°, 45° … 315°;
   - waits 10 s after each turn for the hair to settle;
   - has the mod record 16 warm-up and 64 recorded frames with the fix on, then the same with it off,
     then game frames only, until every jitter of the first 64 frames was seen in every phase of the
     game's hair dither (for averaged targets, below; a few seconds);
   - turns the character back.

   Every view goes to `bin/x64/hairfix_capture/<date>_<time>_pm<set>_r<rotation>_w<render width>/`.
   Progress is written to `mods/hairfix_capture/progress.txt`. Don't touch anything until it says
   the sweep is done (~3 minutes).
5. Leave photo mode **without moving**, switch FSR to **Native AA** in the graphics settings,
   enter photo mode again and press **F8** once more. The game only applies an FSR change from its
   settings menu, so this switch is manual.

The two sweeps together make 8 training pairs: same views, low quality vs Native AA. Repeat with
other scenes, characters, hairstyles and lighting. More variety helps more than more frames.

Outside photo mode, F8 makes a single capture in the current quality, without the CET script.

To check a capture:

```
python training/capture_preview.py "<game>/bin/x64/hairfix_capture/<capture>"
```

It writes `sheet.png` (game frame, the mod's resolve, FSR output with / without the mod) into the
capture folder. It also prints how much each frame differs from the mean: a still scene stays low
and flat. Frames where the hair still moved are dropped automatically later.

## 2. Prepare

```
python training/prepare.py "<game>/bin/x64/hairfix_capture" D:/hairfix_train/scenes
```

This pairs consecutive photo mode sweeps of different render widths by rotation. It skips pairs
that don't line up (the camera or character moved between the sweeps) and pairs already prepared.
For each pair it writes `<low capture>.npz`: raw frames, coverage, hair-surface mask, the mod's
resolve and the per-frame Native AA target, cropped to the hair.

### Experimental: targets without Native AA

The game's hair dither repeats every 5 frames, FSR's jitter every few dozen. Captures from this
version record every frame's number, and keep recording until each jitter of the training frames was
seen in all 5 dither phases. Averaging those frames per jitter gives that frame without the dither,
at the same resolution and jitter, from the low quality sweep alone (no Native AA switch, nothing to
line up):

```
python training/prepare.py "<game>/bin/x64/hairfix_capture" D:/hairfix_train/averaged averaged
```

It prints a self-check per capture: frames of the same jitter and dither phase should be (almost)
identical, different phases should differ. This target hasn't been compared with the Native AA one
on real captures yet. To do that on one view captured both ways:

```
python training/compare_targets.py "<game>/bin/x64/hairfix_capture" <low capture> <Native AA capture> sheet.png
```

## 3. Train

Close the game first: training needs the GPU and its memory.

```
set HAIRFIX_CHANNELS=12
python training/train.py D:/hairfix_train/run1 pm20521265 6000 D:/hairfix_train/scenes
```

- `run1`: output folder.
- `pm20521265`: validation pattern. Scenes whose name contains it are held out, so use one photo
  mode set id from your captures.
- `6000`: iterations.
- `HAIRFIX_CHANNELS` (12), `HAIRFIX_BATCH` (64) and `HAIRFIX_PATCH` (96) set the network width,
  the batch size and the patch size. 12 channels is the shipped model; 24 costs about three times
  the GPU time in the game for little gain.

The log shows the loss every 250 iterations. Every 5000 iterations and at the end it shows hair PSNR
against Native AA on the validation scenes for the raw frame, the mod's resolve and the network.
The network should beat the resolve. At the end `run1/net.pt` holds the model, and
`val_<scene>.png` shows baselines | network | Native AA. The shipped model took about 6 minutes on
an RX 9070 XT.

## 4. Export and use

```
python training/export.py D:/hairfix_train/run1/net.pt hairfix_resolve.bin
```

Copy `hairfix_resolve.bin` next to the mod in `bin/x64` and set `Resolve = neural` (or pick
"Strand + network" in the overlay). The log shows `neural resolve: 12 channels, 11 inputs …` on
first use. Debug views 5 (network mask) and 6 (network correction, grey = none) show where it
works. To ship it with a release, replace `models/hairfix_resolve.bin`.

## Optional: check the engine

The mod runs the network with its own compute shaders (`src/neural.cpp`, `shaders/nn_*.hlsl`).
To check that they match PyTorch, build the test harness after `./build.sh`:

```
clang++ -std=c++17 -O2 -static -o build/test/engine_test.exe training/engine_test.cpp src/neural.cpp -ld3d12 -ldxgi -luuid -lpsapi
python training/engine_test.py build/test/engine_test.exe D:/hairfix_train/run1/net.pt D:/hairfix_train/scenes/<scene>.npz D:/hairfix_train/engine
```

Expect a PSNR of the mod against PyTorch above 60 dB (16-bit maths; about 70 dB is normal), a
network change far larger than the difference, and no change outside the hair.

## Files

| File | What it does |
|---|---|
| `cet/hairfix_capture/init.lua` | CET script: photo mode sweeps, talks to the mod through files |
| `capture_preview.py` | reads a capture, preview sheet, still-frame check |
| `pairs.py` | pairs low / Native AA sweeps, alignment check |
| `prepare.py` | builds training scenes from the pairs (or, `averaged`, from single captures) |
| `compare_targets.py` | Native AA vs averaged targets on one view |
| `train.py` | the network, its inputs, the loss and the training loop |
| `export.py` | writes `hairfix_resolve.bin` (the format is in its docstring) |
| `engine_test.py`, `engine_test.cpp` | runs the mod's network engine without the game and compares |
