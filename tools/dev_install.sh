#!/bin/sh
# Development helper: installs build/ into a game folder (default: the author's Steam library).
# Installs the proxy next to the game's FFX loader; the original is kept as *_orig.dll.
set -e
GAME="${1:-/run/media/anatolij/Data/SteamLibrary/steamapps/common/Cyberpunk 2077}/bin/x64"
SRC="$(cd "$(dirname "$0")/.." && pwd)/build/amd_fidelityfx_loader_dx12.dll"
cd "$GAME"
if [ ! -f amd_fidelityfx_loader_dx12_orig.dll ]; then
    mv amd_fidelityfx_loader_dx12.dll amd_fidelityfx_loader_dx12_orig.dll
fi
cp "$SRC" amd_fidelityfx_loader_dx12.dll
cp "$(dirname "$SRC")/../models/hairfix_resolve.bin" .
echo "installed into $GAME"
