#!/bin/sh
# Installs the hair dither fix. Usage: ./install.sh "/path/to/steamapps/common/Cyberpunk 2077"
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
GAME="${1:?usage: $0 <Cyberpunk 2077 folder>}"
BIN="$GAME/bin/x64"
[ -f "$BIN/Cyberpunk2077.exe" ] || { echo "not a Cyberpunk 2077 folder: $GAME" >&2; exit 1; }
if [ ! -f "$BIN/amd_fidelityfx_loader_dx12_orig.dll" ]; then
    mv "$BIN/amd_fidelityfx_loader_dx12.dll" "$BIN/amd_fidelityfx_loader_dx12_orig.dll"
fi
cp "$HERE/files/amd_fidelityfx_loader_dx12.dll" "$HERE/files/hairfix_resolve.bin" "$BIN/"
[ -f "$BIN/hairfix.ini" ] || cp "$HERE/files/hairfix.ini" "$BIN/"
echo "installed into $BIN (an existing hairfix.ini was kept)"
