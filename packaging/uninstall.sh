#!/bin/sh
# Restores the game's original FidelityFX loader. Usage: ./uninstall.sh "<Cyberpunk 2077 folder>"
set -e
GAME="${1:?usage: $0 <Cyberpunk 2077 folder>}"
BIN="$GAME/bin/x64"
if [ -f "$BIN/amd_fidelityfx_loader_dx12_orig.dll" ]; then
    mv -f "$BIN/amd_fidelityfx_loader_dx12_orig.dll" "$BIN/amd_fidelityfx_loader_dx12.dll"
    echo "original loader restored; hairfix.ini / hairfix.log were left in $BIN"
else
    echo "nothing to do: no amd_fidelityfx_loader_dx12_orig.dll in $BIN"
fi
