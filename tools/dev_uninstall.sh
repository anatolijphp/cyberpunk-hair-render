#!/bin/sh
# Development helper.
# Restores the game's original FFX loader. hairfix.ini / hairfix.log / hairfix_dump are left in place.
set -e
GAME="${1:-/run/media/anatolij/Data/SteamLibrary/steamapps/common/Cyberpunk 2077}/bin/x64"
cd "$GAME"
if [ -f amd_fidelityfx_loader_dx12_orig.dll ]; then
    mv -f amd_fidelityfx_loader_dx12_orig.dll amd_fidelityfx_loader_dx12.dll
    echo "original FFX loader restored in $GAME"
else
    echo "nothing to do: no amd_fidelityfx_loader_dx12_orig.dll in $GAME"
fi
