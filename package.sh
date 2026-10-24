#!/bin/sh
# Builds the DLL and packages it:
#   dist/cp2077-hairfix-<version>-windows.zip     (install.bat / uninstall.bat)
#   dist/cp2077-hairfix-<version>-linux.tar.gz    (install.sh / uninstall.sh, for Proton)
# VERSION defaults to the current git tag / description, SKIP_BUILD=1 packages an existing build.
set -e
cd "$(dirname "$0")"
VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null | sed 's/^v//')}"
VERSION="${VERSION:-dev}"
[ "${SKIP_BUILD:-0}" = 1 ] || ./build.sh
rm -rf dist
for target in windows linux; do
    name="cp2077-hairfix-$VERSION-$target"
    out="dist/$name"
    mkdir -p "$out/files"
    cp build/amd_fidelityfx_loader_dx12.dll config/hairfix.ini models/hairfix_resolve.bin "$out/files/"
    cp README.md THIRD_PARTY_NOTICES.md "$out/"
    if [ "$target" = windows ]; then
        cp packaging/install.bat packaging/uninstall.bat "$out/"
        (cd dist && zip -qr "$name.zip" "$name")
    else
        cp packaging/install.sh packaging/uninstall.sh "$out/"
        chmod +x "$out/install.sh" "$out/uninstall.sh"
        (cd dist && tar czf "$name.tar.gz" "$name")
    fi
    rm -rf "$out"
done
ls -la dist
