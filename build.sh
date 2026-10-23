#!/bin/sh
# Cross-compiles the proxy amd_fidelityfx_loader_dx12.dll with llvm-mingw.
set -e
cd "$(dirname "$0")"
TC="${LLVM_MINGW:-$HOME/.local/opt/llvm-mingw-20261020-ucrt-ubuntu-22.04-x86_64}"
mkdir -p build/gen
DXC="${DXC:-$HOME/.local/opt/dxc-1.9.2609}"
python3 tools/gen_shaders.py "$DXC" "$(pwd)"
"$TC/bin/x86_64-w64-mingw32-clang++" -std=c++17 -O2 -Wall -Wno-unused-function \
    -shared -static -static-libgcc -static-libstdc++ \
    -o build/amd_fidelityfx_loader_dx12.dll \
    src/proxy.cpp src/hooks.cpp src/pso.cpp src/overlay.cpp src/hair.cpp src/usm.cpp src/ui.cpp src/imgui_build.cpp src/common.cpp \
    third_party/imgui/imgui.cpp third_party/imgui/imgui_draw.cpp third_party/imgui/imgui_tables.cpp \
    third_party/imgui/imgui_widgets.cpp third_party/imgui/backends/imgui_impl_win32.cpp \
    -Ithird_party/imgui \
    -luuid -ldwmapi -lgdi32 -limm32 -Wl,--exclude-all-symbols
"$TC/bin/llvm-objdump" -p build/amd_fidelityfx_loader_dx12.dll | grep -E '^\s+\[ *[0-9]+\]' || true
