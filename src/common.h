// Shared helpers: logging, config, paths.
#pragma once

#include <windows.h>

#include <cstdint>
#include <string>

namespace hf {

struct Config {
    bool enabled = true;
    bool dumpShaders = false;      // dump hair shaders / root signatures
    bool logFrames = false;        // periodic hair draw counts in the log
    std::string coverage = "raw";  // raw | sharpen
    std::string resolve = "strand";  // box | strand
    int resolveMode = 1;           // 0 box, 1 strand (read by the render thread)
    float strength = 0.5f;
    float sharpenThreshold = 0.3f;
    float sharpenWidth = 2.0f;
    int coverageMode = 0;          // 0 raw, 1 sharpen (read by the render thread)
    bool usmEnabled = true;        // unsharp mask on hair after the upscale
    float usmAmount = 1.0f;
    float usmRadius = 2.5f;
    float usmThreshold = 0.05f;
    int toggleKey = VK_F10;
    int overlayKey = VK_F9;        // opens the settings overlay
    bool stabilizeDepth = false;   // hand FSR stable hair depth / motion (can reduce flicker)
    int debugView = 0;             // 0 off, 1 coverage, 2 hair-surface mask, 3 strand coherence, 4 USM mask
};

extern Config g_config;
extern HMODULE g_self;

std::wstring ModuleDir();               // directory of this DLL, with trailing backslash
void LoadConfig();                      // reads (and creates if missing) hairfix.ini
void Log(const char* fmt, ...);
uint64_t Hash64(const void* data, size_t size);
void WriteFile(const std::wstring& path, const void* data, size_t size);

bool InstallD3D12Hooks();               // patches the game's D3D12CreateDevice import
void HookDeviceFromFfx(void* device);   // device handed to the FFX backend (fallback)
void OnFrameBoundary();                 // called once per upscaler dispatch

// banner / toggle
extern void* g_ffxDevice;               // ID3D12Device* handed to the FFX backend
bool IsActive();                        // effect currently switched on (F10)
void SetActive(bool on);
void ReloadRuntimeConfig();             // re-read the settings that can change while running

// settings overlay (ui.cpp)
bool OverlayVisible();
void OverlayAttach(void* swapChain, void* queue, HWND hwnd);
void OverlayOnPresent(void* swapChain);
void OverlayBeforeResize();
bool OverlayRedirectWndProc(HWND hwnd, LONG_PTR newProc, LONG_PTR* previous);
bool OverlayGameWndProc(HWND hwnd, LONG_PTR* proc);
void AfterUpscale(void* commandList, void* output, unsigned outputState);
void DrawBanner(void* device, void* commandList, void* target, unsigned ffxState, const char* text);

}  // namespace hf
