// Shared helpers: logging, config, paths.
#pragma once

#include <windows.h>

#include <cstdint>
#include <string>

namespace hf {

constexpr float kAreaAll = 16.0f;

struct Config {
    bool enabled = true;
    bool dumpShaders = false;      // dump hair shaders / root signatures
    bool logFrames = false;        // periodic hair draw counts in the log
    std::string coverage = "raw";  // raw | sharpen
    std::string resolve = "strand";  // box | strand | neural
    int resolveMode = 1;           // 0 box, 1 strand, 2 strand + resolve network (read by the render thread)
    float strength = 0.65f;
    float area = 0.0f;             // the mod works only this far (render px) from strand edges; >= kAreaAll: all hair
    float areaFeather = 9.5f;      // width (render px) of the transition at the area's border
    float sharpenThreshold = 0.3f;
    float sharpenWidth = 2.0f;
    int coverageMode = 0;          // 0 raw, 1 sharpen (read by the render thread)
    bool usmEnabled = true;        // unsharp mask on hair after the upscale
    float usmAmount = 1.15f;
    float usmRadius = 2.99f;
    float usmThreshold = 0.01f;
    int toggleKey = VK_F10;
    int overlayKey = VK_HOME;      // opens the settings overlay
    bool stabilizeDepth = true;    // hand FSR stable hair depth / motion (can reduce flicker)
    int debugView = 0;             // 0 off, 1 coverage, 2 hair-surface mask, 3 strand coherence, 4 USM mask,
                                   // 5 network mask, 6 network correction, 7 mod area
    bool probe = false;            // log compute work of a few frames (finding the built-in FSR 2.1)
    int captureKey = 0;            // training capture key (0 = off), see capture.h
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
void FfxCreateBegin();                  // swap chains created inside ffxCreateContext are FFX's own
void FfxCreateEnd();
void HookFfxSwapChain(void* swapChain); // frame generation swap chain handed to the game
void OnFrameBoundary();                 // called once per upscaler dispatch

// banner / toggle
extern void* g_ffxDevice;               // ID3D12Device* handed to the FFX backend
bool IsActive();                        // effect currently switched on (F10)
void SetActive(bool on);
void ReloadRuntimeConfig();             // re-read the settings that can change while running

// settings overlay (ui.cpp)
bool OverlayVisible();
void* OverlayQueue();                   // the game's presenting queue (ID3D12CommandQueue*), or null
void PerfOnPresent();                   // periodic performance line in the log (neural.cpp)
void OverlayAttach(void* swapChain, void* queue, HWND hwnd);
void OverlayOnPresent(void* swapChain);
void OverlayBeforeResize();
bool OverlayRedirectWndProc(HWND hwnd, LONG_PTR newProc, LONG_PTR* previous);
bool OverlayGameWndProc(HWND hwnd, LONG_PTR* proc);
void AfterUpscale(void* commandList, void* output, unsigned outputState);
void DrawBanner(void* device, void* commandList, void* target, unsigned ffxState, const char* text);

}  // namespace hf
