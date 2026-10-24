#include "common.h"
#include "capture.h"
#include "../build/gen/default_ini.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace hf {

Config g_config;
HMODULE g_self = nullptr;
void* g_ffxDevice = nullptr;

namespace {
std::atomic<bool> g_active{true};
std::atomic<uint64_t> g_bannerUntil{0};
bool g_keyWasDown = false;
bool g_firstFrame = true;
}  // namespace

bool IsActive() { return g_active.load(std::memory_order_relaxed); }

void SetActive(bool on) {
    g_active = on;
    g_bannerUntil = GetTickCount64() + 2500;
    Log("hair fix %s", on ? "ON" : "OFF");
}

// Re-reads hairfix.ini when it changed on disk (checked at most twice a second).
void PollConfigFile(uint64_t now) {
    static uint64_t lastCheck = 0;
    static FILETIME lastWrite{};
    if (now - lastCheck < 500) return;
    lastCheck = now;
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExW((ModuleDir() + L"hairfix.ini").c_str(), GetFileExInfoStandard, &a)) return;
    bool first = lastWrite.dwLowDateTime == 0 && lastWrite.dwHighDateTime == 0;
    bool changed = CompareFileTime(&a.ftLastWriteTime, &lastWrite) != 0;
    lastWrite = a.ftLastWriteTime;
    if (!first && changed) {
        Log("hairfix.ini changed, reloading");
        ReloadRuntimeConfig();
    }
}

void AfterUpscale(void* commandList, void* output, unsigned outputState) {
    uint64_t now = GetTickCount64();
    PollConfigFile(now);
    if (g_firstFrame) {   // show the state once at startup so it is obvious the mod is loaded
        g_firstFrame = false;
        g_bannerUntil = now + 4000;
    }
    bool down = (GetAsyncKeyState(g_config.toggleKey) & 0x8000) != 0;
    if (down && !g_keyWasDown) {
        bool on = !g_active.load();
        ReloadRuntimeConfig();
        g_active = on;
        g_bannerUntil = now + 2500;
        Log("toggle: hair fix %s", on ? "ON" : "OFF");
    }
    g_keyWasDown = down;
    if (now < g_bannerUntil.load() || CaptureActive())   // the banner stays up while a capture runs
        DrawBanner(g_ffxDevice, commandList, output, outputState, IsActive() ? "HAIR FIX ON" : "HAIR FIX OFF");
}

namespace {

std::mutex g_logMutex;
FILE* g_logFile = nullptr;

std::string ReadIni(const std::wstring& path, const char* section, const char* key, const char* def) {
    char buf[256];
    std::string p(path.begin(), path.end());
    GetPrivateProfileStringA(section, key, def, buf, sizeof(buf), p.c_str());
    // strip inline comments and whitespace
    std::string s(buf);
    size_t c = s.find(';');
    if (c != std::string::npos) s.resize(c);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

bool ParseBool(const std::string& s, bool def) {
    if (s == "true" || s == "1" || s == "yes" || s == "on") return true;
    if (s == "false" || s == "0" || s == "no" || s == "off") return false;
    return def;
}

}  // namespace

std::wstring ModuleDir() {
    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(g_self, path, MAX_PATH);
    std::wstring dir(path, n);
    return dir.substr(0, dir.find_last_of(L"\\/") + 1);
}

void LoadConfig() {
    std::wstring ini = ModuleDir() + L"hairfix.ini";
    if (GetFileAttributesW(ini.c_str()) == INVALID_FILE_ATTRIBUTES)
        WriteFile(ini, kDefaultIni, sizeof(kDefaultIni) - 1);

    Config c;
    c.enabled = ParseBool(ReadIni(ini, "Hair", "Enabled", "true"), true);
    c.coverage = ReadIni(ini, "Hair", "Coverage", "raw");
    c.coverageMode = c.coverage == "sharpen" ? 1 : 0;
    c.resolve = ReadIni(ini, "Hair", "Resolve", "strand");
    c.resolveMode = c.resolve == "box" ? 0 : c.resolve == "neural" ? 2 : 1;
    c.strength = strtof(ReadIni(ini, "Hair", "Strength", "0.65").c_str(), nullptr);
    c.area = strtof(ReadIni(ini, "Hair", "Area", "0").c_str(), nullptr);
    c.areaFeather = strtof(ReadIni(ini, "Hair", "AreaFeather", "9.5").c_str(), nullptr);
    c.sharpenThreshold = strtof(ReadIni(ini, "Hair", "SharpenThreshold", "0.3").c_str(), nullptr);
    c.sharpenWidth = strtof(ReadIni(ini, "Hair", "SharpenWidth", "2.0").c_str(), nullptr);
    c.usmEnabled = ParseBool(ReadIni(ini, "Hair", "UnsharpMask", "true"), true);
    c.usmAmount = strtof(ReadIni(ini, "Hair", "USMAmount", "1.15").c_str(), nullptr);
    c.usmRadius = strtof(ReadIni(ini, "Hair", "USMRadius", "2.99").c_str(), nullptr);
    c.usmThreshold = strtof(ReadIni(ini, "Hair", "USMThreshold", "0.01").c_str(), nullptr);
    c.stabilizeDepth = ParseBool(ReadIni(ini, "Hair", "StabilizeDepth", "true"), true);
    c.mirror = ParseBool(ReadIni(ini, "Hair", "Mirror", "true"), true);
    ViewSettings& m = c.mirrorView;
    const std::string mr = ReadIni(ini, "Mirror", "Resolve", "strand");
    m.resolveMode = mr == "box" ? 0 : mr == "neural" ? 2 : 1;
    m.strength = strtof(ReadIni(ini, "Mirror", "Strength", "0.65").c_str(), nullptr);
    m.area = strtof(ReadIni(ini, "Mirror", "Area", "0").c_str(), nullptr);
    m.areaFeather = strtof(ReadIni(ini, "Mirror", "AreaFeather", "12.6").c_str(), nullptr);
    m.usmEnabled = ParseBool(ReadIni(ini, "Mirror", "UnsharpMask", "true"), true);
    m.usmAmount = strtof(ReadIni(ini, "Mirror", "USMAmount", "1.65").c_str(), nullptr);
    m.usmRadius = strtof(ReadIni(ini, "Mirror", "USMRadius", "2.52").c_str(), nullptr);
    m.usmThreshold = strtof(ReadIni(ini, "Mirror", "USMThreshold", "0").c_str(), nullptr);
    c.toggleKey = (int)strtol(ReadIni(ini, "Hair", "ToggleKey", "0x79").c_str(), nullptr, 0);
    c.overlayKey = (int)strtol(ReadIni(ini, "Hair", "OverlayKey", "0x24").c_str(), nullptr, 0);
    c.debugView = (int)strtol(ReadIni(ini, "Debug", "View", "0").c_str(), nullptr, 0);
    c.dumpShaders = ParseBool(ReadIni(ini, "Debug", "DumpShaders", "false"), false);
    c.logFrames = ParseBool(ReadIni(ini, "Debug", "LogFrames", "false"), false);
    c.probe = ParseBool(ReadIni(ini, "Debug", "ProbeUpscaler", "false"), false);
    c.captureKey = (int)strtol(ReadIni(ini, "Debug", "CaptureKey", "0").c_str(), nullptr, 0);
    g_config = c;
    Log("config: enabled=%d coverage=%s resolve=%s strength=%.2f sharpen=%.2f/%.2f key=0x%x dump=%d view=%d stabilize=%d usm=%d (%.2f/%.2f/%.3f) probe=%d",
        c.enabled, c.coverage.c_str(), c.resolve.c_str(), c.strength, c.sharpenThreshold,
        c.sharpenWidth, c.toggleKey, c.dumpShaders, c.debugView, c.stabilizeDepth, c.usmEnabled, c.usmAmount, c.usmRadius,
        c.usmThreshold, c.probe);
}

void Log(const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    if (!g_logFile) {
        g_logFile = _wfopen((ModuleDir() + L"hairfix.log").c_str(), L"w");
        if (!g_logFile) return;
    }
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(g_logFile, "%02d:%02d:%02d.%03d [%5lu] ", t.wHour, t.wMinute, t.wSecond,
            t.wMilliseconds, GetCurrentThreadId());
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_logFile, fmt, ap);
    va_end(ap);
    fputc('\n', g_logFile);
    fflush(g_logFile);
}

uint64_t Hash64(const void* data, size_t size) {
    // FNV-1a
    uint64_t h = 1469598103934665603ull;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

void WriteFile(const std::wstring& path, const void* data, size_t size) {
    FILE* f = _wfopen(path.c_str(), L"wb");
    if (!f) return;
    fwrite(data, 1, size, f);
    fclose(f);
}

// Re-reads hairfix.ini (on F10) and updates only the numeric settings the render thread uses.
void ReloadRuntimeConfig() {
    Config old = g_config;
    LoadConfig();
    Config fresh = g_config;
    g_config = old;
    g_config.resolveMode = fresh.resolveMode;
    g_config.strength = fresh.strength;
    g_config.area = fresh.area;
    g_config.areaFeather = fresh.areaFeather;
    g_config.debugView = fresh.debugView;
    g_config.stabilizeDepth = fresh.stabilizeDepth;
    g_config.mirror = fresh.mirror;
    g_config.mirrorView = fresh.mirrorView;
    g_config.coverageMode = fresh.coverageMode;
    g_config.usmEnabled = fresh.usmEnabled;
    g_config.usmAmount = fresh.usmAmount;
    g_config.usmRadius = fresh.usmRadius;
    g_config.usmThreshold = fresh.usmThreshold;
}

}  // namespace hf
