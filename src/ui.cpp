// Settings overlay (Dear ImGui) drawn into the swap chain back buffer at Present.
//
// - Home (OverlayKey) opens / closes it; Esc closes it.
// - Window messages are captured by subclassing the game window. While the overlay is open,
//   mouse / keyboard / raw input is kept from the game (see also the input hooks in hooks.cpp).
//   ImGui isn't thread-safe, so the window procedure only queues messages and the render thread
//   feeds them to ImGui before each overlay frame.
#include <d3d12.h>
#include <dxgi1_4.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>

#include "common.h"
#include "imgui.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

namespace hf {
namespace {

constexpr UINT kMaxBuffers = 8;
constexpr UINT kSrvSlots = 64;

std::atomic<bool> g_visible{false};
std::mutex g_mutex;   // overlay state (render thread) vs. attach / resize

HWND g_hwnd = nullptr;
WNDPROC g_gameProc = nullptr;   // the game's window procedure (we call it for everything we don't eat)
ID3D12CommandQueue* g_queue = nullptr;
ID3D12Device* g_device = nullptr;

bool g_imguiReady = false;
bool g_initFailed = false;
UINT g_numBuffers = 0;
DXGI_FORMAT g_rtvFormat = DXGI_FORMAT_UNKNOWN;
ID3D12DescriptorHeap* g_rtvHeap = nullptr;
ID3D12DescriptorHeap* g_srvHeap = nullptr;
std::vector<bool> g_srvUsed;
UINT g_rtvInc = 0, g_srvInc = 0;
ID3D12CommandAllocator* g_alloc[kMaxBuffers] = {};
UINT64 g_allocFence[kMaxBuffers] = {};
ID3D12GraphicsCommandList* g_list = nullptr;
ID3D12Fence* g_fence = nullptr;
UINT64 g_fenceValue = 0;
HANDLE g_fenceEvent = nullptr;

struct Msg {
    UINT msg;
    WPARAM wp;
    LPARAM lp;
};
std::mutex g_msgMutex;
std::vector<Msg> g_msgs;

// status line under the buttons
char g_status[64] = "";
uint64_t g_statusUntil = 0;

bool IsInputMessage(UINT m) {
    return (m >= WM_MOUSEFIRST && m <= WM_MOUSELAST) || (m >= WM_KEYFIRST && m <= WM_KEYLAST) || m == WM_INPUT;
}

void SetVisible(bool v) {
    g_visible = v;
    if (v) ClipCursor(nullptr);   // the game re-clips by itself once the overlay closes
    Log("overlay %s", v ? "opened" : "closed");
}

LRESULT CALLBACK OverlayWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    bool keyDown = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
    bool keyUp = msg == WM_KEYUP || msg == WM_SYSKEYUP;
    bool firstPress = keyDown && !(lp & (1 << 30));
    if ((keyDown || keyUp) && ((int)wp == g_config.overlayKey || (int)wp == g_config.toggleKey ||
                               (g_config.captureKey && (int)wp == g_config.captureKey))) {
        if (firstPress && (int)wp == g_config.overlayKey) SetVisible(!g_visible);
        return 0;   // our hotkeys are never passed to the game (F10 is read with GetAsyncKeyState)
    }
    if (g_visible) {
        if (firstPress && wp == VK_ESCAPE) {
            SetVisible(false);
            return 0;
        }
        if (IsInputMessage(msg) || msg == WM_CHAR || msg == WM_SETFOCUS || msg == WM_KILLFOCUS ||
            msg == WM_MOUSELEAVE) {
            std::lock_guard<std::mutex> lock(g_msgMutex);
            if (g_msgs.size() < 1024) g_msgs.push_back({msg, wp, lp});
        }
        if (msg == WM_INPUT) return DefWindowProcW(hwnd, msg, wp, lp);   // releases the raw input data
        if (IsInputMessage(msg) || msg == WM_CHAR) return 0;
    }
    return CallWindowProcW(g_gameProc, hwnd, msg, wp, lp);
}

void SrvAlloc(ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE* cpu, D3D12_GPU_DESCRIPTOR_HANDLE* gpu) {
    for (UINT i = 0; i < kSrvSlots; ++i)
        if (!g_srvUsed[i]) {
            g_srvUsed[i] = true;
            *cpu = g_srvHeap->GetCPUDescriptorHandleForHeapStart();
            *gpu = g_srvHeap->GetGPUDescriptorHandleForHeapStart();
            cpu->ptr += (SIZE_T)i * g_srvInc;
            gpu->ptr += (UINT64)i * g_srvInc;
            return;
        }
    *cpu = {};
    *gpu = {};
}

void SrvFree(ImGui_ImplDX12_InitInfo*, D3D12_CPU_DESCRIPTOR_HANDLE cpu, D3D12_GPU_DESCRIPTOR_HANDLE) {
    UINT i = (UINT)((cpu.ptr - g_srvHeap->GetCPUDescriptorHandleForHeapStart().ptr) / g_srvInc);
    if (i < kSrvSlots) g_srvUsed[i] = false;
}

DXGI_FORMAT RtvFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        default: return f;
    }
}

void WaitIdle() {
    if (!g_fence || !g_queue) return;
    g_queue->Signal(g_fence, ++g_fenceValue);
    if (g_fence->GetCompletedValue() < g_fenceValue) {
        g_fence->SetEventOnCompletion(g_fenceValue, g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, 2000);
    }
}

bool InitImGui(IDXGISwapChain3* sc) {
    if (g_imguiReady) return true;
    if (g_initFailed || !g_queue) return false;
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sc->GetDesc1(&sd);
    g_numBuffers = sd.BufferCount < kMaxBuffers ? sd.BufferCount : kMaxBuffers;
    g_rtvFormat = RtvFormat(sd.Format);
    if (FAILED(g_queue->GetDevice(IID_PPV_ARGS(&g_device)))) {
        g_initFailed = true;
        return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC rh{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, kMaxBuffers, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC sh{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kSrvSlots,
                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    bool ok = SUCCEEDED(g_device->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&g_rtvHeap))) &&
              SUCCEEDED(g_device->CreateDescriptorHeap(&sh, IID_PPV_ARGS(&g_srvHeap))) &&
              SUCCEEDED(g_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
    for (UINT i = 0; ok && i < g_numBuffers; ++i)
        ok = SUCCEEDED(g_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_alloc[i])));
    ok = ok && SUCCEEDED(g_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_alloc[0], nullptr,
                                                    IID_PPV_ARGS(&g_list)));
    if (!ok) {
        Log("overlay: D3D12 init failed");
        g_initFailed = true;
        return false;
    }
    g_list->Close();
    g_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    g_rtvInc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    g_srvInc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    g_srvUsed.assign(kSrvSlots, false);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;     // no imgui.ini next to the game
    io.MouseDrawCursor = true;    // the game hides the system cursor
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    float scale = sd.Height > 1080 ? sd.Height / 1080.0f : 1.0f;
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(scale);
    style.FontScaleMain = scale;
    style.WindowRounding = 6 * scale;
    style.FrameRounding = 4 * scale;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX12_InitInfo info;
    info.Device = g_device;
    info.CommandQueue = g_queue;
    info.NumFramesInFlight = (int)g_numBuffers;
    info.RTVFormat = g_rtvFormat;
    info.DSVFormat = DXGI_FORMAT_UNKNOWN;
    info.SrvDescriptorHeap = g_srvHeap;
    info.SrvDescriptorAllocFn = SrvAlloc;
    info.SrvDescriptorFreeFn = SrvFree;
    if (!ImGui_ImplDX12_Init(&info)) {
        Log("overlay: ImGui DX12 init failed");
        g_initFailed = true;
        return false;
    }
    g_imguiReady = true;
    Log("overlay: initialised (%u buffers, format %d, %ux%u)", g_numBuffers, g_rtvFormat, sd.Width, sd.Height);
    return true;
}

void SetStatus(const char* text) {
    snprintf(g_status, sizeof(g_status), "%s", text);
    g_statusUntil = GetTickCount64() + 2500;
}

void SaveConfig() {
    std::string p;
    {
        std::wstring w = ModuleDir() + L"hairfix.ini";
        p.assign(w.begin(), w.end());
    }
    const Config& c = g_config;
    char b[32];
    auto put = [&](const char* section, const char* key, const char* value) {
        WritePrivateProfileStringA(section, key, value, p.c_str());
    };
    auto putf = [&](const char* key, float v) {
        snprintf(b, sizeof(b), "%.3g", v);
        put("Hair", key, b);
    };
    put("Hair", "Coverage", c.coverageMode == 1 ? "sharpen" : "raw");
    put("Hair", "Resolve", c.resolveMode == 0 ? "box" : c.resolveMode == 2 ? "neural" : "strand");
    putf("Strength", c.strength);
    putf("Area", c.area);
    putf("AreaFeather", c.areaFeather);
    put("Hair", "StabilizeDepth", c.stabilizeDepth ? "true" : "false");
    putf("SharpenThreshold", c.sharpenThreshold);
    putf("SharpenWidth", c.sharpenWidth);
    put("Hair", "UnsharpMask", c.usmEnabled ? "true" : "false");
    putf("USMAmount", c.usmAmount);
    putf("USMRadius", c.usmRadius);
    putf("USMThreshold", c.usmThreshold);
    snprintf(b, sizeof(b), "%d", c.debugView);
    put("Debug", "View", b);
    Log("overlay: settings saved");
    SetStatus("Saved to hairfix.ini");
}

void ResetToDefaults() {
    const Config d;
    Config& c = g_config;
    c.coverageMode = d.coverageMode;
    c.resolveMode = d.resolveMode;
    c.strength = d.strength;
    c.area = d.area;
    c.areaFeather = d.areaFeather;
    c.stabilizeDepth = d.stabilizeDepth;
    c.sharpenThreshold = d.sharpenThreshold;
    c.sharpenWidth = d.sharpenWidth;
    c.usmEnabled = d.usmEnabled;
    c.usmAmount = d.usmAmount;
    c.usmRadius = d.usmRadius;
    c.usmThreshold = d.usmThreshold;
    c.debugView = d.debugView;
    SetStatus("Defaults restored (not saved yet)");
}

// combo over a fixed list of float choices; returns true when changed
bool FloatChoice(const char* label, float* value, const float* choices, int n, const char* fmt) {
    int current = 0;
    for (int i = 1; i < n; ++i)
        if (fabsf(choices[i] - *value) < fabsf(choices[current] - *value)) current = i;
    char preview[16];
    snprintf(preview, sizeof(preview), fmt, choices[current]);
    bool changed = false;
    if (ImGui::BeginCombo(label, preview)) {
        for (int i = 0; i < n; ++i) {
            char item[16];
            snprintf(item, sizeof(item), fmt, choices[i]);
            if (ImGui::Selectable(item, i == current)) {
                *value = choices[i];
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

void DrawSettings() {
    Config& c = g_config;
    float s = ImGui::GetStyle().FontScaleMain;
    ImGui::SetNextWindowPos(ImVec2(40 * s, 40 * s), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430 * s, 0), ImGuiCond_FirstUseEver);
    bool open = true;
    if (!ImGui::Begin("Hair dither fix", &open, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    ImGui::PushItemWidth(-160 * s);

    bool active = IsActive();
    if (ImGui::Checkbox("Hair fix active (F10)", &active)) SetActive(active);

    ImGui::SeparatorText("Coverage");
    const char* coverage[] = {"Raw (soft, translucent)", "Sharpen (crisp strands)"};
    ImGui::Combo("Coverage", &c.coverageMode, coverage, 2);
    if (c.coverageMode == 1) {
        static const float thresholds[] = {0.2f, 0.3f, 0.4f, 0.5f};
        static const float widths[] = {1.0f, 1.5f, 2.0f, 3.0f};
        FloatChoice("Edge threshold", &c.sharpenThreshold, thresholds, 4, "%.1f");
        FloatChoice("Edge width (px)", &c.sharpenWidth, widths, 4, "%.1f");
        ImGui::TextDisabled("A new preset compiles on first use (brief hitch).");
    }

    ImGui::SeparatorText("Resolve");
    const char* resolve[] = {"Box (5x5)", "Strand-aligned", "Strand + network"};
    ImGui::Combo("Resolve", &c.resolveMode, resolve, 3);
    ImGui::SetItemTooltip("Strand + network: a network refines the resolve from the raw frame;\n"
                          "needs hairfix_resolve.bin next to the mod, else the strand resolve stays");
    ImGui::SliderFloat("Strength", &c.strength, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("0 = keep the most strand detail, 1 = smoothest");
    ImGui::SliderFloat("Area", &c.area, 0.0f, kAreaAll, c.area >= kAreaAll ? "all hair" : "%.1f px from edges");
    ImGui::SetItemTooltip("Where the mod works: how far (render pixels) from strand edges into fully\n"
                          "filled hair. Outside, the image is the game's own, as without the mod\n"
                          "(debug view: Mod area)");
    ImGui::BeginDisabled(c.area >= kAreaAll);
    ImGui::SliderFloat("Area transition", &c.areaFeather, 0.5f, 16.0f, "%.1f px");
    ImGui::SetItemTooltip("How softly the mod fades out beyond the area: 0.5 = hard border, 16 = very soft");
    ImGui::EndDisabled();
    ImGui::Checkbox("Stabilise depth / motion for FSR", &c.stabilizeDepth);

    ImGui::SeparatorText("Unsharp mask (after FSR, hair only)");
    ImGui::Checkbox("Enabled", &c.usmEnabled);
    ImGui::BeginDisabled(!c.usmEnabled);
    ImGui::SliderFloat("Amount", &c.usmAmount, 0.0f, 3.0f, "%.2f");
    ImGui::SliderFloat("Radius (px)", &c.usmRadius, 0.5f, 4.0f, "%.2f");
    ImGui::SliderFloat("Threshold", &c.usmThreshold, 0.0f, 0.3f, "%.3f");
    ImGui::EndDisabled();

    ImGui::SeparatorText("Debug");
    const char* views[] = {"Off", "Coverage", "Hair-surface mask", "Strand coherence", "Unsharp-mask mask",
                           "Network mask", "Network correction", "Mod area"};
    ImGui::Combo("View", &c.debugView, views, 8);

    ImGui::Separator();
    if (ImGui::Button("Save")) SaveConfig();
    ImGui::SameLine();
    if (ImGui::Button("Reset to defaults")) ResetToDefaults();
    ImGui::SameLine();
    if (ImGui::Button("Reload file")) {
        ReloadRuntimeConfig();
        SetStatus("Reloaded hairfix.ini");
    }
    if (GetTickCount64() < g_statusUntil) ImGui::TextUnformatted(g_status);
    else ImGui::TextDisabled("Home / Esc: close   F10: fix on/off");

    ImGui::PopItemWidth();
    ImGui::End();
    if (!open) SetVisible(false);
}

}  // namespace

bool OverlayVisible() { return g_visible.load(std::memory_order_relaxed); }

void* OverlayQueue() { return g_queue; }

void OverlayAttach(void* swapChain, void* queue, HWND hwnd) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (queue && !g_queue) {
        ID3D12CommandQueue* q = nullptr;
        if (SUCCEEDED(static_cast<IUnknown*>(queue)->QueryInterface(IID_PPV_ARGS(&q)))) g_queue = q;
    }
    if (hwnd && hwnd != g_hwnd) {
        if (g_hwnd && g_gameProc) SetWindowLongPtrW(g_hwnd, GWLP_WNDPROC, (LONG_PTR)g_gameProc);
        g_hwnd = hwnd;
        g_gameProc = (WNDPROC)SetWindowLongPtrW(hwnd, GWLP_WNDPROC, (LONG_PTR)OverlayWndProc);
    }
    Log("overlay: attached to swap chain %p (window %p, queue %p)", swapChain, hwnd, g_queue);
}

// Keeps our window procedure first when the game installs its own later; returns the previous one.
bool OverlayRedirectWndProc(HWND hwnd, LONG_PTR newProc, LONG_PTR* previous) {
    if (!g_hwnd || hwnd != g_hwnd || (WNDPROC)newProc == OverlayWndProc) return false;
    *previous = (LONG_PTR)g_gameProc;
    g_gameProc = (WNDPROC)newProc;
    return true;
}

bool OverlayGameWndProc(HWND hwnd, LONG_PTR* proc) {
    if (!g_hwnd || hwnd != g_hwnd) return false;
    *proc = (LONG_PTR)g_gameProc;
    return true;
}

void OverlayOnPresent(void* swapChain) {
    if (!g_visible.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    IDXGISwapChain3* sc = nullptr;
    if (FAILED(static_cast<IUnknown*>(swapChain)->QueryInterface(IID_PPV_ARGS(&sc)))) return;
    if (!InitImGui(sc)) {
        sc->Release();
        return;
    }

    std::vector<Msg> msgs;
    {
        std::lock_guard<std::mutex> l(g_msgMutex);
        msgs.swap(g_msgs);
    }
    for (const Msg& m : msgs) ImGui_ImplWin32_WndProcHandler(g_hwnd, m.msg, m.wp, m.lp);

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    DrawSettings();
    ImGui::Render();

    UINT index = sc->GetCurrentBackBufferIndex();
    UINT frame = index % g_numBuffers;
    if (g_fence->GetCompletedValue() < g_allocFence[frame]) {
        g_fence->SetEventOnCompletion(g_allocFence[frame], g_fenceEvent);
        WaitForSingleObject(g_fenceEvent, 2000);
    }
    ID3D12Resource* buffer = nullptr;
    if (FAILED(sc->GetBuffer(index, IID_PPV_ARGS(&buffer)))) {
        sc->Release();
        return;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = g_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += (SIZE_T)(index % kMaxBuffers) * g_rtvInc;
    D3D12_RENDER_TARGET_VIEW_DESC rd{};
    rd.Format = g_rtvFormat;
    rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    g_device->CreateRenderTargetView(buffer, &rd, rtv);

    g_alloc[frame]->Reset();
    g_list->Reset(g_alloc[frame], nullptr);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = buffer;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    g_list->ResourceBarrier(1, &b);
    g_list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    g_list->SetDescriptorHeaps(1, &g_srvHeap);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_list);
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_list->ResourceBarrier(1, &b);
    g_list->Close();
    ID3D12CommandList* lists[] = {g_list};
    g_queue->ExecuteCommandLists(1, lists);
    g_queue->Signal(g_fence, ++g_fenceValue);
    g_allocFence[frame] = g_fenceValue;
    buffer->Release();
    sc->Release();
}

void OverlayBeforeResize() {
    std::lock_guard<std::mutex> lock(g_mutex);
    WaitIdle();   // we hold no back-buffer references between frames, so only in-flight work matters
}

}  // namespace hf
