// Proxy for amd_fidelityfx_loader_dx12.dll: forwards the FFX API to the original
// (renamed to amd_fidelityfx_loader_dx12_orig.dll) and installs the D3D12 hooks. Upscale dispatches
// (FSR 3 / 4) go through fsr3.cpp; the built-in FSR 2.1 never calls this DLL (fsr2.cpp).
#include <mutex>

#include "common.h"
#include "fsr3.h"
#include "../third_party/FidelityFX/api/include/ffx_api.h"
#include "../third_party/FidelityFX/upscalers/include/ffx_upscale.h"
#include "../third_party/FidelityFX/api/include/dx12/ffx_api_dx12.h"

namespace {

PfnFfxCreateContext g_createContext;
PfnFfxDestroyContext g_destroyContext;
PfnFfxConfigure g_configure;
PfnFfxQuery g_query;
PfnFfxDispatch g_dispatch;
std::once_flag g_loadOnce;
bool g_loaded = false;

void LoadOriginal() {
    std::call_once(g_loadOnce, [] {
        std::wstring path = hf::ModuleDir() + L"amd_fidelityfx_loader_dx12_orig.dll";
        HMODULE m = LoadLibraryW(path.c_str());
        if (!m) {
            hf::Log("ERROR: cannot load %ls (error %lu)", path.c_str(), GetLastError());
            return;
        }
        g_createContext = (PfnFfxCreateContext)GetProcAddress(m, "ffxCreateContext");
        g_destroyContext = (PfnFfxDestroyContext)GetProcAddress(m, "ffxDestroyContext");
        g_configure = (PfnFfxConfigure)GetProcAddress(m, "ffxConfigure");
        g_query = (PfnFfxQuery)GetProcAddress(m, "ffxQuery");
        g_dispatch = (PfnFfxDispatch)GetProcAddress(m, "ffxDispatch");
        g_loaded = g_createContext && g_destroyContext && g_configure && g_query && g_dispatch;
        hf::Log("original FFX loader %s", g_loaded ? "loaded" : "MISSING EXPORTS");
    });
}

const ffxReturnCode_t kErrorRuntime = 0x2;   // FFX_API_RETURN_ERROR_RUNTIME_ERROR

// from ffx_api_framegeneration_dx12.h (not vendored): the swap chain descriptors share this prefix
const uint64_t kFgSwapChainWrap = 0x30001, kFgSwapChainNew = 0x30005, kFgSwapChainForHwnd = 0x30006;
struct FgSwapChainDesc {
    ffxCreateContextDescHeader header;
    void** swapchain;   // IDXGISwapChain4**: in the game's swap chain, out the frame generation one
};

}  // namespace

extern "C" {

__declspec(dllexport) ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc,
                                                       const ffxAllocationCallbacks* memCb) {
    LoadOriginal();
    if (!g_loaded) return kErrorRuntime;
    if (hf::g_config.enabled) {
        for (auto* h = desc; h; h = h->pNext)
            if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12)
            {
                hf::g_ffxDevice = reinterpret_cast<ffxCreateBackendDX12Desc*>(h)->device;
                hf::HookDeviceFromFfx(hf::g_ffxDevice);
            }
    }
    if (hf::g_config.enabled) hf::FfxCreateBegin();
    ffxReturnCode_t r = g_createContext(context, desc, memCb);
    if (hf::g_config.enabled) {
        hf::FfxCreateEnd();
        // frame generation swap chain (wrap / new / for-hwnd): all start with IDXGISwapChain4** swapchain
        for (auto* h = desc; h && r == 0; h = h->pNext)
            if (h->type == kFgSwapChainWrap || h->type == kFgSwapChainNew || h->type == kFgSwapChainForHwnd) {
                void** sc = reinterpret_cast<FgSwapChainDesc*>(h)->swapchain;
                if (sc) hf::HookFfxSwapChain(*sc);
            }
    }
    for (auto* h = desc; h; h = h->pNext)
        hf::Log("ffxCreateContext type=0x%llx -> %u ctx=%p", (unsigned long long)h->type, r,
                context ? *context : nullptr);
    return r;
}

__declspec(dllexport) ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    LoadOriginal();
    if (!g_loaded) return kErrorRuntime;
    return g_destroyContext(context, memCb);
}

__declspec(dllexport) ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    LoadOriginal();
    if (!g_loaded) return kErrorRuntime;
    return g_configure(context, desc);
}

__declspec(dllexport) ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    LoadOriginal();
    if (!g_loaded) return kErrorRuntime;
    return g_query(context, desc);
}

__declspec(dllexport) ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    LoadOriginal();
    if (!g_loaded) return kErrorRuntime;
    if (hf::g_config.enabled && desc && desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE)
        return hf::Fsr3Upscale(g_dispatch, context, reinterpret_cast<const ffxDispatchDescUpscale*>(desc));
    return g_dispatch(context, desc);
}

}  // extern "C"

BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        hf::g_self = inst;
        DisableThreadLibraryCalls(inst);
        hf::LoadConfig();
        if (hf::g_config.enabled)
            hf::InstallD3D12Hooks();
        else
            hf::Log("disabled in hairfix.ini: pass-through only");
    }
    return TRUE;
}
