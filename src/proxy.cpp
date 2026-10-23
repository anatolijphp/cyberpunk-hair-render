// Proxy for amd_fidelityfx_loader_dx12.dll: forwards the FFX API to the original
// (renamed to amd_fidelityfx_loader_dx12_orig.dll) and installs the D3D12 hooks.
#include <mutex>

#include "common.h"
#include "hair.h"
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
    if (hf::g_config.enabled && desc && desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
        static int logged = 0;
        auto* up = reinterpret_cast<const ffxDispatchDescUpscale*>(desc);
        if (logged < 1) {
            ++logged;
            hf::Log("upscale: color=%p fmt=%u %ux%u state=%u depth=%p (fmt=%u state=%u) mv=%p (fmt=%u state=%u) out=%p (fmt=%u state=%u) render=%ux%u -> %ux%u",
                    up->color.resource, up->color.description.format, up->color.description.width,
                    up->color.description.height, up->color.state, up->depth.resource,
                    up->depth.description.format, up->depth.state, up->motionVectors.resource,
                    up->motionVectors.description.format, up->motionVectors.state, up->output.resource, up->output.description.format, up->output.state,
                    up->renderSize.width,
                    up->renderSize.height, up->upscaleSize.width, up->upscaleSize.height);
        }
        ffxDispatchDescUpscale patched = *up;
        hf::UpscaleInputs in, out;
        in.color = up->color.resource;
        in.colorState = up->color.state;
        in.depth = up->depth.resource;
        in.depthState = up->depth.state;
        in.motion = up->motionVectors.resource;
        in.motionState = up->motionVectors.state;
        if (hf::ResolveBeforeUpscale(up->commandList, in, out)) {
            patched.color.resource = out.color;
            patched.color.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
            if (out.depth) {
                patched.depth.resource = out.depth;
                patched.depth.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
                patched.depth.description.format = FFX_API_SURFACE_FORMAT_R32_FLOAT;
            }
            if (out.motion) {
                patched.motionVectors.resource = out.motion;
                patched.motionVectors.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
            }
        }
        ffxReturnCode_t r = g_dispatch(context, &patched.header);
        if (r == 0) {
            hf::ApplyUnsharpMask(up->commandList, up->output.resource, up->output.state, up->renderSize.width,
                                 up->renderSize.height);
            hf::AfterUpscale(up->commandList, up->output.resource, up->output.state);
        }
        hf::FinishUpscale(up->commandList);
        hf::OnFrameBoundary();
        return r;
    }
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
