// D3D12 hooks (phase 1: detection + diagnostics only, no rendering changes).
//
// Device discovery (the game's real device creation is not visible through its import table):
// - IAT patch of D3D12CreateDevice in the exe; a capability-check call (null out pointer)
//   is used to create a throwaway device and hook its vtable (vkd3d shares one vtable
//   between devices)
// - IAT patch of GetProcAddress in the exe and sl.interposer.dll, so runtime lookups of
//   D3D12CreateDevice also go through the hook
// - the device handed to ffxCreateContext is hooked as a last resort
//
// Hooks keep their original function pointers per vtable, so a Streamline proxy and the
// native device can both be hooked safely.
#define CINTERFACE
#define COBJMACROS
#define INITGUID
#include <d3d12.h>
#include <dxgi1_4.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "hair.h"
#include "pso.h"

namespace hf {
namespace {

// ---------------------------------------------------------------------------------------------
// vtable / IAT patching

#define VSLOT(Vtbl, Method) (offsetof(Vtbl, Method) / sizeof(void*))

enum Fn {
    kCreateRootSig, kCreateGPSO, kCreateCL, kCreatePSO, kCreateCL1, kCreateLib, kLoadGPipe, kLoadPipe,
    kSetPSO, kDrawIndexed, kDraw, kOMSetRT, kBeginRP, kEndRP,
    kCreateSCHwnd, kCreateSC, kPresent, kPresent1, kResizeBuffers, kResizeBuffers1, kFnCount
};

struct VtHooks {
    std::atomic<void**> vt{nullptr};
    void* orig[kFnCount]{};
};

constexpr int kMaxVt = 16;
VtHooks g_vts[kMaxVt];
std::atomic<int> g_numVt{0};
std::mutex g_patchMutex;

void** VtableOf(void* obj) { return *reinterpret_cast<void***>(obj); }

VtHooks* FindVt(void** vt) {
    int n = g_numVt.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i)
        if (g_vts[i].vt.load(std::memory_order_relaxed) == vt) return &g_vts[i];
    return nullptr;
}

template <typename T>
T Orig(void* self, Fn fn) {
    VtHooks* e = FindVt(VtableOf(self));
    return e ? reinterpret_cast<T>(e->orig[fn]) : nullptr;
}

bool WritePtr(void** slot, void* value) {
    DWORD old;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
    *slot = value;
    VirtualProtect(slot, sizeof(void*), old, &old);
    return true;
}

// Hooks vtable slot `slot` of `vt` for function `fn`; the original is stored per vtable.
bool HookSlot(void** vt, size_t slot, Fn fn, void* hook) {
    std::lock_guard<std::mutex> lock(g_patchMutex);
    if (vt[slot] == hook) return false;
    VtHooks* e = FindVt(vt);
    if (!e) {
        int i = g_numVt.load();
        if (i >= kMaxVt) return false;
        g_vts[i].vt.store(vt);
        e = &g_vts[i];
        g_numVt.store(i + 1, std::memory_order_release);
    }
    e->orig[fn] = vt[slot];
    return WritePtr(&vt[slot], hook);
}

// patch every import of `func` in `module`, whatever DLL it comes from
int PatchImport(HMODULE module, const char* func, void* hook, void** orig) {
    auto* base = reinterpret_cast<uint8_t*>(module);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return 0;
    int patched = 0;
    for (auto* imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); imp->Name; ++imp) {
        if (!imp->OriginalFirstThunk) continue;
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imp->OriginalFirstThunk);
        auto* addrs = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++addrs) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            auto* byName = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (strcmp(reinterpret_cast<const char*>(byName->Name), func) != 0) continue;
            void** slot = reinterpret_cast<void**>(&addrs->u1.Function);
            if (*slot == hook) continue;
            if (orig && !*orig) *orig = *slot;
            if (WritePtr(slot, hook)) ++patched;
        }
    }
    return patched;
}

// ---------------------------------------------------------------------------------------------
// state

std::mutex g_mutex;
std::unordered_map<void*, std::vector<uint8_t>> g_rootSigBlobs;   // ID3D12RootSignature* -> blob
std::unordered_set<uint64_t> g_dumped;

constexpr int kMaxHairPsos = 256;
std::atomic<void*> g_hairPsos[kMaxHairPsos];       // alpha_accum PSOs (lock-free lookup in draws)
std::atomic<void*> g_hairGbufPsos[kMaxHairPsos];   // gbuffer_solid PSOs
std::atomic<int> g_numHair{0}, g_numHairGbuf{0};

std::atomic<uint32_t> g_frameAlphaDraws{0}, g_frameGbufDraws{0};
std::atomic<uint32_t> g_framePsosCreated{0};   // total PSOs seen by the hooks
std::atomic<bool> g_deviceHooked{false};
uint64_t g_frame = 0;

bool InSet(std::atomic<void*>* set, int n, void* p) {
    if (!p) return false;   // freed slots hold nullptr
    for (int i = 0; i < n; ++i)
        if (set[i].load(std::memory_order_relaxed) == p) return true;
    return false;
}

void AddToSet(std::atomic<void*>* set, std::atomic<int>& n, void* p) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int count = n.load();
    for (int i = 0; i < count; ++i)
        if (set[i].load() == p) return;
    for (int i = 0; i < count; ++i)   // reuse a slot freed by RemoveFromSet
        if (!set[i].load()) {
            set[i].store(p);
            return;
        }
    if (count < kMaxHairPsos) {
        set[count].store(p);
        n.store(count + 1);
    }
}

void RemoveFromSet(std::atomic<void*>* set, std::atomic<int>& n, void* p) {
    int count = n.load();
    for (int i = 0; i < count; ++i)
        if (set[i].load(std::memory_order_relaxed) == p) set[i].store(nullptr);
}

void DumpPso(HairKind kind, const PsoInfo& p) {
    if (!g_config.dumpShaders || !p.ps.pShaderBytecode) return;
    uint64_t h = Hash64(p.ps.pShaderBytecode, p.ps.BytecodeLength);
    std::vector<uint8_t> rootSig;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_dumped.insert(h).second) return;
        auto it = g_rootSigBlobs.find(p.rootSig);
        if (it != g_rootSigBlobs.end()) rootSig = it->second;
    }
    std::wstring dir = ModuleDir() + L"hairfix_dump\\";
    CreateDirectoryW(dir.c_str(), nullptr);
    wchar_t name[128];
    swprintf(name, 128, L"%hs_%016llx", KindName(kind), (unsigned long long)h);
    std::wstring base = dir + name;
    WriteFile(base + L"_ps.dxil", p.ps.pShaderBytecode, p.ps.BytecodeLength);
    if (p.vs.pShaderBytecode) WriteFile(base + L"_vs.dxil", p.vs.pShaderBytecode, p.vs.BytecodeLength);
    if (!rootSig.empty()) WriteFile(base + L"_rootsig.bin", rootSig.data(), rootSig.size());
    Log("dumped %s ps=%016llx (%zu bytes) vs=%zu bytes rootsig=%zu bytes", KindName(kind),
        (unsigned long long)h, (size_t)p.ps.BytecodeLength, (size_t)p.vs.BytecodeLength, rootSig.size());
}

// Logs (once per distinct signature) every PSO that writes an A8_UNORM target, in case the
// hair classifier is too strict.
void LogA8Candidates(const PsoInfo& p, const char* source) {
    bool a8 = false;
    for (UINT i = 0; i < p.numRT && i < 8; ++i) a8 |= p.rtv[i] == DXGI_FORMAT_A8_UNORM;
    if (!a8) return;
    const auto& b = p.blend0;
    uint64_t sig = Hash64(&p.blend0, sizeof(p.blend0)) ^ (p.numRT * 0x9E3779B97F4A7C15ull) ^
                   ((uint64_t)p.stencilEnable << 40) ^ ((uint64_t)p.stencilFunc << 44) ^ ((uint64_t)p.depthWrite << 50);
    static std::unordered_set<uint64_t> seen;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (seen.size() > 64 || !seen.insert(sig).second) return;
    }
    Log("A8 PSO (%s): rts=%u fmt0=%d blend=%d src=%d dst=%d op=%d srcA=%d dstA=%d opA=%d mask=0x%x "
        "stencil=%d func=%d dwrite=%d",
        source, p.numRT, p.rtv[0], b.BlendEnable, b.SrcBlend, b.DestBlend, b.BlendOp, b.SrcBlendAlpha,
        b.DestBlendAlpha, b.BlendOpAlpha, b.RenderTargetWriteMask, p.stencilEnable, p.stencilFunc, p.depthWrite);
}

void OnPsoCreated(const PsoInfo& p, void* pso, const char* source, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* gdesc,
                  const D3D12_PIPELINE_STATE_STREAM_DESC* sdesc) {
    g_framePsosCreated.fetch_add(1, std::memory_order_relaxed);
    LogA8Candidates(p, source);
    HairKind kind = Classify(p);
    if (!pso) return;
    // PSO addresses get reused after the game frees pipelines: drop stale hair entries
    if (kind != HairKind::AlphaAccum) {
        RemoveFromSet(g_hairPsos, g_numHair, pso);
        ForgetGamePso(pso);
    }
    if (kind != HairKind::GBuffer) RemoveFromSet(g_hairGbufPsos, g_numHairGbuf, pso);
    if (kind == HairKind::None) return;
    Log("hair PSO %s (%s): %p rts=%u fmt0=%d dsv=%d stencil=%d func=%d rmask=0x%x wmask=0x%x dwrite=%d",
        KindName(kind), source, pso, p.numRT, p.rtv[0], p.dsv, p.stencilEnable, p.stencilFunc, p.stencilReadMask,
        p.stencilWriteMask, p.depthWrite);
    if (kind == HairKind::AlphaAccum) {
        AddToSet(g_hairPsos, g_numHair, pso);
        RegisterAlphaAccumPso(pso, Hash64(p.ps.pShaderBytecode, p.ps.BytecodeLength), gdesc, sdesc);
    }
    if (kind == HairKind::GBuffer) AddToSet(g_hairGbufPsos, g_numHairGbuf, pso);
    DumpPso(kind, p);
}

// ---------------------------------------------------------------------------------------------
// command list hooks

thread_local void* t_list = nullptr;   // last list SetPipelineState was called on (this thread)
thread_local int t_listKind = 0;       // 1 = alpha_accum PSO bound, 2 = gbuffer_solid bound
thread_local void* t_pso = nullptr;    // PSO bound by that SetPipelineState

using PFN_SetPSO = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
using PFN_DrawIndexed = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT, INT, UINT);
using PFN_Draw = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT, UINT);

void STDMETHODCALLTYPE H_SetPipelineState(ID3D12GraphicsCommandList* self, ID3D12PipelineState* pso) {
    t_list = self;
    t_pso = pso;
    int n1 = g_numHair.load(std::memory_order_relaxed), n2 = g_numHairGbuf.load(std::memory_order_relaxed);
    t_listKind = InSet(g_hairPsos, n1, pso) ? 1 : InSet(g_hairGbufPsos, n2, pso) ? 2 : 0;
    Orig<PFN_SetPSO>(self, kSetPSO)(self, pso);
}

inline void CountDraw(ID3D12GraphicsCommandList* self) {
    if (t_list != self || !t_listKind) return;
    (t_listKind == 1 ? g_frameAlphaDraws : g_frameGbufDraws).fetch_add(1, std::memory_order_relaxed);
}

void STDMETHODCALLTYPE H_DrawIndexedInstanced(ID3D12GraphicsCommandList* self, UINT a, UINT b, UINT c, INT d, UINT e) {
    Orig<PFN_DrawIndexed>(self, kDrawIndexed)(self, a, b, c, d, e);
    CountDraw(self);
    if (t_list == self && t_listKind == 1) AfterAlphaAccumDraw(self, t_pso, true, a, b, c, d, e);
}

void STDMETHODCALLTYPE H_DrawInstanced(ID3D12GraphicsCommandList* self, UINT a, UINT b, UINT c, UINT d) {
    Orig<PFN_Draw>(self, kDraw)(self, a, b, c, d);
    CountDraw(self);
    if (t_list == self && t_listKind == 1) AfterAlphaAccumDraw(self, t_pso, false, a, b, c, (INT)d, 0);
}

using PFN_OMSetRT = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL,
                                             const D3D12_CPU_DESCRIPTOR_HANDLE*);
using PFN_BeginRP = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*, UINT, const D3D12_RENDER_PASS_RENDER_TARGET_DESC*,
                                             const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC*, D3D12_RENDER_PASS_FLAGS);
using PFN_EndRP = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList4*);

void STDMETHODCALLTYPE H_OMSetRenderTargets(ID3D12GraphicsCommandList* self, UINT n, const D3D12_CPU_DESCRIPTOR_HANDLE* rts,
                                            BOOL single, const D3D12_CPU_DESCRIPTOR_HANDLE* dsv) {
    TrackRenderTargets(self, n, rts, single, dsv);
    Orig<PFN_OMSetRT>(self, kOMSetRT)(self, n, rts, single, dsv);
}

void STDMETHODCALLTYPE H_BeginRenderPass(ID3D12GraphicsCommandList4* self, UINT n,
                                         const D3D12_RENDER_PASS_RENDER_TARGET_DESC* rts,
                                         const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC* ds, D3D12_RENDER_PASS_FLAGS flags) {
    TrackRenderPass(self, true);
    Orig<PFN_BeginRP>(self, kBeginRP)(self, n, rts, ds, flags);
}

void STDMETHODCALLTYPE H_EndRenderPass(ID3D12GraphicsCommandList4* self) {
    TrackRenderPass(self, false);
    Orig<PFN_EndRP>(self, kEndRP)(self);
}

void HookCommandList(void* list) {
    void** vt = VtableOf(list);
    bool any = false;
    any |= HookSlot(vt, VSLOT(ID3D12GraphicsCommandListVtbl, SetPipelineState), kSetPSO, (void*)H_SetPipelineState);
    any |= HookSlot(vt, VSLOT(ID3D12GraphicsCommandListVtbl, DrawIndexedInstanced), kDrawIndexed,
                    (void*)H_DrawIndexedInstanced);
    any |= HookSlot(vt, VSLOT(ID3D12GraphicsCommandListVtbl, DrawInstanced), kDraw, (void*)H_DrawInstanced);
    any |= HookSlot(vt, VSLOT(ID3D12GraphicsCommandListVtbl, OMSetRenderTargets), kOMSetRT,
                    (void*)H_OMSetRenderTargets);
    ID3D12GraphicsCommandList4* l4 = nullptr;
    if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)list, IID_ID3D12GraphicsCommandList4, (void**)&l4)) && l4) {
        void** vt4 = VtableOf(l4);
        any |= HookSlot(vt4, VSLOT(ID3D12GraphicsCommandList4Vtbl, BeginRenderPass), kBeginRP, (void*)H_BeginRenderPass);
        any |= HookSlot(vt4, VSLOT(ID3D12GraphicsCommandList4Vtbl, EndRenderPass), kEndRP, (void*)H_EndRenderPass);
        IUnknown_Release((IUnknown*)l4);
    }
    if (any) Log("hooked command list vtable %p", vt);
}

// ---------------------------------------------------------------------------------------------
// device hooks

using PFN_CreateRootSig = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const void*, SIZE_T, REFIID, void**);
using PFN_CreateGPSO = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_GRAPHICS_PIPELINE_STATE_DESC*, REFIID, void**);
using PFN_CreatePSO = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device2*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
using PFN_CreateCL = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_COMMAND_LIST_TYPE, ID3D12CommandAllocator*,
                                                 ID3D12PipelineState*, REFIID, void**);
using PFN_CreateCL1 = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device4*, UINT, D3D12_COMMAND_LIST_TYPE,
                                                  D3D12_COMMAND_LIST_FLAGS, REFIID, void**);

HRESULT STDMETHODCALLTYPE H_CreateRootSignature(ID3D12Device* self, UINT mask, const void* blob, SIZE_T len,
                                                REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_CreateRootSig>(self, kCreateRootSig)(self, mask, blob, len, riid, out);
    if (SUCCEEDED(hr) && out && *out && g_config.dumpShaders) {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto* b = static_cast<const uint8_t*>(blob);
        g_rootSigBlobs[*out].assign(b, b + len);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateGraphicsPipelineState(ID3D12Device* self, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* d,
                                                        REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_CreateGPSO>(self, kCreateGPSO)(self, d, riid, out);
    if (SUCCEEDED(hr) && d && out) OnPsoCreated(FromGraphicsDesc(d), *out, "create", d, nullptr);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreatePipelineState(ID3D12Device2* self, const D3D12_PIPELINE_STATE_STREAM_DESC* d,
                                                REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_CreatePSO>(self, kCreatePSO)(self, d, riid, out);
    PsoInfo p;
    if (SUCCEEDED(hr) && d && out && FromStream(d, p) && p.ps.pShaderBytecode) OnPsoCreated(p, *out, "create", nullptr, d);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateCommandList(ID3D12Device* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
                                              ID3D12CommandAllocator* alloc, ID3D12PipelineState* pso, REFIID riid,
                                              void** out) {
    HRESULT hr = Orig<PFN_CreateCL>(self, kCreateCL)(self, mask, type, alloc, pso, riid, out);
    if (SUCCEEDED(hr) && out && *out && type == D3D12_COMMAND_LIST_TYPE_DIRECT) HookCommandList(*out);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateCommandList1(ID3D12Device4* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
                                               D3D12_COMMAND_LIST_FLAGS flags, REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_CreateCL1>(self, kCreateCL1)(self, mask, type, flags, riid, out);
    if (SUCCEEDED(hr) && out && *out && type == D3D12_COMMAND_LIST_TYPE_DIRECT) HookCommandList(*out);
    return hr;
}

using PFN_CreateLib = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device1*, const void*, SIZE_T, REFIID, void**);
using PFN_LoadGPipe = HRESULT(STDMETHODCALLTYPE*)(ID3D12PipelineLibrary*, LPCWSTR,
                                                  const D3D12_GRAPHICS_PIPELINE_STATE_DESC*, REFIID, void**);
using PFN_LoadPipe = HRESULT(STDMETHODCALLTYPE*)(ID3D12PipelineLibrary1*, LPCWSTR,
                                                 const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);

HRESULT STDMETHODCALLTYPE H_LoadGraphicsPipeline(ID3D12PipelineLibrary* self, LPCWSTR name,
                                                 const D3D12_GRAPHICS_PIPELINE_STATE_DESC* d, REFIID riid,
                                                 void** out) {
    HRESULT hr = Orig<PFN_LoadGPipe>(self, kLoadGPipe)(self, name, d, riid, out);
    if (SUCCEEDED(hr) && d && out) OnPsoCreated(FromGraphicsDesc(d), *out, "library", d, nullptr);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_LoadPipeline(ID3D12PipelineLibrary1* self, LPCWSTR name,
                                         const D3D12_PIPELINE_STATE_STREAM_DESC* d, REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_LoadPipe>(self, kLoadPipe)(self, name, d, riid, out);
    PsoInfo p;
    if (SUCCEEDED(hr) && d && out && FromStream(d, p) && p.ps.pShaderBytecode) OnPsoCreated(p, *out, "library", nullptr, d);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreatePipelineLibrary(ID3D12Device1* self, const void* blob, SIZE_T len, REFIID riid,
                                                  void** out) {
    HRESULT hr = Orig<PFN_CreateLib>(self, kCreateLib)(self, blob, len, riid, out);
    Log("CreatePipelineLibrary(%zu bytes) -> 0x%lx lib=%p", (size_t)len, hr, out ? *out : nullptr);
    if (SUCCEEDED(hr) && out && *out) {
        ID3D12PipelineLibrary* lib = nullptr;
        if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)*out, IID_ID3D12PipelineLibrary, (void**)&lib)) && lib) {
            HookSlot(VtableOf(lib), VSLOT(ID3D12PipelineLibraryVtbl, LoadGraphicsPipeline), kLoadGPipe,
                     (void*)H_LoadGraphicsPipeline);
            ID3D12PipelineLibrary_Release(lib);
        }
        ID3D12PipelineLibrary1* lib1 = nullptr;
        if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)*out, IID_ID3D12PipelineLibrary1, (void**)&lib1)) && lib1) {
            HookSlot(VtableOf(lib1), VSLOT(ID3D12PipelineLibrary1Vtbl, LoadPipeline), kLoadPipe,
                     (void*)H_LoadPipeline);
            ID3D12PipelineLibrary1_Release(lib1);
        }
    }
    return hr;
}

void HookDevice(void* device, const char* source) {
    ID3D12Device* dev = nullptr;
    if (FAILED(IUnknown_QueryInterface((IUnknown*)device, IID_ID3D12Device, (void**)&dev)) || !dev) {
        Log("%s: object %p is not an ID3D12Device", source, device);
        return;
    }
    void** vt = VtableOf(dev);
    int n = 0;
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateRootSignature), kCreateRootSig, (void*)H_CreateRootSignature);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateGraphicsPipelineState), kCreateGPSO,
                  (void*)H_CreateGraphicsPipelineState);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateCommandList), kCreateCL, (void*)H_CreateCommandList);

    ID3D12Device1* dev1 = nullptr;
    if (SUCCEEDED(ID3D12Device_QueryInterface(dev, IID_ID3D12Device1, (void**)&dev1)) && dev1) {
        n += HookSlot(VtableOf(dev1), VSLOT(ID3D12Device1Vtbl, CreatePipelineLibrary), kCreateLib,
                      (void*)H_CreatePipelineLibrary);
        ID3D12Device1_Release(dev1);
    }
    ID3D12Device2* dev2 = nullptr;
    if (SUCCEEDED(ID3D12Device_QueryInterface(dev, IID_ID3D12Device2, (void**)&dev2)) && dev2) {
        n += HookSlot(VtableOf(dev2), VSLOT(ID3D12Device2Vtbl, CreatePipelineState), kCreatePSO,
                      (void*)H_CreatePipelineState);
        ID3D12Device2_Release(dev2);
    }
    ID3D12Device4* dev4 = nullptr;
    if (SUCCEEDED(ID3D12Device_QueryInterface(dev, IID_ID3D12Device4, (void**)&dev4)) && dev4) {
        n += HookSlot(VtableOf(dev4), VSLOT(ID3D12Device4Vtbl, CreateCommandList1), kCreateCL1,
                      (void*)H_CreateCommandList1);
        ID3D12Device4_Release(dev4);
    }
    Log("%s: device %p vtable %p, %d slot(s) newly hooked", source, dev, vt, n);
    g_deviceHooked.store(true);
    ID3D12Device_Release(dev);
}

using PFN_D3D12CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
PFN_D3D12CreateDevice o_D3D12CreateDeviceImport;   // exe import (sl.interposer.dll)
PFN_D3D12CreateDevice o_D3D12CreateDeviceLookup;   // resolved through GetProcAddress
HRESULT WINAPI H_D3D12CreateDeviceLookup(IUnknown* adapter, D3D_FEATURE_LEVEL level, REFIID riid, void** out);

HRESULT CreateDeviceCommon(PFN_D3D12CreateDevice orig, const char* source, IUnknown* adapter,
                           D3D_FEATURE_LEVEL level, REFIID riid, void** out) {
    HRESULT hr = orig(adapter, level, riid, out);
    Log("%s D3D12CreateDevice(adapter=%p level=0x%x) -> 0x%lx device=%p", source, adapter, level, hr,
        out ? *out : nullptr);
    if (SUCCEEDED(hr) && out && *out) {
        HookDevice(*out, source);
    } else if (SUCCEEDED(hr) && !out && !g_deviceHooked.load()) {
        // Capability check: create a throwaway device straight from d3d12.dll (bypassing
        // Streamline) on the default adapter, just to reach and hook the shared device vtable.
        HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
        if (!d3d12) d3d12 = LoadLibraryW(L"d3d12.dll");
        auto native = d3d12 ? (PFN_D3D12CreateDevice)GetProcAddress(d3d12, "D3D12CreateDevice") : nullptr;
        if (native == H_D3D12CreateDeviceLookup) native = o_D3D12CreateDeviceLookup;
        void* tmp = nullptr;
        if (native && SUCCEEDED(native(nullptr, level, IID_ID3D12Device, &tmp)) && tmp) {
            HookDevice(tmp, "probe device");
            IUnknown_Release((IUnknown*)tmp);
        } else {
            Log("probe device creation failed");
        }
    }
    return hr;
}

HRESULT WINAPI H_D3D12CreateDeviceImport(IUnknown* adapter, D3D_FEATURE_LEVEL level, REFIID riid, void** out) {
    return CreateDeviceCommon(o_D3D12CreateDeviceImport, "import", adapter, level, riid, out);
}

HRESULT WINAPI H_D3D12CreateDeviceLookup(IUnknown* adapter, D3D_FEATURE_LEVEL level, REFIID riid, void** out) {
    return CreateDeviceCommon(o_D3D12CreateDeviceLookup, "lookup", adapter, level, riid, out);
}

using PFN_GetProcAddress = FARPROC(WINAPI*)(HMODULE, LPCSTR);
PFN_GetProcAddress o_GetProcAddress;

FARPROC WINAPI H_GetProcAddress(HMODULE module, LPCSTR name) {
    FARPROC p = o_GetProcAddress(module, name);
    if (p && HIWORD((ULONG_PTR)name) && strcmp(name, "D3D12CreateDevice") == 0 &&
        p != (FARPROC)H_D3D12CreateDeviceImport && p != (FARPROC)H_D3D12CreateDeviceLookup) {
        wchar_t path[MAX_PATH] = L"?";
        GetModuleFileNameW(module, path, MAX_PATH);
        Log("GetProcAddress(%ls, D3D12CreateDevice) = %p -> hooked", path, (void*)p);
        o_D3D12CreateDeviceLookup = (PFN_D3D12CreateDevice)p;
        return (FARPROC)H_D3D12CreateDeviceLookup;
    }
    return p;
}


// ---------------------------------------------------------------------------------------------
// DXGI hooks for the settings overlay: factory -> swap chain creation -> Present / ResizeBuffers

using PFN_Present = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using PFN_Present1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
using PFN_ResizeBuffers = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
using PFN_ResizeBuffers1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*, UINT, UINT, UINT, DXGI_FORMAT, UINT,
                                                       const UINT*, IUnknown* const*);
using PFN_CreateSCHwnd = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
                                                     const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*,
                                                     IDXGISwapChain1**);
using PFN_CreateSC = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);

HRESULT STDMETHODCALLTYPE H_Present(IDXGISwapChain* self, UINT sync, UINT flags) {
    if (!(flags & DXGI_PRESENT_TEST)) OverlayOnPresent(self);
    return Orig<PFN_Present>(self, kPresent)(self, sync, flags);
}

HRESULT STDMETHODCALLTYPE H_Present1(IDXGISwapChain1* self, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) {
    if (!(flags & DXGI_PRESENT_TEST)) OverlayOnPresent(self);
    return Orig<PFN_Present1>(self, kPresent1)(self, sync, flags, p);
}

HRESULT STDMETHODCALLTYPE H_ResizeBuffers(IDXGISwapChain* self, UINT n, UINT w, UINT h, DXGI_FORMAT f, UINT flags) {
    OverlayBeforeResize();
    return Orig<PFN_ResizeBuffers>(self, kResizeBuffers)(self, n, w, h, f, flags);
}

HRESULT STDMETHODCALLTYPE H_ResizeBuffers1(IDXGISwapChain3* self, UINT n, UINT w, UINT h, DXGI_FORMAT f, UINT flags,
                                           const UINT* masks, IUnknown* const* queues) {
    OverlayBeforeResize();
    return Orig<PFN_ResizeBuffers1>(self, kResizeBuffers1)(self, n, w, h, f, flags, masks, queues);
}

void HookSwapChain(void* swapChain, IUnknown* queue, HWND hwnd) {
    void** vt = VtableOf(swapChain);
    HookSlot(vt, VSLOT(IDXGISwapChainVtbl, Present), kPresent, (void*)H_Present);
    HookSlot(vt, VSLOT(IDXGISwapChainVtbl, ResizeBuffers), kResizeBuffers, (void*)H_ResizeBuffers);
    IDXGISwapChain1* sc1 = nullptr;
    if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)swapChain, IID_IDXGISwapChain1, (void**)&sc1)) && sc1) {
        HookSlot(VtableOf(sc1), VSLOT(IDXGISwapChain1Vtbl, Present1), kPresent1, (void*)H_Present1);
        IUnknown_Release((IUnknown*)sc1);
    }
    IDXGISwapChain3* sc3 = nullptr;
    if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)swapChain, IID_IDXGISwapChain3, (void**)&sc3)) && sc3) {
        HookSlot(VtableOf(sc3), VSLOT(IDXGISwapChain3Vtbl, ResizeBuffers1), kResizeBuffers1,
                 (void*)H_ResizeBuffers1);
        IUnknown_Release((IUnknown*)sc3);
    }
    OverlayAttach(swapChain, queue, hwnd);
}

HRESULT STDMETHODCALLTYPE H_CreateSwapChainForHwnd(IDXGIFactory2* self, IUnknown* queue, HWND hwnd,
                                                   const DXGI_SWAP_CHAIN_DESC1* desc,
                                                   const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs, IDXGIOutput* output,
                                                   IDXGISwapChain1** out) {
    HRESULT hr = Orig<PFN_CreateSCHwnd>(self, kCreateSCHwnd)(self, queue, hwnd, desc, fs, output, out);
    if (SUCCEEDED(hr) && out && *out) HookSwapChain(*out, queue, hwnd);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateSwapChain(IDXGIFactory* self, IUnknown* queue, DXGI_SWAP_CHAIN_DESC* desc,
                                            IDXGISwapChain** out) {
    HRESULT hr = Orig<PFN_CreateSC>(self, kCreateSC)(self, queue, desc, out);
    if (SUCCEEDED(hr) && out && *out && desc) HookSwapChain(*out, queue, desc->OutputWindow);
    return hr;
}

void HookFactory(void* factory) {
    if (!factory) return;
    HookSlot(VtableOf(factory), VSLOT(IDXGIFactoryVtbl, CreateSwapChain), kCreateSC, (void*)H_CreateSwapChain);
    IDXGIFactory2* f2 = nullptr;
    if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)factory, IID_IDXGIFactory2, (void**)&f2)) && f2) {
        HookSlot(VtableOf(f2), VSLOT(IDXGIFactory2Vtbl, CreateSwapChainForHwnd), kCreateSCHwnd,
                 (void*)H_CreateSwapChainForHwnd);
        IUnknown_Release((IUnknown*)f2);
    }
}

using PFN_CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
using PFN_CreateFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
PFN_CreateFactory o_CreateDXGIFactory;
PFN_CreateFactory2 o_CreateDXGIFactory2;

HRESULT WINAPI H_CreateDXGIFactory(REFIID riid, void** out) {
    HRESULT hr = o_CreateDXGIFactory(riid, out);
    if (SUCCEEDED(hr) && out) HookFactory(*out);
    return hr;
}

HRESULT WINAPI H_CreateDXGIFactory2(UINT flags, REFIID riid, void** out) {
    HRESULT hr = o_CreateDXGIFactory2(flags, riid, out);
    if (SUCCEEDED(hr) && out) HookFactory(*out);
    return hr;
}

// ---------------------------------------------------------------------------------------------
// input hooks: while the overlay is open the game must not move the camera or grab the cursor

using PFN_SetCursorPos = BOOL(WINAPI*)(int, int);
using PFN_ClipCursor = BOOL(WINAPI*)(const RECT*);
using PFN_GetRawInputData = UINT(WINAPI*)(HRAWINPUT, UINT, LPVOID, PUINT, UINT);
using PFN_SetWindowLongPtrW = LONG_PTR(WINAPI*)(HWND, int, LONG_PTR);
using PFN_GetWindowLongPtrW = LONG_PTR(WINAPI*)(HWND, int);
PFN_SetCursorPos o_SetCursorPos;
PFN_ClipCursor o_ClipCursor;
PFN_GetRawInputData o_GetRawInputData;
PFN_SetWindowLongPtrW o_SetWindowLongPtrW;
PFN_GetWindowLongPtrW o_GetWindowLongPtrW;

BOOL WINAPI H_SetCursorPos(int x, int y) { return OverlayVisible() ? TRUE : o_SetCursorPos(x, y); }

BOOL WINAPI H_ClipCursor(const RECT* r) { return o_ClipCursor(OverlayVisible() ? nullptr : r); }

UINT WINAPI H_GetRawInputData(HRAWINPUT input, UINT cmd, LPVOID data, PUINT size, UINT headerSize) {
    UINT r = o_GetRawInputData(input, cmd, data, size, headerSize);
    if (cmd != RID_INPUT || !data || r == (UINT)-1) return r;
    auto* ri = static_cast<RAWINPUT*>(data);
    if (ri->header.dwType == RIM_TYPEKEYBOARD) {
        // our hotkeys never reach the game (F9 would quickload, for instance); while the overlay is
        // open no key does
        USHORT vk = ri->data.keyboard.VKey;
        if (OverlayVisible() || vk == g_config.overlayKey || vk == g_config.toggleKey) {
            ri->data.keyboard.VKey = 0xFF;   // "no key"
            ri->data.keyboard.MakeCode = 0;
        }
    } else if (ri->header.dwType == RIM_TYPEMOUSE && OverlayVisible()) {
        ri->data.mouse.lLastX = ri->data.mouse.lLastY = 0;
        ri->data.mouse.usButtonFlags = 0;
    }
    return r;
}

LONG_PTR WINAPI H_SetWindowLongPtrW(HWND hwnd, int index, LONG_PTR value) {
    LONG_PTR previous;
    if (index == GWLP_WNDPROC && OverlayRedirectWndProc(hwnd, value, &previous)) return previous;
    return o_SetWindowLongPtrW(hwnd, index, value);
}

LONG_PTR WINAPI H_GetWindowLongPtrW(HWND hwnd, int index) {
    LONG_PTR proc;
    if (index == GWLP_WNDPROC && OverlayGameWndProc(hwnd, &proc)) return proc;
    return o_GetWindowLongPtrW(hwnd, index);
}

}  // namespace

bool InstallD3D12Hooks() {
    HMODULE exe = GetModuleHandleW(nullptr);
    int n = PatchImport(exe, "D3D12CreateDevice", (void*)H_D3D12CreateDeviceImport,
                        (void**)&o_D3D12CreateDeviceImport);
    Log("D3D12CreateDevice import patched in %d place(s)", n);
    int g = PatchImport(exe, "GetProcAddress", (void*)H_GetProcAddress, (void**)&o_GetProcAddress);
    if (HMODULE sl = GetModuleHandleW(L"sl.interposer.dll"))
        g += PatchImport(sl, "GetProcAddress", (void*)H_GetProcAddress, (void**)&o_GetProcAddress);
    Log("GetProcAddress import patched in %d place(s)", g);

    // settings overlay
    int f = PatchImport(exe, "CreateDXGIFactory", (void*)H_CreateDXGIFactory, (void**)&o_CreateDXGIFactory);
    f += PatchImport(exe, "CreateDXGIFactory2", (void*)H_CreateDXGIFactory2, (void**)&o_CreateDXGIFactory2);
    int in = PatchImport(exe, "SetCursorPos", (void*)H_SetCursorPos, (void**)&o_SetCursorPos);
    in += PatchImport(exe, "ClipCursor", (void*)H_ClipCursor, (void**)&o_ClipCursor);
    in += PatchImport(exe, "GetRawInputData", (void*)H_GetRawInputData, (void**)&o_GetRawInputData);
    in += PatchImport(exe, "SetWindowLongPtrW", (void*)H_SetWindowLongPtrW, (void**)&o_SetWindowLongPtrW);
    in += PatchImport(exe, "GetWindowLongPtrW", (void*)H_GetWindowLongPtrW, (void**)&o_GetWindowLongPtrW);
    Log("overlay: DXGI factory imports patched in %d place(s), input imports in %d", f, in);
    return n > 0 || g > 0;
}

void HookDeviceFromFfx(void* device) {
    if (device) HookDevice(device, "ffx backend");
}

void OnFrameBoundary() {
    uint32_t a = g_frameAlphaDraws.exchange(0), g = g_frameGbufDraws.exchange(0);
    uint32_t created = g_framePsosCreated.load();
    ++g_frame;
    OnHairFrameBoundary();
    if (g_config.logFrames && (g_frame <= 10 || g_frame % 1000 == 0))
        Log("frame %llu: hair alpha_accum draws=%u gbuffer draws=%u (known PSOs: %d alpha, %d gbuffer; "
            "PSOs seen so far: %u)",
            (unsigned long long)g_frame, a, g, g_numHair.load(), g_numHairGbuf.load(), created);
}

}  // namespace hf
