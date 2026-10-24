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
#include "capture.h"
#include "descriptors.h"
#include "fsr2.h"
#include "hair.h"
#include "probe.h"
#include "pso.h"

namespace hf {
namespace {

// ---------------------------------------------------------------------------------------------
// vtable / IAT patching

#define VSLOT(Vtbl, Method) (offsetof(Vtbl, Method) / sizeof(void*))

enum Fn {
    kCreateRootSig, kCreateGPSO, kCreateCL, kCreatePSO, kCreateCL1, kCreateLib, kLoadGPipe, kLoadPipe,
    kSetPSO, kDrawIndexed, kDraw, kOMSetRT, kBeginRP, kEndRP,
    // built-in FSR 2.1: compute PSOs, descriptors, compute bindings
    kCreateCPSO, kLoadCPipe, kCreateHeap, kCreateSRV, kCreateUAV, kCopyDescs, kCopyDescsSimple,
    kDispatch, kSetCTable, kBarrier, kSetHeaps, kSetCRootSig, kSetCConst, kSetCConsts, kSetCCbv, kSetCSrv, kSetCUav,
    kClose,
    // upscaler probe only
    kCreateQueue, kExecute,
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
    // Already hooked once: if the slot now holds something else, another hook (Streamline, FFX,
    // an overlay) was installed over ours and chains to it. Re-hooking would store that hook as
    // our original and recurse forever.
    if (e->orig[fn]) return false;
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
    Fsr2PsoCreated(pso, 0);   // a graphics PSO may reuse a freed FSR pass PSO's address
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

void OnComputePsoCreated(void* pso, const D3D12_SHADER_BYTECODE& cs) {
    if (!pso || !cs.pShaderBytecode) return;
    uint64_t h = Hash64(cs.pShaderBytecode, cs.BytecodeLength);
    Fsr2PsoCreated(pso, h);
    if (ProbeEnabled()) ProbeComputePso(pso, h);
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
    TrackPipelineState(self, pso);
    if (ProbeCapturing()) ProbeSetPso(self, pso);
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

// compute work and bindings, for the built-in FSR 2.1 (fsr2.h) and the probe
using PFN_Dispatch = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using PFN_SetCTable = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_DESCRIPTOR_HANDLE);
using PFN_Barrier = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
using PFN_SetHeaps = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, ID3D12DescriptorHeap* const*);
using PFN_SetCRootSig = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12RootSignature*);
using PFN_SetCConst = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using PFN_SetCConsts = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT);
using PFN_SetCView = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, D3D12_GPU_VIRTUAL_ADDRESS);
using PFN_Close = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);

void STDMETHODCALLTYPE H_Dispatch(ID3D12GraphicsCommandList* self, UINT x, UINT y, UINT z) {
    Fsr2BeforeDispatch(self);
    ProbeDispatch(self, x, y, z);
    Orig<PFN_Dispatch>(self, kDispatch)(self, x, y, z);
    Fsr2AfterDispatch(self);
}

void STDMETHODCALLTYPE H_SetComputeRootDescriptorTable(ID3D12GraphicsCommandList* self, UINT index,
                                                        D3D12_GPU_DESCRIPTOR_HANDLE table) {
    TrackComputeTable(self, index, table);
    ProbeSetComputeTable(self, index, table);
    Orig<PFN_SetCTable>(self, kSetCTable)(self, index, table);
}

void STDMETHODCALLTYPE H_ResourceBarrier(ID3D12GraphicsCommandList* self, UINT n, const D3D12_RESOURCE_BARRIER* b) {
    Fsr2OnBarriers(self, n, b);
    ProbeBarriers(self, n, b);
    Orig<PFN_Barrier>(self, kBarrier)(self, n, b);
}

void STDMETHODCALLTYPE H_SetDescriptorHeaps(ID3D12GraphicsCommandList* self, UINT n, ID3D12DescriptorHeap* const* heaps) {
    TrackDescriptorHeaps(self, n, heaps);
    Orig<PFN_SetHeaps>(self, kSetHeaps)(self, n, heaps);
}

void STDMETHODCALLTYPE H_SetComputeRootSignature(ID3D12GraphicsCommandList* self, ID3D12RootSignature* rs) {
    TrackComputeRootSignature(self, rs);
    Orig<PFN_SetCRootSig>(self, kSetCRootSig)(self, rs);
}

void STDMETHODCALLTYPE H_SetComputeRoot32BitConstant(ID3D12GraphicsCommandList* self, UINT index, UINT value, UINT offset) {
    TrackComputeConstants(self, index, 1, &value, offset);
    Orig<PFN_SetCConst>(self, kSetCConst)(self, index, value, offset);
}

void STDMETHODCALLTYPE H_SetComputeRoot32BitConstants(ID3D12GraphicsCommandList* self, UINT index, UINT n,
                                                      const void* data, UINT offset) {
    TrackComputeConstants(self, index, n, data, offset);
    Orig<PFN_SetCConsts>(self, kSetCConsts)(self, index, n, data, offset);
}

void STDMETHODCALLTYPE H_SetComputeRootConstantBufferView(ID3D12GraphicsCommandList* self, UINT index,
                                                          D3D12_GPU_VIRTUAL_ADDRESS va) {
    TrackComputeRootView(self, index, 3, va);
    Orig<PFN_SetCView>(self, kSetCCbv)(self, index, va);
}

void STDMETHODCALLTYPE H_SetComputeRootShaderResourceView(ID3D12GraphicsCommandList* self, UINT index,
                                                          D3D12_GPU_VIRTUAL_ADDRESS va) {
    TrackComputeRootView(self, index, 4, va);
    Orig<PFN_SetCView>(self, kSetCSrv)(self, index, va);
}

void STDMETHODCALLTYPE H_SetComputeRootUnorderedAccessView(ID3D12GraphicsCommandList* self, UINT index,
                                                           D3D12_GPU_VIRTUAL_ADDRESS va) {
    TrackComputeRootView(self, index, 5, va);
    Orig<PFN_SetCView>(self, kSetCUav)(self, index, va);
}

HRESULT STDMETHODCALLTYPE H_Close(ID3D12GraphicsCommandList* self) {
    Fsr2BeforeClose(self);
    return Orig<PFN_Close>(self, kClose)(self);
}

void HookCommandList(void* list) {
    void** vt = VtableOf(list);
    bool any = false;
#define HOOK_CL(Method, fn) any |= HookSlot(vt, VSLOT(ID3D12GraphicsCommandListVtbl, Method), fn, (void*)H_##Method)
    HOOK_CL(Dispatch, kDispatch);
    HOOK_CL(SetComputeRootDescriptorTable, kSetCTable);
    HOOK_CL(ResourceBarrier, kBarrier);
    HOOK_CL(SetDescriptorHeaps, kSetHeaps);
    HOOK_CL(SetComputeRootSignature, kSetCRootSig);
    HOOK_CL(SetComputeRoot32BitConstant, kSetCConst);
    HOOK_CL(SetComputeRoot32BitConstants, kSetCConsts);
    HOOK_CL(SetComputeRootConstantBufferView, kSetCCbv);
    HOOK_CL(SetComputeRootShaderResourceView, kSetCSrv);
    HOOK_CL(SetComputeRootUnorderedAccessView, kSetCUav);
    HOOK_CL(Close, kClose);
#undef HOOK_CL
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
    if (SUCCEEDED(hr) && d && out && FromStream(d, p)) {
        if (p.ps.pShaderBytecode) OnPsoCreated(p, *out, "create", nullptr, d);
        if (p.cs.pShaderBytecode) OnComputePsoCreated(*out, p.cs);
    }
    return hr;
}

// command lists the hooks care about: direct ones, and compute ones too while probing
void OnCommandListCreated(void* list, D3D12_COMMAND_LIST_TYPE type) {
    if (type == D3D12_COMMAND_LIST_TYPE_DIRECT || (ProbeEnabled() && type == D3D12_COMMAND_LIST_TYPE_COMPUTE)) {
        HookCommandList(list);
        if (ProbeEnabled()) ProbeListCreated(list, type);
    }
}

HRESULT STDMETHODCALLTYPE H_CreateCommandList(ID3D12Device* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
                                              ID3D12CommandAllocator* alloc, ID3D12PipelineState* pso, REFIID riid,
                                              void** out) {
    HRESULT hr = Orig<PFN_CreateCL>(self, kCreateCL)(self, mask, type, alloc, pso, riid, out);
    if (SUCCEEDED(hr) && out && *out) OnCommandListCreated(*out, type);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateCommandList1(ID3D12Device4* self, UINT mask, D3D12_COMMAND_LIST_TYPE type,
                                               D3D12_COMMAND_LIST_FLAGS flags, REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_CreateCL1>(self, kCreateCL1)(self, mask, type, flags, riid, out);
    if (SUCCEEDED(hr) && out && *out) OnCommandListCreated(*out, type);
    return hr;
}

// ---------------------------------------------------------------------------------------------
// compute PSOs and descriptors (built-in FSR 2.1, see fsr2.h); queue submissions (probe only)

using PFN_CreateCPSO = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
using PFN_LoadCPipe = HRESULT(STDMETHODCALLTYPE*)(ID3D12PipelineLibrary*, LPCWSTR,
                                                  const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
using PFN_CreateHeap = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_DESCRIPTOR_HEAP_DESC*, REFIID, void**);
using PFN_CreateSRV = void(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*,
                                               D3D12_CPU_DESCRIPTOR_HANDLE);
using PFN_CreateUAV = void(STDMETHODCALLTYPE*)(ID3D12Device*, ID3D12Resource*, ID3D12Resource*,
                                               const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using PFN_CopyDescs = void(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, UINT,
                                               const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, D3D12_DESCRIPTOR_HEAP_TYPE);
using PFN_CopyDescsSimple = void(STDMETHODCALLTYPE*)(ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE,
                                                     D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
using PFN_CreateQueue = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMMAND_QUEUE_DESC*, REFIID, void**);
using PFN_Execute = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

HRESULT STDMETHODCALLTYPE H_CreateComputePipelineState(ID3D12Device* self, const D3D12_COMPUTE_PIPELINE_STATE_DESC* d,
                                                       REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_CreateCPSO>(self, kCreateCPSO)(self, d, riid, out);
    if (SUCCEEDED(hr) && d && out && *out) OnComputePsoCreated(*out, d->CS);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_LoadComputePipeline(ID3D12PipelineLibrary* self, LPCWSTR name,
                                                const D3D12_COMPUTE_PIPELINE_STATE_DESC* d, REFIID riid, void** out) {
    HRESULT hr = Orig<PFN_LoadCPipe>(self, kLoadCPipe)(self, name, d, riid, out);
    if (SUCCEEDED(hr) && d && out && *out) OnComputePsoCreated(*out, d->CS);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateDescriptorHeap(ID3D12Device* self, const D3D12_DESCRIPTOR_HEAP_DESC* d, REFIID riid,
                                                 void** out) {
    HRESULT hr = Orig<PFN_CreateHeap>(self, kCreateHeap)(self, d, riid, out);
    if (SUCCEEDED(hr) && out && *out && IsEqualIID(riid, IID_ID3D12DescriptorHeap)) DescOnHeapCreated(*out);
    return hr;
}

void STDMETHODCALLTYPE H_CreateShaderResourceView(ID3D12Device* self, ID3D12Resource* res,
                                                  const D3D12_SHADER_RESOURCE_VIEW_DESC* d, D3D12_CPU_DESCRIPTOR_HANDLE dst) {
    Orig<PFN_CreateSRV>(self, kCreateSRV)(self, res, d, dst);
    DescOnView(res, dst, false, d ? d->Format : DXGI_FORMAT_UNKNOWN);
}

void STDMETHODCALLTYPE H_CreateUnorderedAccessView(ID3D12Device* self, ID3D12Resource* res, ID3D12Resource* counter,
                                                   const D3D12_UNORDERED_ACCESS_VIEW_DESC* d,
                                                   D3D12_CPU_DESCRIPTOR_HANDLE dst) {
    Orig<PFN_CreateUAV>(self, kCreateUAV)(self, res, counter, d, dst);
    DescOnView(res, dst, true, d ? d->Format : DXGI_FORMAT_UNKNOWN);
}

void STDMETHODCALLTYPE H_CopyDescriptors(ID3D12Device* self, UINT numDst, const D3D12_CPU_DESCRIPTOR_HANDLE* dst,
                                         const UINT* dstSizes, UINT numSrc, const D3D12_CPU_DESCRIPTOR_HANDLE* src,
                                         const UINT* srcSizes, D3D12_DESCRIPTOR_HEAP_TYPE type) {
    Orig<PFN_CopyDescs>(self, kCopyDescs)(self, numDst, dst, dstSizes, numSrc, src, srcSizes, type);
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) DescOnCopy(numDst, dst, dstSizes, numSrc, src, srcSizes);
}

void STDMETHODCALLTYPE H_CopyDescriptorsSimple(ID3D12Device* self, UINT n, D3D12_CPU_DESCRIPTOR_HANDLE dst,
                                               D3D12_CPU_DESCRIPTOR_HANDLE src, D3D12_DESCRIPTOR_HEAP_TYPE type) {
    Orig<PFN_CopyDescsSimple>(self, kCopyDescsSimple)(self, n, dst, src, type);
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) DescOnCopySimple(n, dst, src);
}

void STDMETHODCALLTYPE H_ExecuteCommandLists(ID3D12CommandQueue* self, UINT n, ID3D12CommandList* const* lists) {
    ProbeExecute(self, n, reinterpret_cast<void* const*>(lists));
    Orig<PFN_Execute>(self, kExecute)(self, n, lists);
}

HRESULT STDMETHODCALLTYPE H_CreateCommandQueue(ID3D12Device* self, const D3D12_COMMAND_QUEUE_DESC* d, REFIID riid,
                                               void** out) {
    HRESULT hr = Orig<PFN_CreateQueue>(self, kCreateQueue)(self, d, riid, out);
    if (SUCCEEDED(hr) && d && out && *out) {
        Log("probe: command queue %p type=%d", *out, d->Type);
        HookSlot(VtableOf(*out), VSLOT(ID3D12CommandQueueVtbl, ExecuteCommandLists), kExecute,
                 (void*)H_ExecuteCommandLists);
    }
    return hr;
}

int HookDeviceCompute(ID3D12Device* dev) {
    void** vt = VtableOf(dev);
    int n = 0;
    if (ProbeEnabled())
        n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateCommandQueue), kCreateQueue, (void*)H_CreateCommandQueue);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateComputePipelineState), kCreateCPSO,
                  (void*)H_CreateComputePipelineState);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateDescriptorHeap), kCreateHeap, (void*)H_CreateDescriptorHeap);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateShaderResourceView), kCreateSRV, (void*)H_CreateShaderResourceView);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CreateUnorderedAccessView), kCreateUAV,
                  (void*)H_CreateUnorderedAccessView);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CopyDescriptors), kCopyDescs, (void*)H_CopyDescriptors);
    n += HookSlot(vt, VSLOT(ID3D12DeviceVtbl, CopyDescriptorsSimple), kCopyDescsSimple, (void*)H_CopyDescriptorsSimple);
    return n;
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
    if (SUCCEEDED(hr) && d && out && FromStream(d, p)) {
        if (p.ps.pShaderBytecode) OnPsoCreated(p, *out, "library", nullptr, d);
        if (p.cs.pShaderBytecode) OnComputePsoCreated(*out, p.cs);
    }
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
            HookSlot(VtableOf(lib), VSLOT(ID3D12PipelineLibraryVtbl, LoadComputePipeline), kLoadCPipe,
                     (void*)H_LoadComputePipeline);
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
    n += HookDeviceCompute(dev);

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

// "module+offset" of a code / data address, for diagnostics
std::string Where(const void* p) {
    HMODULE m = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<const char*>(p), &m) && m) {
        GetModuleFileNameA(m, name, MAX_PATH);
        const char* base = strrchr(name, '\\');
        char buf[MAX_PATH + 32];
        snprintf(buf, sizeof(buf), "%s+0x%llx", base ? base + 1 : name,
                 (unsigned long long)(static_cast<const uint8_t*>(p) - reinterpret_cast<uint8_t*>(m)));
        return buf;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%p", p);
    return buf;
}

// Swap chain calls can nest on one thread (dxgi.dll may forward Present / ResizeBuffers to their
// "1" variants through the vtable). Only the outermost call does overlay work; otherwise the
// overlay locks its (non-recursive) mutex twice on the same thread and the game hangs.
thread_local int t_dxgiDepth = 0;

struct DxgiScope {
    bool outer;
    DxgiScope() : outer(t_dxgiDepth++ == 0) {}
    ~DxgiScope() { --t_dxgiDepth; }
};

HRESULT STDMETHODCALLTYPE H_Present(IDXGISwapChain* self, UINT sync, UINT flags) {
    DxgiScope scope;
    if (scope.outer && !(flags & DXGI_PRESENT_TEST)) {
        ProbePresent();
        PerfOnPresent();
        CapturePollKey();
        OverlayOnPresent(self);
    }
    return Orig<PFN_Present>(self, kPresent)(self, sync, flags);
}

HRESULT STDMETHODCALLTYPE H_Present1(IDXGISwapChain1* self, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* p) {
    DxgiScope scope;
    if (scope.outer && !(flags & DXGI_PRESENT_TEST)) {
        ProbePresent();
        PerfOnPresent();
        CapturePollKey();
        OverlayOnPresent(self);
    }
    return Orig<PFN_Present1>(self, kPresent1)(self, sync, flags, p);
}

HRESULT STDMETHODCALLTYPE H_ResizeBuffers(IDXGISwapChain* self, UINT n, UINT w, UINT h, DXGI_FORMAT f, UINT flags) {
    DxgiScope scope;
    if (scope.outer) OverlayBeforeResize();
    return Orig<PFN_ResizeBuffers>(self, kResizeBuffers)(self, n, w, h, f, flags);
}

HRESULT STDMETHODCALLTYPE H_ResizeBuffers1(IDXGISwapChain3* self, UINT n, UINT w, UINT h, DXGI_FORMAT f, UINT flags,
                                           const UINT* masks, IUnknown* const* queues) {
    DxgiScope scope;
    if (scope.outer) OverlayBeforeResize();
    return Orig<PFN_ResizeBuffers1>(self, kResizeBuffers1)(self, n, w, h, f, flags, masks, queues);
}

void HookSwapChainImpl(void* swapChain, IUnknown* queue, HWND hwnd) {
    void** vt = VtableOf(swapChain);
    Log("swap chain %p: hooking vtable %s", swapChain, Where(vt).c_str());
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

// Only the swap chain the game itself talks to is hooked. Swap chains created underneath it (the
// native one behind a Streamline proxy, the real one inside FSR frame generation) are left alone:
// overlays such as Steam's inline-hook dxgi.dll's own code and call back through the vtable, so a
// vtable hook on the native swap chain recurses forever.
thread_local int t_scCreateDepth = 0;
thread_local int t_ffxCreateDepth = 0;   // inside ffxCreateContext (see proxy.cpp)

HRESULT STDMETHODCALLTYPE H_CreateSwapChainForHwnd(IDXGIFactory2* self, IUnknown* queue, HWND hwnd,
                                                   const DXGI_SWAP_CHAIN_DESC1* desc,
                                                   const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fs, IDXGIOutput* output,
                                                   IDXGISwapChain1** out) {
    ++t_scCreateDepth;
    HRESULT hr = Orig<PFN_CreateSCHwnd>(self, kCreateSCHwnd)(self, queue, hwnd, desc, fs, output, out);
    --t_scCreateDepth;
    if (SUCCEEDED(hr) && out && *out && !t_scCreateDepth && !t_ffxCreateDepth) HookSwapChainImpl(*out, queue, hwnd);
    return hr;
}

HRESULT STDMETHODCALLTYPE H_CreateSwapChain(IDXGIFactory* self, IUnknown* queue, DXGI_SWAP_CHAIN_DESC* desc,
                                            IDXGISwapChain** out) {
    ++t_scCreateDepth;
    HRESULT hr = Orig<PFN_CreateSC>(self, kCreateSC)(self, queue, desc, out);
    --t_scCreateDepth;
    if (SUCCEEDED(hr) && out && *out && desc && !t_scCreateDepth && !t_ffxCreateDepth)
        HookSwapChainImpl(*out, queue, desc->OutputWindow);
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
        if (OverlayVisible() || vk == g_config.overlayKey || vk == g_config.toggleKey ||
            (g_config.captureKey && vk == g_config.captureKey)) {
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

void FfxCreateBegin() { ++t_ffxCreateDepth; }
void FfxCreateEnd() { --t_ffxCreateDepth; }

void HookFfxSwapChain(void* swapChain) {
    if (!swapChain) return;
    HWND hwnd = nullptr;
    IDXGISwapChain1* sc1 = nullptr;
    if (SUCCEEDED(IUnknown_QueryInterface((IUnknown*)swapChain, IID_IDXGISwapChain1, (void**)&sc1)) && sc1) {
        IDXGISwapChain1_GetHwnd(sc1, &hwnd);
        IUnknown_Release((IUnknown*)sc1);
    }
    HookSwapChainImpl(swapChain, nullptr, hwnd);
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
