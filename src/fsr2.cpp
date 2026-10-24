// Built-in FSR 2.1: see fsr2.h.
#include "fsr2.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "capture.h"
#include "common.h"
#include "descriptors.h"
#include "hair.h"

namespace hf {
namespace {

// FSR 2.1 compute shaders of the game (2.x), from a [Debug] ProbeUpscaler capture
enum Pass { kOther, kLuminance, kAccumulate, kAccumulateFinal, kRcas, kPassCount };
constexpr uint64_t kLuminanceHash = 0x45e7974ee0f43114ull;    // compute luminance pyramid: reads the input colour
constexpr uint64_t kAccumulateHash = 0x09caa46ea4ac6fc9ull;   // accumulate when sharpening is on (RCAS follows)
constexpr uint64_t kAccumulateFinalHash = 0x75e26d49f7be43bbull;   // accumulate when sharpening is off: final output
// the accumulate pass's UAV table: internal colour (ping-pong), lock status, then (sharpening off) the output
constexpr UINT kFinalOutputSlot = 2;
constexpr uint64_t kRcasHash = 0x623f6b3044f56e94ull;         // RCAS: final output

constexpr int kMaxPsos = 8;
std::atomic<void*> g_psos[kPassCount][kMaxPsos];

int PassOf(void* pso) {
    if (!pso) return kOther;
    for (int p = kLuminance; p < kPassCount; ++p)
        for (int i = 0; i < kMaxPsos; ++i)
            if (g_psos[p][i].load(std::memory_order_relaxed) == pso) return p;
    return kOther;
}

// ---------------------------------------------------------------------------------------------
// compute bindings per command list (thread-local: a list is recorded by one thread at a time)

constexpr UINT kMaxParams = 16, kMaxConstants = 64;
enum ParamKind : uint8_t { kUnset, kTable, kConstants, kCbv, kSrv, kUav };

struct Param {
    ParamKind kind = kUnset;
    uint64_t value = 0;   // GPU handle / address
    UINT numConstants = 0;
    uint32_t constants[kMaxConstants];
};

struct Bindings {
    void* list = nullptr;
    ID3D12DescriptorHeap* heaps[2]{};
    UINT numHeaps = 0;
    ID3D12RootSignature* rootSig = nullptr;
    ID3D12PipelineState* pso = nullptr;
    int pass = kOther;
    bool overflow = false;   // a root parameter index the tracking cannot hold
    Param params[kMaxParams];
};

constexpr int kCached = 8;
thread_local Bindings t_bind[kCached];
thread_local int t_next = 0;
thread_local bool t_injecting = false;   // our own commands go through the hooks too

Bindings* Find(void* list) {
    for (auto& b : t_bind)
        if (b.list == list) return &b;
    return nullptr;
}

Bindings& Get(void* list) {
    if (Bindings* b = Find(list)) return *b;
    Bindings& b = t_bind[t_next++ % kCached];
    b.list = list;
    b.numHeaps = 0;
    b.rootSig = nullptr;
    b.pso = nullptr;
    b.pass = kOther;
    b.overflow = false;
    for (auto& p : b.params) p.kind = kUnset;
    return b;
}

Param* ParamOf(void* list, UINT index) {
    Bindings& b = Get(list);
    if (index >= kMaxParams) {
        b.overflow = true;
        return nullptr;
    }
    return &b.params[index];
}

// everything an insertion changes can be put back: heaps, compute root signature + arguments, PSO
bool Restorable(const Bindings& b) { return !b.overflow && b.numHeaps && b.rootSig && b.pso; }

void Restore(ID3D12GraphicsCommandList* cl, const Bindings& s) {
    if (s.numHeaps) cl->SetDescriptorHeaps(s.numHeaps, s.heaps);
    if (s.rootSig) {
        cl->SetComputeRootSignature(s.rootSig);
        for (UINT i = 0; i < kMaxParams; ++i) {
            const Param& p = s.params[i];
            switch (p.kind) {
                case kTable: cl->SetComputeRootDescriptorTable(i, D3D12_GPU_DESCRIPTOR_HANDLE{p.value}); break;
                case kConstants: cl->SetComputeRoot32BitConstants(i, p.numConstants, p.constants, 0); break;
                case kCbv: cl->SetComputeRootConstantBufferView(i, p.value); break;
                case kSrv: cl->SetComputeRootShaderResourceView(i, p.value); break;
                case kUav: cl->SetComputeRootUnorderedAccessView(i, p.value); break;
                default: break;
            }
        }
    }
    if (s.pso) cl->SetPipelineState(s.pso);
}

// first texture descriptor (SRV or UAV) at the start of a bound table, in root parameter order
ID3D12Resource* FirstTexture(const Bindings& b, bool uav, DescInfo* info) {
    for (UINT i = 0; i < kMaxParams; ++i) {
        if (b.params[i].kind != kTable) continue;
        DescInfo d;
        if (DescAt(b.params[i].value, 0, &d) && d.dim == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.uav == uav) {
            *info = d;
            return static_cast<ID3D12Resource*>(d.res);
        }
    }
    return nullptr;
}

void LogOnce(const char* msg) {
    static std::mutex m;
    static std::vector<const char*> seen;
    std::lock_guard<std::mutex> lock(m);
    for (auto* s : seen)
        if (s == msg) return;
    seen.push_back(msg);
    Log("%s", msg);
}

// ---------------------------------------------------------------------------------------------
// per-frame state (one FSR 2.1 invocation per frame)

std::mutex g_m;
std::atomic<void*> g_watchList{nullptr};    // list whose barriers reveal the colour's state (this frame)
ID3D12Resource* g_color = nullptr;          // input colour of the current frame
UINT g_renderW = 0, g_renderH = 0;
struct Learned {
    ID3D12Resource* color = nullptr;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};
Learned g_learned[4];                       // colour textures (the game may alternate) and their FSR state
int g_learnedNext = 0;
std::atomic<void*> g_openList{nullptr};     // resolved, post work (which clears the hair targets) not yet
std::atomic<void*> g_pendingList{nullptr};  // accumulate done, post work not yet
ID3D12Resource* g_pendingOutput = nullptr;

bool HavePass(int pass) {
    for (int i = 0; i < kMaxPsos; ++i)
        if (g_psos[pass][i].load(std::memory_order_relaxed)) return true;
    return false;
}

constexpr unsigned kFfxUav = 1 << 1;           // FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
constexpr unsigned kFfxComputeRead = 1 << 2;   // FFX_API_RESOURCE_STATE_COMPUTE_READ

void Transition(std::vector<D3D12_RESOURCE_BARRIER>& out, ID3D12Resource* r, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    out.push_back(b);
}

void Resolve(ID3D12GraphicsCommandList* cl, const Bindings& b) {
    DescInfo info;
    ID3D12Resource* color = FirstTexture(b, false, &info);
    if (!color) {
        LogOnce("fsr2: input colour not found in the luminance pass bindings");
        return;
    }
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    bool stateKnown = false;
    {
        std::lock_guard<std::mutex> lock(g_m);
        g_color = color;
        g_renderW = info.w;
        g_renderH = info.h;
        CaptureNoteRenderSize(info.w, info.h);
        for (const Learned& l : g_learned)
            if (l.color == color) {
                stateKnown = true;
                state = l.state;
            }
        g_watchList = cl;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        D3D12_RESOURCE_DESC d = color->GetDesc();
        Log("fsr2: built-in FSR 2.1 found: input colour %p %llux%u format %d", (void*)color,
            (unsigned long long)d.Width, d.Height, d.Format);
    }
    // without a known accumulate pass nothing would clear the hair targets after the resolve
    if (!HavePass(kAccumulate) && !HavePass(kAccumulateFinal)) {
        LogOnce("fsr2: accumulate pass unknown - resolve skipped");
        return;
    }
    UpscaleInputs in, out;
    in.color = color;
    in.colorState = kFfxComputeRead;   // the luminance pass reads it as a compute SRV
    bool resolved = ResolveBeforeUpscale(cl, in, out);
    if (resolved) g_openList = cl;
    if (!stateKnown) return;   // first frame: the colour's state is learned from the game's next barrier
    CaptureFrame(cl, color, state, nullptr, "fsr2");   // before the colour is overwritten
    if (!resolved || !out.color) return;
    // the resolved colour replaces the game's input in place, for every FSR pass that reads it
    auto* result = static_cast<ID3D12Resource*>(out.color);
    std::vector<D3D12_RESOURCE_BARRIER> bars;
    Transition(bars, color, state, D3D12_RESOURCE_STATE_COPY_DEST);
    Transition(bars, result, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier((UINT)bars.size(), bars.data());
    cl->CopyResource(color, result);
    for (auto& br : bars) std::swap(br.Transition.StateBefore, br.Transition.StateAfter);
    cl->ResourceBarrier((UINT)bars.size(), bars.data());
}

void PostWork(ID3D12GraphicsCommandList* cl, ID3D12Resource* output) {
    UINT w, h;
    {
        std::lock_guard<std::mutex> lock(g_m);
        w = g_renderW;
        h = g_renderH;
    }
    g_pendingList = nullptr;
    g_openList = nullptr;
    if (output) {
        CaptureOutput(cl, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);   // FSR's own output
        ApplyUnsharpMask(cl, output, kFfxUav, w, h);
        if (!g_ffxDevice) {
            ID3D12Device* dev = nullptr;
            if (SUCCEEDED(cl->GetDevice(IID_PPV_ARGS(&dev))) && dev) {
                g_ffxDevice = dev;   // lives as long as the game
                dev->Release();
            }
        }
        AfterUpscale(cl, output, kFfxUav);
    }
    FinishUpscale(cl);
    OnFrameBoundary();
}

}  // namespace

void Fsr2PsoCreated(void* pso, uint64_t csHash) {
    if (!pso) return;
    int pass = csHash == kLuminanceHash ? kLuminance : csHash == kAccumulateHash ? kAccumulate
             : csHash == kAccumulateFinalHash ? kAccumulateFinal : csHash == kRcasHash ? kRcas : kOther;
    // PSO addresses are reused after the game frees pipelines: drop stale entries
    for (int p = kLuminance; p < kPassCount; ++p)
        for (int i = 0; i < kMaxPsos; ++i)
            if (p != pass && g_psos[p][i].load() == pso) g_psos[p][i].store(nullptr);
    if (pass == kOther || PassOf(pso) == pass) return;
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    for (int i = 0; i < kMaxPsos; ++i) {
        void* expected = nullptr;
        if (g_psos[pass][i].compare_exchange_strong(expected, pso)) {
            Log("fsr2: pass %d PSO %p", pass, pso);
            return;
        }
    }
}

void TrackDescriptorHeaps(void* list, UINT n, ID3D12DescriptorHeap* const* heaps) {
    Bindings& b = Get(list);
    b.numHeaps = n > 2 ? 2 : n;
    for (UINT i = 0; i < b.numHeaps; ++i) b.heaps[i] = heaps[i];
}

void TrackComputeRootSignature(void* list, ID3D12RootSignature* rootSig) {
    Bindings& b = Get(list);
    b.rootSig = rootSig;
    b.overflow = false;
    for (auto& p : b.params) p.kind = kUnset;
}

void TrackPipelineState(void* list, ID3D12PipelineState* pso) {
    Bindings& b = Get(list);
    b.pso = pso;
    b.pass = PassOf(pso);
}

void TrackComputeTable(void* list, UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table) {
    if (Param* p = ParamOf(list, index)) {
        p->kind = kTable;
        p->value = table.ptr;
    }
}

void TrackComputeConstants(void* list, UINT index, UINT n, const void* data, UINT offset) {
    Param* p = ParamOf(list, index);
    if (!p) return;
    if (offset + n > kMaxConstants) {
        Get(list).overflow = true;
        return;
    }
    if (p->kind != kConstants) {
        p->kind = kConstants;
        p->numConstants = 0;
        memset(p->constants, 0, sizeof(p->constants));
    }
    memcpy(p->constants + offset, data, n * 4);
    if (offset + n > p->numConstants) p->numConstants = offset + n;
}

void TrackComputeRootView(void* list, UINT index, int kind, D3D12_GPU_VIRTUAL_ADDRESS va) {
    if (Param* p = ParamOf(list, index)) {
        p->kind = (ParamKind)kind;
        p->value = va;
    }
}

void Fsr2BeforeDispatch(void* list) {
    if (t_injecting) return;
    Bindings* b = Find(list);
    int pass = b ? b->pass : kOther;
    bool pending = g_pendingList.load(std::memory_order_relaxed) == list;
    if (pass != kLuminance && !(pending && pass != kRcas)) return;
    if (!b || !Restorable(*b)) {
        LogOnce("fsr2: game bindings not restorable - FSR 2.1 pass left alone");
        return;
    }
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(list);
    Bindings saved = *b;
    t_injecting = true;
    if (pending) {   // sharpening off: the pass after accumulate is the game's own
        ID3D12Resource* out;
        {
            std::lock_guard<std::mutex> lock(g_m);
            out = g_pendingOutput;
        }
        PostWork(cl, out);
    }
    if (pass == kLuminance) {
        if (g_openList.load() == list) PostWork(cl, nullptr);   // last frame's FSR never finished
        Resolve(cl, saved);
    }
    Restore(cl, saved);
    t_injecting = false;
}

// the final output of the accumulate pass without sharpening: kFinalOutputSlot of the table whose first
// descriptor is a texture UAV (the internal colour; same size and format as the output)
ID3D12Resource* FinalOutput(const Bindings& b) {
    for (UINT i = 0; i < kMaxParams; ++i) {
        if (b.params[i].kind != kTable) continue;
        DescInfo first, out;
        if (!DescAt(b.params[i].value, 0, &first) || !first.uav || first.dim != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
            continue;
        if (DescAt(b.params[i].value, kFinalOutputSlot, &out) && out.uav && out.dim == first.dim &&
            out.w == first.w && out.h == first.h && out.fmt == first.fmt)
            return static_cast<ID3D12Resource*>(out.res);
        LogOnce("fsr2: accumulate output not where expected - unsharp mask off");
        return nullptr;
    }
    return nullptr;
}

void Fsr2AfterDispatch(void* list) {
    if (t_injecting) return;
    Bindings* b = Find(list);
    if (!b || (b->pass != kAccumulate && b->pass != kAccumulateFinal && b->pass != kRcas)) return;
    DescInfo info;
    ID3D12Resource* out = b->pass == kAccumulateFinal ? FinalOutput(*b) : FirstTexture(*b, true, &info);
    if (b->pass == kAccumulate) {   // RCAS follows; without it the post work runs at the next dispatch
        std::lock_guard<std::mutex> lock(g_m);
        g_pendingOutput = out;
        g_pendingList = list;
        return;
    }
    // RCAS, or accumulate without sharpening: the final output
    if (!Restorable(*b)) {
        LogOnce("fsr2: game bindings not restorable - FSR 2.1 output left alone");
        return;
    }
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(list);
    Bindings saved = *b;
    t_injecting = true;
    PostWork(cl, out);
    Restore(cl, saved);
    t_injecting = false;
}

void Fsr2OnBarriers(void* list, UINT n, const D3D12_RESOURCE_BARRIER* barriers) {
    if (t_injecting || !barriers || g_watchList.load(std::memory_order_relaxed) != list) return;
    std::lock_guard<std::mutex> lock(g_m);
    if (!g_color) return;
    for (UINT i = 0; i < n; ++i) {
        const auto& br = barriers[i];
        if (br.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION || br.Transition.pResource != g_color) continue;
        // the first transition of the colour after the luminance pass leaves the state FSR read it in
        Learned* l = nullptr;
        for (Learned& e : g_learned)
            if (e.color == g_color) l = &e;
        if (!l) l = &g_learned[g_learnedNext++ % 4];
        if (l->color != g_color || l->state != br.Transition.StateBefore) {
            l->color = g_color;
            l->state = br.Transition.StateBefore;
            Log("fsr2: input colour %p is in state 0x%x during FSR", (void*)g_color, l->state);
        }
        g_color = nullptr;   // learn once per frame
        g_watchList = nullptr;
        return;
    }
}

void Fsr2BeforeClose(void* list) {
    if (t_injecting) return;
    bool pending = g_pendingList.load(std::memory_order_relaxed) == list;
    if (!pending && g_openList.load(std::memory_order_relaxed) != list) return;
    ID3D12Resource* out = nullptr;   // no accumulate seen: only clear the hair targets
    if (pending) {
        std::lock_guard<std::mutex> lock(g_m);
        out = g_pendingOutput;
    }
    t_injecting = true;
    PostWork(static_cast<ID3D12GraphicsCommandList*>(list), out);
    t_injecting = false;
}

}  // namespace hf
