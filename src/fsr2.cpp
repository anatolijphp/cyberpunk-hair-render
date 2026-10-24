// Built-in FSR 2.1: see fsr2.h.
#include "fsr2.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "capture.h"
#include "common.h"
#include "descriptors.h"
#include "hair.h"

namespace hf {
namespace {

// FSR 2.1 compute shaders of the game (2.x), from a [Debug] ProbeUpscaler capture
enum Pass { kOther, kLuminance, kAccumulate, kAccumulateFinal, kRcas, kMirror, kPassCount };
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

void AddPassPso(int pass, void* pso) {
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
    uint64_t psoHash = 0;    // its compute shader's hash (0: graphics / unknown)
    int pass = kOther;
    bool overflow = false;   // a root parameter index the tracking cannot hold
    Param params[kMaxParams];
    ListGraphics gfx;
};

thread_local bool t_injecting = false;   // our own commands go through the hooks too

// every command list's bindings, in one table: a list may be recorded by different threads (one at a
// time) over its life, so per-thread caches would see stale or missing state. Slots are never freed:
// a list is (re)initialised on creation and Reset, also when a new list gets a freed one's address.
constexpr size_t kListSlots = 2048;   // power of two
struct ListSlot {
    std::atomic<void*> list{nullptr};
    Bindings b;
};
ListSlot g_listSlots[kListSlots];
thread_local Bindings t_overflowSlot;   // the table is full: tracked here, never restorable

size_t SlotHash(void* list) { return (size_t)(((uintptr_t)list >> 4) * 0x9E3779B97F4A7C15ull >> 53) & (kListSlots - 1); }

Bindings* Find(void* list) {
    if (!list) return nullptr;
    const size_t h = SlotHash(list);
    for (size_t i = 0; i < kListSlots; ++i) {
        ListSlot& s = g_listSlots[(h + i) & (kListSlots - 1)];
        void* l = s.list.load(std::memory_order_acquire);
        if (l == list) return &s.b;
        if (!l) return nullptr;
    }
    return nullptr;
}

void ResetBindings(Bindings& b, void* list) {
    b.list = list;
    b.numHeaps = 0;
    b.rootSig = nullptr;
    b.pso = nullptr;
    b.psoHash = 0;
    b.pass = kOther;
    b.overflow = false;
    for (auto& p : b.params) p.kind = kUnset;
    b.gfx = ListGraphics{};
}

Bindings& Get(void* list) {
    if (Bindings* b = Find(list)) return *b;
    const size_t h = SlotHash(list);
    for (size_t i = 0; i < kListSlots; ++i) {
        ListSlot& s = g_listSlots[(h + i) & (kListSlots - 1)];
        void* expected = nullptr;
        if (s.list.compare_exchange_strong(expected, list, std::memory_order_acq_rel)) {
            ResetBindings(s.b, list);
            return s.b;
        }
        if (expected == list) return s.b;   // another thread inserted it (cannot happen while recording)
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        Log("fsr2: more than %zu command lists - bindings of further ones are not tracked", kListSlots);
    }
    ResetBindings(t_overflowSlot, list);
    t_overflowSlot.overflow = true;   // never restorable
    return t_overflowSlot;
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
Learned g_learned[8];                       // colour textures (the game may alternate) and their state
int g_learnedNext = 0;

// mirrors (hair.h): the mirror view's final compute pass reads its lit colour (an RGBA16F texture of
// the view's size) and writes the image the mirror shows (another one). While the mirror's hair is
// drawn, dispatches with those bindings are candidates; the one shader that is the only candidate in
// a few frames is the pass, recognised by its hash from then on.
std::shared_mutex g_hashMutex;
std::unordered_map<void*, uint64_t> g_csHashes;   // compute PSO -> shader hash

uint64_t HashOf(void* pso) {   // 0: not a compute PSO (or unknown)
    if (!pso) return 0;
    std::shared_lock<std::shared_mutex> lock(g_hashMutex);
    auto it = g_csHashes.find(pso);
    return it != g_csHashes.end() ? it->second : 0;
}
constexpr uint64_t kMirrorHash = 0xb06e565d137c14f4ull;   // the game's (2.x), found this way
std::atomic<uint64_t> g_mirrorHash{kMirrorHash};
constexpr int kMirrorFrames = 3;

bool StateOf(ID3D12Resource* color, D3D12_RESOURCE_STATES* state) {   // g_m held
    for (const Learned& l : g_learned)
        if (l.color == color) {
            *state = l.state;
            return true;
        }
    return false;
}
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
        stateKnown = StateOf(color, &state);
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

// mirror candidates of the current frame and the shader that was the only one in the last frames (g_m)
uint64_t g_candFrame = 0, g_candLast = 0;
int g_candCount = 0;
std::unordered_set<uint64_t> g_cands;

// while the mirror's hair is drawn: is this dispatch the mirror's final pass (one RGBA16F texture of
// the mirror's size read, one written)?
void LearnMirror(const Bindings& b) {
    if (HavePass(kMirror)) return;   // known (by default its hash; else learned below)
    const uint32_t size = MirrorSize();
    if (!size) return;
    const UINT w = size >> 16, h = size & 0xffff;
    DescInfo in, out;
    if (!FirstTexture(b, false, &in) || !FirstTexture(b, true, &out) || in.res == out.res) return;
    auto fits = [&](const DescInfo& d) {
        return d.w == w && d.h == h && d.fmt == DXGI_FORMAT_R16G16B16A16_FLOAT;
    };
    if (!fits(in) || !fits(out)) return;
    const uint64_t hash = b.psoHash;
    if (!hash) return;
    std::lock_guard<std::mutex> lock(g_m);
    const uint64_t frame = HairFrame();
    if (frame != g_candFrame) {   // the last frame's verdict: exactly one candidate
        if (g_cands.size() == 1) {
            uint64_t only = *g_cands.begin();
            g_candCount = only == g_candLast ? g_candCount + 1 : 1;
            g_candLast = only;
        } else {
            if (g_cands.size() > 1) Log("mirror: %zu candidate passes in one frame, waiting", g_cands.size());
            g_candCount = 0;
        }
        g_cands.clear();
        g_candFrame = frame;
        if (g_candCount >= kMirrorFrames) {
            g_mirrorHash = g_candLast;
            Log("mirror: final pass found: shader %016llx, colour %ux%u", (unsigned long long)g_candLast, w, h);
            std::shared_lock<std::shared_mutex> hl(g_hashMutex);
            for (const auto& e : g_csHashes)
                if (e.second == g_candLast) AddPassPso(kMirror, e.first);
            return;
        }
    }
    g_cands.insert(hash);
}

std::atomic<uint32_t> g_mirrorPassCalls{0}, g_mirrorNoLayout{0};

// where a root signature's first UAV sits: root parameter (a descriptor table) and descriptor offset
// in it - the mirror pass's output image (from the root signature itself: valid while recording)
struct UavLocation {
    bool known = false, found = false;
    UINT param = 0, offset = 0;
};
std::mutex g_layoutMutex;
std::unordered_map<void*, UavLocation> g_layouts;

UavLocation FirstUavLocation(ID3D12RootSignature* rootSig) {
    {
        std::lock_guard<std::mutex> lock(g_layoutMutex);
        auto it = g_layouts.find(rootSig);
        if (it != g_layouts.end()) return it->second;
    }
    UavLocation loc;
    loc.known = true;
    std::vector<uint8_t> blob;
    using PFN_Deserializer = HRESULT(WINAPI*)(LPCVOID, SIZE_T, REFIID, void**);
    static auto create = reinterpret_cast<PFN_Deserializer>(
        GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12CreateVersionedRootSignatureDeserializer"));
    ID3D12VersionedRootSignatureDeserializer* des = nullptr;
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
    if (rootSig && RootSignatureBlob(rootSig, &blob) && create &&
        SUCCEEDED(create(blob.data(), blob.size(), IID_PPV_ARGS(&des))) && des &&
        SUCCEEDED(des->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc)) && desc) {
        const D3D12_ROOT_SIGNATURE_DESC1& d = desc->Desc_1_1;
        for (UINT i = 0; i < d.NumParameters && !loc.found; ++i) {
            const D3D12_ROOT_PARAMETER1& p = d.pParameters[i];
            if (p.ParameterType != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) continue;
            UINT next = 0;
            for (UINT r = 0; r < p.DescriptorTable.NumDescriptorRanges; ++r) {
                const D3D12_DESCRIPTOR_RANGE1& range = p.DescriptorTable.pDescriptorRanges[r];
                UINT offset = range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
                                  ? next : range.OffsetInDescriptorsFromTableStart;
                if (range.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_UAV) {
                    loc.found = true;
                    loc.param = i;
                    loc.offset = offset;
                    break;
                }
                next = offset + range.NumDescriptors;
            }
        }
    }
    if (des) des->Release();
    Log("mirror: final pass root signature %p: output %s (parameter %u, descriptor %u)", (void*)rootSig,
        loc.found ? "found" : "not found", loc.param, loc.offset);
    std::lock_guard<std::mutex> lock(g_layoutMutex);
    g_layouts[rootSig] = loc;
    return loc;
}

// right after the mirror's final pass: the resolve of its output (hair.cpp)
void MirrorAfterDispatch(ID3D12GraphicsCommandList* cl, const Bindings& b) {
    g_mirrorPassCalls++;
    const UavLocation loc = FirstUavLocation(b.rootSig);
    if (!loc.found || loc.param >= kMaxParams || b.params[loc.param].kind != kTable) {
        g_mirrorNoLayout++;
        return;
    }
    MirrorAfterPass(cl, b.params[loc.param].value, loc.offset);
}

// the first transition of `color` in a watched list leaves the state the pass read it in (g_m held)
bool LearnState(ID3D12Resource*& color, UINT n, const D3D12_RESOURCE_BARRIER* barriers, const char* what) {
    if (!color) return false;
    for (UINT i = 0; i < n; ++i) {
        const auto& br = barriers[i];
        if (br.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION || br.Transition.pResource != color) continue;
        Learned* l = nullptr;
        for (Learned& e : g_learned)
            if (e.color == color) l = &e;
        if (!l) l = &g_learned[g_learnedNext++ % 8];
        if (l->color != color || l->state != br.Transition.StateBefore) {
            l->color = color;
            l->state = br.Transition.StateBefore;
            Log("%s %p is in state 0x%x during its pass", what, (void*)color, l->state);
        }
        color = nullptr;   // learn once per frame
        return true;
    }
    return false;
}

}  // namespace

void MirrorPassStats(char* buf, size_t size) {
    snprintf(buf, size, "mirror shader %016llx, pass dispatches %u, no output in its root signature %u",
             (unsigned long long)g_mirrorHash.load(), g_mirrorPassCalls.exchange(0), g_mirrorNoLayout.exchange(0));
}

void Fsr2PsoCreated(void* pso, uint64_t csHash) {
    if (!pso) return;
    const uint64_t mirror = g_mirrorHash.load();
    {   // every compute PSO's shader hash (for recognising the mirror pass at its dispatch)
        std::unique_lock<std::shared_mutex> lock(g_hashMutex);
        if (csHash) g_csHashes[pso] = csHash;
        else g_csHashes.erase(pso);   // a graphics PSO at a freed compute PSO's address
    }
    int pass = csHash == kLuminanceHash ? kLuminance : csHash == kAccumulateHash ? kAccumulate
             : csHash == kAccumulateFinalHash ? kAccumulateFinal : csHash == kRcasHash ? kRcas
             : csHash && csHash == mirror ? kMirror : kOther;
    // PSO addresses are reused after the game frees pipelines: drop stale entries
    for (int p = kLuminance; p < kPassCount; ++p)
        for (int i = 0; i < kMaxPsos; ++i)
            if (p != pass && g_psos[p][i].load() == pso) g_psos[p][i].store(nullptr);
    AddPassPso(pass, pso);
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

void TrackListReset(void* list, ID3D12PipelineState* pso) {
    Bindings& b = Get(list);
    ResetBindings(b, list);
    b.pso = pso;
    b.psoHash = HashOf(pso);
    b.pass = PassOf(pso);
}

ListGraphics* GraphicsOf(void* list, bool create) {
    if (create) return &Get(list).gfx;
    Bindings* b = Find(list);
    return b ? &b->gfx : nullptr;
}

void TrackPipelineState(void* list, ID3D12PipelineState* pso) {
    Bindings& b = Get(list);
    b.pso = pso;
    b.psoHash = HashOf(pso);
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
    // the mirror's final pass by the bound shader itself; if its hash is unknown (a patch), learned
    // while other views' hair is drawn
    if (b && pass == kOther) {
        if (b->psoHash && b->psoHash == g_mirrorHash.load(std::memory_order_relaxed)) return;   // after it
        if (MirrorSize()) LearnMirror(*b);
    }
    if (pass == kMirror) return;   // the work goes after the dispatch
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
    if (b && (b->pass == kMirror || (b->psoHash && b->psoHash == g_mirrorHash.load(std::memory_order_relaxed)))) {
        if (!Restorable(*b)) {
            LogOnce("mirror: game bindings not restorable - mirror left alone");
            return;
        }
        auto* cl = static_cast<ID3D12GraphicsCommandList*>(list);
        Bindings saved = *b;
        t_injecting = true;
        MirrorAfterDispatch(cl, saved);
        Restore(cl, saved);
        t_injecting = false;
        return;
    }
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
    if (t_injecting || !barriers) return;
    if (g_watchList.load(std::memory_order_relaxed) != list) return;
    std::lock_guard<std::mutex> lock(g_m);
    // the first transition of the colour after the luminance pass leaves the state FSR read it in
    if (LearnState(g_color, n, barriers, "fsr2: input colour")) g_watchList = nullptr;
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
