// Hair fix: side coverage pass (re-issued hair alpha_accum draws into our own targets) and the
// guided resolve on the upscaler's colour input.
#include "hair.h"
#include "neural.h"

#include <atomic>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "pso.h"
#include "../build/gen/coverage_table.h"
#include "../build/gen/resolve_cso.h"
#include "../build/gen/resolve_tensor_cso.h"
#include "../build/gen/area_cso.h"
#include "../build/gen/area_h_cso.h"

namespace hf {
namespace {

// ---------------------------------------------------------------------------------------------
// side PSOs

// Raw coverage shader for a game variant, or the sharpen preset nearest to (threshold, width).
const CoverageShader* FindCoverageShader(uint64_t gameHash, bool sharpen, float threshold, float width) {
    const CoverageShader* best = nullptr;
    float bestDist = 1e9f;
    for (const auto& c : kCoverageShaders) {
        if (c.gameHash != gameHash || (c.threshold >= 0) != sharpen) continue;
        float dist = sharpen ? (c.threshold - threshold) * (c.threshold - threshold) * 16 +
                                   (c.width - width) * (c.width - width)
                             : 0;
        if (dist < bestDist) {
            bestDist = dist;
            best = &c;
        }
    }
    return best;
}

// Per game alpha_accum PSO: a deep copy of its desc (so side PSOs for any coverage preset can be
// built later, when the setting changes) and the side PSOs built so far:
// [0] raw coverage, [1 + i] the i-th sharpen preset of that variant in kCoverageShaders.
constexpr int kPresets = 17;
struct DescCopy {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    std::vector<uint8_t> vs, hs, ds, gs;
    std::vector<D3D12_INPUT_ELEMENT_DESC> layout;
    std::vector<std::string> names;
};
struct Slot {
    std::atomic<void*> game{nullptr};
    int rawIndex = 0;                  // index of the variant's raw entry in kCoverageShaders
    DescCopy* desc = nullptr;
    std::atomic<ID3D12PipelineState*> side[kPresets];
};
constexpr int kMaxPairs = 256;
Slot g_slots[kMaxPairs];
std::atomic<int> g_numPairs{0};

DescCopy* CopyDesc(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& src) {
    auto* c = new DescCopy;
    c->d = src;
    auto copy = [](std::vector<uint8_t>& v, D3D12_SHADER_BYTECODE& b) {
        if (!b.pShaderBytecode) return;
        auto* p = static_cast<const uint8_t*>(b.pShaderBytecode);
        v.assign(p, p + b.BytecodeLength);
        b.pShaderBytecode = v.data();
    };
    copy(c->vs, c->d.VS);
    copy(c->hs, c->d.HS);
    copy(c->ds, c->d.DS);
    copy(c->gs, c->d.GS);
    c->d.PS = {};
    c->d.StreamOutput = {};
    c->d.CachedPSO = {};
    c->names.reserve(src.InputLayout.NumElements);   // stable c_str() pointers below
    for (UINT i = 0; i < src.InputLayout.NumElements; ++i) {
        D3D12_INPUT_ELEMENT_DESC e = src.InputLayout.pInputElementDescs[i];
        c->names.emplace_back(e.SemanticName ? e.SemanticName : "");
        c->layout.push_back(e);
    }
    for (size_t i = 0; i < c->layout.size(); ++i) c->layout[i].SemanticName = c->names[i].c_str();
    c->d.InputLayout = {c->layout.data(), (UINT)c->layout.size()};
    if (c->d.pRootSignature) c->d.pRootSignature->AddRef();   // kept for the lifetime of the process
    return c;
}

void MakeSideDesc(D3D12_GRAPHICS_PIPELINE_STATE_DESC& d, const CoverageShader& v) {
    d.PS = {v.bytes, (SIZE_T)v.size};
    d.NumRenderTargets = 2;
    for (auto& f : d.RTVFormats) f = DXGI_FORMAT_UNKNOWN;
    d.RTVFormats[0] = DXGI_FORMAT_R8_UNORM;
    d.RTVFormats[1] = DXGI_FORMAT_R8_UNORM;
    d.BlendState = {};
    d.BlendState.IndependentBlendEnable = TRUE;
    auto& c = d.BlendState.RenderTarget[0];   // transmittance: dst * (1 - a)
    c = {TRUE, FALSE, D3D12_BLEND_ZERO, D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_OP_ADD,
         D3D12_BLEND_ZERO, D3D12_BLEND_ONE, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_RED};
    auto& k = d.BlendState.RenderTarget[1];   // hair-surface mask: max
    k = {TRUE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ONE, D3D12_BLEND_OP_MAX,
         D3D12_BLEND_ONE, D3D12_BLEND_ONE, D3D12_BLEND_OP_MAX, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_RED};
    d.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    d.DepthStencilState.StencilEnable = FALSE;
    d.CachedPSO = {};
    d.StreamOutput = {};
}

// ---------------------------------------------------------------------------------------------
// render-target / render-pass tracking (per recording thread)

struct RtState {
    void* list = nullptr;
    UINT n = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE rts[8]{};
    BOOL single = FALSE;
    bool hasDsv = false;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
};
thread_local RtState t_rt;
thread_local void* t_passList = nullptr;
thread_local bool t_inPass = false;

// ---------------------------------------------------------------------------------------------
// resources

struct Targets {
    UINT w = 0, h = 0;
    DXGI_FORMAT colorFormat = DXGI_FORMAT_UNKNOWN;
    ID3D12Resource* cov = nullptr;      // R8 transmittance (RT, cleared to 1)
    ID3D12Resource* kept = nullptr;     // R8 hair-surface mask (RT, cleared to 0)
    ID3D12Resource* tensor = nullptr;   // RGBA16F strand direction / coherence (UAV)
    ID3D12Resource* output = nullptr;   // resolved colour handed to FSR
    ID3D12Resource* depthOut = nullptr; // stabilised depth (R32F)
    ID3D12Resource* motionOut = nullptr;// stabilised motion vectors
    DXGI_FORMAT motionFormat = DXGI_FORMAT_UNKNOWN;
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[2]{};
    D3D12_RESOURCE_STATES tensorState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES outputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES auxState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;   // depthOut + motionOut
    // Cleared once by the first side draw; afterwards by every resolve, right after reading them,
    // so the next frame's hair always lands on cleared targets in GPU order.
    std::atomic<bool> needsInitialClear{true};
    bool areaValid = false;             // tensor .y holds this frame's Area map
};

std::mutex g_resMutex;
ID3D12Device* g_device = nullptr;
Targets* g_targets = nullptr;
std::vector<std::pair<uint64_t, Targets*>> g_graveyard;   // (frame retired, targets)

std::atomic<uint64_t> g_frameNo{1};   // only used to age retired targets
std::atomic<UINT> g_colorW{0}, g_colorH{0};
std::atomic<int> g_colorFormat{0};
std::atomic<int> g_motionFormat{0};

ID3D12Resource* MakeTexture(UINT w, UINT h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags,
                            D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    ID3D12Resource* r = nullptr;
    HRESULT hr = g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, clear, IID_PPV_ARGS(&r));
    if (FAILED(hr)) Log("hair: CreateCommittedResource %ux%u fmt=%d failed 0x%lx", w, h, fmt, hr);
    return r;
}

void Release(Targets* t) {
    if (!t) return;
    for (IUnknown* u : {(IUnknown*)t->cov, (IUnknown*)t->kept, (IUnknown*)t->tensor, (IUnknown*)t->output,
                        (IUnknown*)t->depthOut, (IUnknown*)t->motionOut, (IUnknown*)t->rtvHeap})
        if (u) u->Release();
    delete t;
}

// current targets matching the last known colour size (created / resized on demand)
Targets* CurrentTargets() {
    UINT w = g_colorW.load(), h = g_colorH.load();
    if (!g_device || !w || !h) return nullptr;
    std::lock_guard<std::mutex> lock(g_resMutex);
    uint64_t frame = g_frameNo.load();
    for (auto it = g_graveyard.begin(); it != g_graveyard.end();) {
        if (frame > it->first + 16) {
            Release(it->second);
            it = g_graveyard.erase(it);
        } else {
            ++it;
        }
    }
    if (g_targets && g_targets->w == w && g_targets->h == h &&
        g_targets->motionFormat == (DXGI_FORMAT)g_motionFormat.load())
        return g_targets;
    if (g_targets) g_graveyard.push_back({frame, g_targets});
    g_targets = nullptr;

    auto* t = new Targets;
    t->w = w;
    t->h = h;
    t->colorFormat = (DXGI_FORMAT)g_colorFormat.load();
    t->motionFormat = (DXGI_FORMAT)g_motionFormat.load();
    D3D12_CLEAR_VALUE one{DXGI_FORMAT_R8_UNORM, {{1, 1, 1, 1}}}, zero{DXGI_FORMAT_R8_UNORM, {{0, 0, 0, 0}}};
    t->cov = MakeTexture(w, h, DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                         D3D12_RESOURCE_STATE_RENDER_TARGET, &one);
    t->kept = MakeTexture(w, h, DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                          D3D12_RESOURCE_STATE_RENDER_TARGET, &zero);
    t->tensor = MakeTexture(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
    t->output = MakeTexture(w, h, t->colorFormat, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
    t->depthOut = MakeTexture(w, h, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
    if (t->motionFormat != DXGI_FORMAT_UNKNOWN)
        t->motionOut = MakeTexture(w, h, t->motionFormat, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    g_device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&t->rtvHeap));
    if (!t->cov || !t->kept || !t->tensor || !t->output || !t->depthOut || !t->rtvHeap) {
        Release(t);
        return nullptr;
    }
    UINT inc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    t->rtv[0] = t->rtvHeap->GetCPUDescriptorHandleForHeapStart();
    t->rtv[1] = t->rtv[0];
    t->rtv[1].ptr += inc;
    g_device->CreateRenderTargetView(t->cov, nullptr, t->rtv[0]);
    g_device->CreateRenderTargetView(t->kept, nullptr, t->rtv[1]);
    Log("hair: targets %ux%u (colour format %d)", w, h, t->colorFormat);
    g_targets = t;
    return t;
}

// side PSO for a slot / preset, built on first use (the game's VS etc. come from the desc copy)
ID3D12PipelineState* BuildSide(Slot& slot, int preset) {
    std::lock_guard<std::mutex> lock(g_resMutex);
    if (ID3D12PipelineState* s = slot.side[preset].load()) return s;
    if (!g_device || !slot.desc) return nullptr;
    const CoverageShader& shader = kCoverageShaders[slot.rawIndex + preset];
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d = slot.desc->d;
    MakeSideDesc(d, shader);
    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = g_device->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        Log("hair: side PSO (%s preset %d) failed 0x%lx", shader.variant, preset, hr);
        return nullptr;
    }
    slot.side[preset].store(pso);
    return pso;
}

// preset index (0 raw, 1.. sharpen) for the current settings, relative to the slot's raw entry
int CurrentPreset(const Slot& slot) {
    if (g_config.coverageMode != 1) return 0;
    const CoverageShader* c = FindCoverageShader(kCoverageShaders[slot.rawIndex].gameHash, true,
                                                 g_config.sharpenThreshold, g_config.sharpenWidth);
    int i = c ? (int)(c - &kCoverageShaders[slot.rawIndex]) : 0;
    return i > 0 && i < kPresets ? i : 0;
}

ID3D12PipelineState* SideFor(void* game) {
    if (!game) return nullptr;
    int n = g_numPairs.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        Slot& slot = g_slots[i];
        if (slot.game.load(std::memory_order_relaxed) != game) continue;
        int preset = CurrentPreset(slot);
        ID3D12PipelineState* s = slot.side[preset].load();
        if (!s) s = BuildSide(slot, preset);   // first use of this preset: one-off compile
        return s ? s : slot.side[0].load();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------------------------
// resolve compute

constexpr UINT kRing = 8;          // descriptor sets in flight
constexpr UINT kSetSize = 10;      // t0..t5, u0..u3
ID3D12RootSignature* g_csRootSig = nullptr;
ID3D12PipelineState* g_tensorPso = nullptr;
ID3D12PipelineState* g_resolvePso = nullptr;
ID3D12PipelineState* g_areaRowPso = nullptr;    // [Hair] Area: row distance to strand edges
ID3D12PipelineState* g_areaPso = nullptr;       // ... distance, and the game's own image outside the area
ID3D12DescriptorHeap* g_csHeap = nullptr;
UINT g_csInc = 0;
UINT g_ring = 0;
bool g_csFailed = false;

bool InitCompute() {
    if (g_resolvePso) return true;
    if (g_csFailed || !g_device) return false;
    HRESULT hr = g_device->CreateRootSignature(0, kResolveCso, sizeof(kResolveCso), IID_PPV_ARGS(&g_csRootSig));
    D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
    if (SUCCEEDED(hr)) {
        d.pRootSignature = g_csRootSig;
        d.CS = {kResolveCso, sizeof(kResolveCso)};
        hr = g_device->CreateComputePipelineState(&d, IID_PPV_ARGS(&g_resolvePso));
    }
    if (SUCCEEDED(hr)) {
        d.CS = {kResolve_tensorCso, sizeof(kResolve_tensorCso)};
        hr = g_device->CreateComputePipelineState(&d, IID_PPV_ARGS(&g_tensorPso));
    }
    if (SUCCEEDED(hr)) {
        d.CS = {kArea_hCso, sizeof(kArea_hCso)};
        hr = g_device->CreateComputePipelineState(&d, IID_PPV_ARGS(&g_areaRowPso));
    }
    if (SUCCEEDED(hr)) {
        d.CS = {kAreaCso, sizeof(kAreaCso)};
        hr = g_device->CreateComputePipelineState(&d, IID_PPV_ARGS(&g_areaPso));
    }
    if (SUCCEEDED(hr)) {
        D3D12_DESCRIPTOR_HEAP_DESC h{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRing * kSetSize,
                                     D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        hr = g_device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&g_csHeap));
    }
    if (FAILED(hr)) {
        Log("hair: compute init failed 0x%lx", hr);
        g_csFailed = true;
        return false;
    }
    g_csInc = g_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    Log("hair: compute initialised");
    return true;
}

DXGI_FORMAT TypedFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
        case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        default: return f;
    }
}

void Barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to,
             std::vector<D3D12_RESOURCE_BARRIER>& out) {
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    out.push_back(b);
}

void Flush(ID3D12GraphicsCommandList* cl, std::vector<D3D12_RESOURCE_BARRIER>& b) {
    if (!b.empty()) cl->ResourceBarrier((UINT)b.size(), b.data());
    b.clear();
}

// SRV format that reads the depth plane of a depth(-stencil) resource as float
DXGI_FORMAT DepthSrvFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R32G8X24_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_D32_FLOAT:
        case DXGI_FORMAT_R32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R16_TYPELESS:
        case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
        case DXGI_FORMAT_R24G8_TYPELESS:
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

thread_local Targets* t_pending = nullptr;   // resolved this dispatch, targets still readable

void ClearTargets(ID3D12GraphicsCommandList* cl, Targets* t) {
    const float one[4] = {1, 1, 1, 1}, zero[4] = {0, 0, 0, 0};
    cl->ClearRenderTargetView(t->rtv[0], one, 0, nullptr);
    cl->ClearRenderTargetView(t->rtv[1], zero, 0, nullptr);
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

}  // namespace

// ---------------------------------------------------------------------------------------------

void RegisterAlphaAccumPso(void* gamePso, uint64_t psHash, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc,
                           const D3D12_PIPELINE_STATE_STREAM_DESC* stream) {
    const CoverageShader* raw = FindCoverageShader(psHash, false, 0, 0);
    if (!raw) {
        Log("hair: unknown alpha_accum shader %016llx, no side pass for PSO %p", (unsigned long long)psHash, gamePso);
        return;
    }
    D3D12_GRAPHICS_PIPELINE_STATE_DESC base{};
    if (desc) base = *desc;
    else if (!stream || !StreamToGraphicsDesc(stream, base)) {
        Log("hair: cannot convert stream PSO %p", gamePso);
        return;
    }
    ID3D12Device* dev = nullptr;
    static_cast<ID3D12PipelineState*>(gamePso)->GetDevice(IID_PPV_ARGS(&dev));
    if (!dev) return;
    {
        std::lock_guard<std::mutex> lock(g_resMutex);
        if (!g_device) {
            g_device = dev;
            g_device->AddRef();
        }
    }
    dev->Release();

    Slot* slot = nullptr;
    {
        // a freed PSO's address can be reused by a new one: replace an existing entry, else take a free slot
        std::lock_guard<std::mutex> lock(g_resMutex);
        int n = g_numPairs.load(), index = -1;
        for (int i = 0; i < n && index < 0; ++i)
            if (g_slots[i].game.load() == gamePso) index = i;
        for (int i = 0; i < n && index < 0; ++i)
            if (!g_slots[i].game.load()) index = i;
        if (index < 0) {
            if (n >= kMaxPairs) return;
            index = n;
            g_numPairs.store(n + 1, std::memory_order_release);
        }
        slot = &g_slots[index];
        slot->game.store(nullptr);   // not usable while it is being refilled
        // previous side PSOs / desc copy (if any) are leaked on purpose: they may still be in flight
        for (auto& p : slot->side) p.store(nullptr);
        slot->rawIndex = (int)(raw - kCoverageShaders);
        slot->desc = CopyDesc(base);
    }
    // build the raw side PSO and the currently selected preset now; other presets on demand
    ID3D12PipelineState* rawSide = BuildSide(*slot, 0);
    int preset = CurrentPreset(*slot);
    if (preset) BuildSide(*slot, preset);
    if (!rawSide) return;
    slot->game.store(gamePso);
    Log("hair: side PSOs (%s) for game PSO %p (preset %d)", raw->variant, gamePso, preset);
}

bool HasSidePso(void* gamePso) { return SideFor(gamePso) != nullptr; }

void ForgetGamePso(void* gamePso) {
    int n = g_numPairs.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i)
        if (g_slots[i].game.load(std::memory_order_relaxed) == gamePso) g_slots[i].game.store(nullptr);
}

void TrackRenderTargets(void* list, UINT n, const D3D12_CPU_DESCRIPTOR_HANDLE* rts, BOOL single,
                        const D3D12_CPU_DESCRIPTOR_HANDLE* dsv) {
    t_rt.list = list;
    t_rt.n = n > 8 ? 8 : n;
    t_rt.single = single;
    UINT copy = single ? (n ? 1 : 0) : t_rt.n;
    for (UINT i = 0; i < copy && rts; ++i) t_rt.rts[i] = rts[i];
    t_rt.hasDsv = dsv != nullptr;
    if (dsv) t_rt.dsv = *dsv;
}

void TrackRenderPass(void* list, bool inside) {
    t_passList = list;
    t_inPass = inside;
}

void AfterAlphaAccumDraw(void* list, void* gamePso, bool indexed, UINT a, UINT b, UINT c, INT d, UINT e) {
    if (!IsActive()) return;
    ID3D12PipelineState* side = SideFor(gamePso);
    if (!side) return;
    if (t_inPass && t_passList == list) {
        LogOnce("hair: alpha_accum drawn inside a D3D12 render pass - side pass not supported yet");
        return;
    }
    if (t_rt.list != list || !t_rt.hasDsv) {
        LogOnce("hair: render targets of the alpha_accum draw unknown - side pass skipped");
        return;
    }
    Targets* t = CurrentTargets();
    if (!t) return;

    auto* cl = static_cast<ID3D12GraphicsCommandList*>(list);
    if (t->needsInitialClear.exchange(false)) ClearTargets(cl, t);
    RtState saved = t_rt;   // our own OMSetRenderTargets below goes through the tracking hook
    cl->OMSetRenderTargets(2, t->rtv, FALSE, &saved.dsv);
    cl->SetPipelineState(side);
    if (indexed) cl->DrawIndexedInstanced(a, b, c, d, e);
    else cl->DrawInstanced(a, b, c, (UINT)d);
    cl->SetPipelineState(static_cast<ID3D12PipelineState*>(gamePso));
    cl->OMSetRenderTargets(saved.n, saved.n ? saved.rts : nullptr, saved.single, &saved.dsv);
}

bool ResolveBeforeUpscale(void* commandList, const UpscaleInputs& in, UpscaleInputs& out) {
    out = {};
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    auto* color = static_cast<ID3D12Resource*>(in.color);
    auto* depth = static_cast<ID3D12Resource*>(in.depth);
    auto* motion = static_cast<ID3D12Resource*>(in.motion);
    if (!cl || !color) return false;
    D3D12_RESOURCE_DESC cd = color->GetDesc();
    g_colorW = (UINT)cd.Width;
    g_colorH = cd.Height;
    g_colorFormat = (int)TypedFormat(cd.Format);
    DXGI_FORMAT motionFmt = motion ? TypedFormat(motion->GetDesc().Format) : DXGI_FORMAT_UNKNOWN;
    g_motionFormat = (int)motionFmt;

    if (!IsActive()) return false;
    if (cl->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) {
        LogOnce("hair: upscaler runs on a non-graphics command list - resolve skipped");
        return false;
    }
    Targets* t;
    {
        std::lock_guard<std::mutex> lock(g_resMutex);
        t = g_targets;
    }
    if (!t || t->w != cd.Width || t->h != cd.Height || t->needsInitialClear.load() || !InitCompute()) return false;
    constexpr unsigned kComputeRead = 1 << 2;   // FFX_API_RESOURCE_STATE_COMPUTE_READ
    if (!(in.colorState & kComputeRead)) {
        LogOnce("hair: colour input not in a compute-read state - resolve skipped");
        return false;
    }
    DXGI_FORMAT depthFmt = depth ? DepthSrvFormat(depth->GetDesc().Format) : DXGI_FORMAT_UNKNOWN;
    bool stabilize = g_config.stabilizeDepth && depthFmt != DXGI_FORMAT_UNKNOWN && t->motionOut &&
                     motionFmt == t->motionFormat && (in.depthState & kComputeRead) && (in.motionState & kComputeRead);
    if (g_config.stabilizeDepth && !stabilize)
        LogOnce("hair: depth / motion inputs not usable - depth stabilisation off");

    UINT slot = (g_ring++ % kRing) * kSetSize;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_csHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_csHeap->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T)slot * g_csInc;
    gpu.ptr += (UINT64)slot * g_csInc;
    auto at = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = cpu;
        h.ptr += (SIZE_T)i * g_csInc;
        return h;
    };
    auto srv = [&](ID3D12Resource* r, DXGI_FORMAT f, UINT i) {   // r may be null: null descriptor
        D3D12_SHADER_RESOURCE_VIEW_DESC s{};
        s.Format = f;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Texture2D.MipLevels = 1;
        g_device->CreateShaderResourceView(r, &s, at(i));
    };
    auto uav = [&](ID3D12Resource* r, DXGI_FORMAT f, UINT i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = f;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        g_device->CreateUnorderedAccessView(r, nullptr, &u, at(i));
    };
    srv(color, TypedFormat(cd.Format), 0);
    srv(t->cov, DXGI_FORMAT_R8_UNORM, 1);
    srv(t->kept, DXGI_FORMAT_R8_UNORM, 2);
    srv(t->tensor, DXGI_FORMAT_R16G16B16A16_FLOAT, 3);
    srv(stabilize ? depth : nullptr, stabilize ? depthFmt : DXGI_FORMAT_R32_FLOAT, 4);
    srv(stabilize ? motion : nullptr, stabilize ? motionFmt : DXGI_FORMAT_R16G16_FLOAT, 5);
    uav(t->tensor, DXGI_FORMAT_R16G16B16A16_FLOAT, 6);
    uav(t->output, t->colorFormat, 7);
    uav(t->depthOut, DXGI_FORMAT_R32_FLOAT, 8);
    uav(t->motionOut, t->motionOut ? t->motionFormat : DXGI_FORMAT_R16G16_FLOAT, 9);

    const auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const auto kUav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    std::vector<D3D12_RESOURCE_BARRIER> bars;
    Barrier(cl, t->cov, D3D12_RESOURCE_STATE_RENDER_TARGET, kRead, bars);
    Barrier(cl, t->kept, D3D12_RESOURCE_STATE_RENDER_TARGET, kRead, bars);
    Barrier(cl, t->tensor, t->tensorState, kUav, bars);
    Barrier(cl, t->output, t->outputState, kUav, bars);
    Barrier(cl, t->depthOut, t->auxState, kUav, bars);
    if (t->motionOut) Barrier(cl, t->motionOut, t->auxState, kUav, bars);
    Flush(cl, bars);

    const Config& cfg = g_config;
    uint32_t mode = cfg.resolveMode == 0 ? 0 : 1;   // box | strand (the network runs on the strand resolve)
    float strength = cfg.strength;
    uint32_t view = cfg.debugView <= 3 ? (uint32_t)cfg.debugView : 0;   // 4 is the unsharp-mask view
    uint32_t consts[8] = {t->w, t->h, mode, 0, view, stabilize ? 1u : 0u, 0, 0};
    memcpy(&consts[3], &strength, 4);

    cl->SetDescriptorHeaps(1, &g_csHeap);
    cl->SetComputeRootSignature(g_csRootSig);
    cl->SetComputeRoot32BitConstants(0, 8, consts, 0);
    cl->SetComputeRootDescriptorTable(1, gpu);
    if (mode == 1 || view == 3) {
        cl->SetPipelineState(g_tensorPso);
        cl->Dispatch((t->w + 15) / 16, (t->h + 15) / 16, 1);
        Barrier(cl, t->tensor, kUav, kRead, bars);
        Flush(cl, bars);
        t->tensorState = kRead;
    }
    cl->SetPipelineState(g_resolvePso);
    cl->Dispatch((t->w + 7) / 8, (t->h + 7) / 8, 1);

    Barrier(cl, t->output, kUav, kRead, bars);
    Barrier(cl, t->depthOut, kUav, kRead, bars);
    if (t->motionOut) Barrier(cl, t->motionOut, kUav, kRead, bars);
    Flush(cl, bars);
    t->outputState = kRead;
    t->auxState = kRead;
    t_pending = t;   // FinishUpscale returns the hair targets to render-target state and clears them
    // the resolve network rewrites the resolve from it and the raw frame (debug views show the resolve)
    if (cfg.resolveMode == 2 && view == 0)
        NeuralResolve(cl, t->output, kRead, color, t->cov, t->kept, t->w, t->h);
    // [Hair] Area: the mod works only this far from strand edges. Outside, FSR gets the game's own
    // colour back, and the unsharp mask (usm.cpp) skips it: the map stays in .y of the tensor target
    // (no longer needed by now; PendingArea hands it on)
    t->areaValid = false;
    if ((cfg.area < kAreaAll || cfg.debugView == 7) && view == 0) {
        // width < 0: all hair (only the debug view gets here then)
        float width = cfg.area < kAreaAll ? std::max(cfg.area, 0.0f) : -1.0f;
        float feather = std::min(std::max(cfg.areaFeather, 0.5f), 32.0f);
        uint32_t ac[8] = {t->w, t->h, 0, 0, (uint32_t)cfg.debugView, stabilize ? 1u : 0u, 0, 0};
        memcpy(&ac[2], &width, 4);
        memcpy(&ac[3], &feather, 4);
        Barrier(cl, t->tensor, t->tensorState, kUav, bars);
        Barrier(cl, t->output, kRead, kUav, bars);
        Barrier(cl, t->depthOut, kRead, kUav, bars);
        if (t->motionOut) Barrier(cl, t->motionOut, kRead, kUav, bars);
        Flush(cl, bars);
        cl->SetDescriptorHeaps(1, &g_csHeap);   // the network uses its own
        cl->SetComputeRootSignature(g_csRootSig);
        cl->SetComputeRoot32BitConstants(0, 8, ac, 0);
        cl->SetComputeRootDescriptorTable(1, gpu);
        cl->SetPipelineState(g_areaRowPso);
        cl->Dispatch((t->w + 7) / 8, (t->h + 7) / 8, 1);
        D3D12_RESOURCE_BARRIER u{};
        u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        u.UAV.pResource = t->tensor;
        cl->ResourceBarrier(1, &u);
        cl->SetPipelineState(g_areaPso);
        cl->Dispatch((t->w + 7) / 8, (t->h + 7) / 8, 1);
        Barrier(cl, t->tensor, kUav, kRead, bars);
        Barrier(cl, t->output, kUav, kRead, bars);
        Barrier(cl, t->depthOut, kUav, kRead, bars);
        if (t->motionOut) Barrier(cl, t->motionOut, kUav, kRead, bars);
        Flush(cl, bars);
        t->tensorState = kRead;
        t->areaValid = true;
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        Log("hair: first resolve %ux%u mode=%s strength=%.2f debug=%d stabilise=%d (depth fmt %d, motion fmt %d)",
            t->w, t->h, mode ? "strand" : "box", strength, cfg.debugView, stabilize, depthFmt, motionFmt);
    }
    out.color = t->output;
    if (stabilize) {
        out.depth = t->depthOut;
        out.motion = t->motionOut;
    }
    return true;
}

bool PendingArea(void** area) {
    if (!t_pending || !t_pending->areaValid) return false;
    *area = t_pending->tensor;
    return true;
}

bool PendingCoverage(void** coverage, unsigned* width, unsigned* height) {
    if (!t_pending) return false;
    *coverage = t_pending->cov;
    *width = t_pending->w;
    *height = t_pending->h;
    return true;
}

bool PendingTargets(void** coverage, void** kept, void** resolved) {
    if (!t_pending) return false;
    *coverage = t_pending->cov;
    *kept = t_pending->kept;
    *resolved = t_pending->output;
    return true;
}

void FinishUpscale(void* commandList) {
    Targets* t = t_pending;
    t_pending = nullptr;
    if (!t) return;
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    std::vector<D3D12_RESOURCE_BARRIER> bars;
    Barrier(cl, t->cov, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET, bars);
    Barrier(cl, t->kept, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET, bars);
    Flush(cl, bars);
    ClearTargets(cl, t);   // ready for the next frame's side pass (GPU order)
}

void OnHairFrameBoundary() { g_frameNo.fetch_add(1); }

}  // namespace hf
