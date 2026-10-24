// Unsharp mask on hair, applied to the upscaler output after the upscale.
#include <d3d12.h>

#include <cstring>
#include <mutex>
#include <vector>

#include "common.h"
#include "hair.h"
#include "../build/gen/usm_cso.h"

namespace hf {
namespace {

constexpr UINT kRing = 8;      // descriptor sets in flight
constexpr UINT kSetSize = 4;   // t0 source copy, t1 coverage, t2 area map, u0 output

ID3D12Device* g_device = nullptr;
ID3D12RootSignature* g_rootSig = nullptr;
ID3D12PipelineState* g_pso = nullptr;
ID3D12DescriptorHeap* g_heap = nullptr;
UINT g_inc = 0;
UINT g_ring = 0;
bool g_failed = false;

ID3D12Resource* g_copy = nullptr;   // copy of the upscaler output (blur source)
D3D12_RESOURCE_STATES g_copyState = D3D12_RESOURCE_STATE_COPY_DEST;
std::mutex g_mutex;

bool Init(ID3D12Device* dev) {
    if (g_pso) return true;
    if (g_failed || !dev) return false;
    HRESULT hr = dev->CreateRootSignature(0, kUsmCso, sizeof(kUsmCso), IID_PPV_ARGS(&g_rootSig));
    if (SUCCEEDED(hr)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
        d.pRootSignature = g_rootSig;
        d.CS = {kUsmCso, sizeof(kUsmCso)};
        hr = dev->CreateComputePipelineState(&d, IID_PPV_ARGS(&g_pso));
    }
    if (SUCCEEDED(hr)) {
        D3D12_DESCRIPTOR_HEAP_DESC h{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kRing * kSetSize,
                                     D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        hr = dev->CreateDescriptorHeap(&h, IID_PPV_ARGS(&g_heap));
    }
    if (FAILED(hr)) {
        Log("usm: init failed 0x%lx", hr);
        g_failed = true;
        return false;
    }
    g_device = dev;
    g_inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    Log("usm: initialised");
    return true;
}

// copy texture matching the output (recreated on size / format change; the old one is leaked,
// it may still be in flight)
ID3D12Resource* CopyTarget(const D3D12_RESOURCE_DESC& out) {
    if (g_copy) {
        D3D12_RESOURCE_DESC d = g_copy->GetDesc();
        if (d.Width == out.Width && d.Height == out.Height && d.Format == out.Format) return g_copy;
    }
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = out.Width;
    d.Height = out.Height;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = out.Format;
    d.SampleDesc.Count = 1;
    ID3D12Resource* r = nullptr;
    if (FAILED(g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
                                                 nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    g_copy = r;
    g_copyState = D3D12_RESOURCE_STATE_COPY_DEST;
    Log("usm: copy target %llux%u fmt=%d", (unsigned long long)out.Width, out.Height, out.Format);
    return g_copy;
}

void Transition(std::vector<D3D12_RESOURCE_BARRIER>& v, ID3D12Resource* r, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    v.push_back(b);
}

void Submit(ID3D12GraphicsCommandList* cl, std::vector<D3D12_RESOURCE_BARRIER>& v) {
    if (!v.empty()) cl->ResourceBarrier((UINT)v.size(), v.data());
    v.clear();
}

}  // namespace

void ApplyUnsharpMask(void* commandList, void* outputRes, unsigned ffxOutputState, unsigned renderW,
                      unsigned renderH) {
    const Config& cfg = g_config;
    if (!cfg.usmEnabled && cfg.debugView != 4 && cfg.debugView != 7) return;
    void* covRes = nullptr;
    unsigned covW = 0, covH = 0;
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    auto* out = static_cast<ID3D12Resource*>(outputRes);
    if (!PendingCoverage(&covRes, &covW, &covH)) {
        static bool warned = false;
        if (!warned && out) Log("usm: no hair coverage for this upscale (hair resolve did not run) - skipped");
        warned = true;
        return;
    }
    auto* cov = static_cast<ID3D12Resource*>(covRes);
    if (!cl || !out || !cov) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    ID3D12Device* dev = nullptr;
    out->GetDevice(IID_PPV_ARGS(&dev));
    bool ok = Init(dev);
    if (dev) dev->Release();
    if (!ok) return;

    D3D12_RESOURCE_DESC od = out->GetDesc();
    DXGI_FORMAT fmt = TypedFormatForView(od.Format);
    if (!(od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || fmt == DXGI_FORMAT_UNKNOWN) {
        static bool warned = false;
        if (!warned) Log("usm: output format %d flags 0x%x not writable", od.Format, od.Flags);
        warned = true;
        return;
    }
    ID3D12Resource* copy = CopyTarget(od);
    if (!copy) return;

    UINT slot = (g_ring++ % kRing) * kSetSize;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_heap->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T)slot * g_inc;
    gpu.ptr += (UINT64)slot * g_inc;
    D3D12_SHADER_RESOURCE_VIEW_DESC s{};
    s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    s.Texture2D.MipLevels = 1;
    s.Format = fmt;
    g_device->CreateShaderResourceView(copy, &s, cpu);
    cpu.ptr += g_inc;
    s.Format = DXGI_FORMAT_R8_UNORM;
    g_device->CreateShaderResourceView(cov, &s, cpu);
    cpu.ptr += g_inc;
    void* areaRes = nullptr;
    bool useArea = PendingArea(&areaRes);
    s.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    g_device->CreateShaderResourceView(useArea ? static_cast<ID3D12Resource*>(areaRes) : nullptr, &s, cpu);
    cpu.ptr += g_inc;
    D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
    u.Format = fmt;
    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device->CreateUnorderedAccessView(out, nullptr, &u, cpu);

    auto outState = (D3D12_RESOURCE_STATES)D3D12StateFromFfx(ffxOutputState);
    std::vector<D3D12_RESOURCE_BARRIER> bars;
    Transition(bars, out, outState, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(bars, copy, g_copyState, D3D12_RESOURCE_STATE_COPY_DEST);
    Submit(cl, bars);
    cl->CopyResource(copy, out);
    Transition(bars, out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Transition(bars, copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Submit(cl, bars);
    g_copyState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    struct {
        uint32_t displayW, displayH;
        float renderW, renderH, covW, covH;
        float amount, radius, threshold;
        uint32_t debugView, useArea, pad;
    } c = {(uint32_t)od.Width, od.Height, (float)renderW, (float)renderH, (float)covW, (float)covH,
           cfg.usmAmount, cfg.usmRadius > 0.3f ? cfg.usmRadius : 0.3f, cfg.usmThreshold,
           (uint32_t)(cfg.debugView == 4 || cfg.debugView == 7 ? cfg.debugView : 0), useArea ? 1u : 0u, 0};
    static_assert(sizeof(c) == 12 * 4, "root constant layout");
    cl->SetDescriptorHeaps(1, &g_heap);
    cl->SetComputeRootSignature(g_rootSig);
    cl->SetComputeRoot32BitConstants(0, 12, &c, 0);
    cl->SetComputeRootDescriptorTable(1, gpu);
    cl->SetPipelineState(g_pso);
    cl->Dispatch(((UINT)od.Width + 7) / 8, (od.Height + 7) / 8, 1);

    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = out;
    bars.push_back(uavBarrier);
    Transition(bars, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outState);
    Submit(cl, bars);

    static ID3D12Resource* lastOut = nullptr;
    static int changes = 0;
    if (lastOut != out && changes < 8) {   // also when the upscaler (and so its output) changes
        lastOut = out;
        ++changes;
        Log("usm: pass on %p %llux%u (render %ux%u) amount=%.2f radius=%.2f threshold=%.3f",
            (void*)out, (unsigned long long)od.Width, od.Height, renderW, renderH, cfg.usmAmount, cfg.usmRadius,
            cfg.usmThreshold);
    }
}

}  // namespace hf
