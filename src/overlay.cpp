// On-screen status banner drawn into the upscaler output with a small compute shader.
#include <d3d12.h>

#include <cstring>
#include <mutex>

#include "common.h"
#include "hair.h"
#include "../build/gen/banner_cso.h"

namespace hf {
namespace {

constexpr UINT kHeapSlots = 16;   // ring of UAV descriptors, one per banner draw

ID3D12Device* g_device = nullptr;
ID3D12RootSignature* g_rootSig = nullptr;
ID3D12PipelineState* g_pso = nullptr;
ID3D12DescriptorHeap* g_heap = nullptr;
UINT g_descSize = 0;
UINT g_nextSlot = 0;
std::mutex g_initMutex;
bool g_initFailed = false;

uint8_t GlyphId(char c) {
    switch (c) {
        case 'H': return 1;
        case 'A': return 2;
        case 'I': return 3;
        case 'R': return 4;
        case 'F': return 5;
        case 'X': return 6;
        case 'O': return 7;
        case 'N': return 8;
        default: return 0;
    }
}

bool Init(ID3D12Device* dev) {
    std::lock_guard<std::mutex> lock(g_initMutex);
    if (g_pso) return true;
    if (g_initFailed || !dev) return false;
    HRESULT hr = dev->CreateRootSignature(0, kBannerCso, sizeof(kBannerCso), IID_PPV_ARGS(&g_rootSig));
    if (SUCCEEDED(hr)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
        d.pRootSignature = g_rootSig;
        d.CS = {kBannerCso, sizeof(kBannerCso)};
        hr = dev->CreateComputePipelineState(&d, IID_PPV_ARGS(&g_pso));
    }
    if (SUCCEEDED(hr)) {
        D3D12_DESCRIPTOR_HEAP_DESC h{};
        h.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        h.NumDescriptors = kHeapSlots;
        h.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = dev->CreateDescriptorHeap(&h, IID_PPV_ARGS(&g_heap));
    }
    if (FAILED(hr)) {
        Log("banner: init failed 0x%lx", hr);
        g_initFailed = true;
        return false;
    }
    g_device = dev;
    g_descSize = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    Log("banner: initialised");
    return true;
}

void Transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cl->ResourceBarrier(1, &b);
}

}  // namespace

DXGI_FORMAT TypedFormatForView(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_UNKNOWN;   // no sRGB UAVs
        default: return f;
    }
}

unsigned D3D12StateFromFfx(unsigned ffxState) {
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    if (ffxState & (1 << 1)) state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    else if ((ffxState & (1 << 2)) && (ffxState & (1 << 4))) state = D3D12_RESOURCE_STATE_GENERIC_READ;
    else if (ffxState & ((1 << 2) | (1 << 3))) {
        if (ffxState & (1 << 2)) state |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (ffxState & (1 << 3)) state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    } else if (ffxState & (1 << 4)) state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    else if (ffxState & (1 << 5)) state = D3D12_RESOURCE_STATE_COPY_DEST;
    else if (ffxState & (1 << 7)) state = D3D12_RESOURCE_STATE_PRESENT;
    else if (ffxState & (1 << 8)) state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    return (unsigned)state;
}

void DrawBanner(void* device, void* commandList, void* target, unsigned ffxState, const char* text) {
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    auto* res = static_cast<ID3D12Resource*>(target);
    if (!cl || !res || !Init(static_cast<ID3D12Device*>(device))) return;

    D3D12_RESOURCE_DESC rd = res->GetDesc();
    DXGI_FORMAT fmt = TypedFormatForView(rd.Format);
    static bool warned = false;
    if (fmt == DXGI_FORMAT_UNKNOWN || !(rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        if (!warned) Log("banner: output format %d flags 0x%x not drawable", rd.Format, rd.Flags);
        warned = true;
        return;
    }

    auto state = (D3D12_RESOURCE_STATES)D3D12StateFromFfx(ffxState);

    UINT slot = g_nextSlot++ % kHeapSlots;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = g_heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = g_heap->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T)slot * g_descSize;
    gpu.ptr += (UINT64)slot * g_descSize;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = fmt;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    g_device->CreateUnorderedAccessView(res, nullptr, &uav, cpu);

    uint32_t c[8] = {};
    size_t n = strlen(text);
    if (n > 16) n = 16;
    uint32_t scale = (uint32_t)(rd.Height / 360);   // 4 px dots at 1440p
    if (scale < 1) scale = 1;
    c[0] = scale * 8;            // origin x
    c[1] = scale * 8;            // origin y
    c[2] = scale;
    c[3] = (uint32_t)n;
    for (size_t i = 0; i < n; ++i) c[4 + i / 4] |= (uint32_t)GlyphId(text[i]) << ((i % 4) * 8);
    uint32_t w = (uint32_t)(n * 6 - 1 + 4) * scale, h = (7 + 4) * scale;

    Transition(cl, res, state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->SetDescriptorHeaps(1, &g_heap);
    cl->SetComputeRootSignature(g_rootSig);
    cl->SetPipelineState(g_pso);
    cl->SetComputeRoot32BitConstants(0, 8, c, 0);
    cl->SetComputeRootDescriptorTable(1, gpu);
    cl->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = res;
    cl->ResourceBarrier(1, &uavBarrier);
    Transition(cl, res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, state);
}

}  // namespace hf
