// Descriptor tracking: see descriptors.h.
//
// Every CBV/SRV/UAV heap gets a shadow array of DescInfo, filled by CreateShaderResourceView /
// CreateUnorderedAccessView and copied along by CopyDescriptors(Simple). Heaps are looked up by
// CPU address under a shared lock; slots are written without one (racing writes to one slot are
// the game's own race).
#include "descriptors.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "common.h"

namespace hf {
namespace {

struct Heap {
    uintptr_t cpu = 0;
    uint64_t gpu = 0;   // 0: not shader visible
    uint32_t num = 0, inc = 0;
    std::vector<DescInfo> slots;
};

std::shared_mutex g_m;
std::map<uintptr_t, std::unique_ptr<Heap>> g_heaps;   // by CPU start
std::vector<Heap*> g_visible;                         // shader-visible heaps
size_t g_created = 0, g_replaced = 0;                 // CBV/SRV/UAV heaps seen / replaced at a reused address

Heap* HeapAt(uintptr_t cpu) {
    auto it = g_heaps.upper_bound(cpu);
    if (it == g_heaps.begin()) return nullptr;
    Heap* h = std::prev(it)->second.get();
    return cpu < h->cpu + (uintptr_t)h->num * h->inc ? h : nullptr;
}

DescInfo* SlotAt(uintptr_t cpu) {
    Heap* h = HeapAt(cpu);
    return h ? &h->slots[(cpu - h->cpu) / h->inc] : nullptr;
}

}  // namespace

void DescOnHeapCreated(void* heap) {
    auto* dh = static_cast<ID3D12DescriptorHeap*>(heap);
    D3D12_DESCRIPTOR_HEAP_DESC d = dh->GetDesc();
    if (d.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV || !d.NumDescriptors) return;
    ID3D12Device* dev = nullptr;
    if (FAILED(dh->GetDevice(IID_PPV_ARGS(&dev))) || !dev) return;
    auto h = std::make_unique<Heap>();
    h->cpu = dh->GetCPUDescriptorHandleForHeapStart().ptr;
    if (d.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) h->gpu = dh->GetGPUDescriptorHandleForHeapStart().ptr;
    h->num = d.NumDescriptors;
    h->inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    dev->Release();
    h->slots.resize(h->num);
    std::unique_lock<std::shared_mutex> lock(g_m);
    ++g_created;
    auto& entry = g_heaps[h->cpu];
    if (entry) {   // a freed heap's address reused: its shadow goes (nobody holds slots across the lock)
        g_visible.erase(std::remove(g_visible.begin(), g_visible.end(), entry.get()), g_visible.end());
        ++g_replaced;
    }
    if (h->gpu) g_visible.push_back(h.get());
    entry = std::move(h);
}

void DescOnView(void* resource, D3D12_CPU_DESCRIPTOR_HANDLE dst, bool uav, DXGI_FORMAT viewFormat) {
    DescInfo s;
    if (resource) {
        D3D12_RESOURCE_DESC d = static_cast<ID3D12Resource*>(resource)->GetDesc();
        s.res = resource;
        s.dim = (uint8_t)d.Dimension;
        s.w = (uint32_t)std::min<UINT64>(d.Width, 0xffffffffu);
        s.h = d.Height;
        s.fmt = (uint16_t)(viewFormat ? viewFormat : d.Format);
        s.uav = uav;
    }
    std::shared_lock<std::shared_mutex> lock(g_m);
    if (DescInfo* d = SlotAt(dst.ptr)) *d = s;
}

void DescOnCopySimple(UINT n, D3D12_CPU_DESCRIPTOR_HANDLE dst, D3D12_CPU_DESCRIPTOR_HANDLE src) {
    std::shared_lock<std::shared_mutex> lock(g_m);
    Heap* dh = HeapAt(dst.ptr);
    if (!dh) return;
    for (UINT i = 0; i < n; ++i) {
        DescInfo* d = SlotAt(dst.ptr + (uintptr_t)i * dh->inc);
        if (!d) break;
        DescInfo* s = SlotAt(src.ptr + (uintptr_t)i * dh->inc);
        *d = s ? *s : DescInfo{};
    }
}

void DescOnCopy(UINT numDst, const D3D12_CPU_DESCRIPTOR_HANDLE* dstStarts, const UINT* dstSizes, UINT numSrc,
                const D3D12_CPU_DESCRIPTOR_HANDLE* srcStarts, const UINT* srcSizes) {
    if (!numDst || !numSrc || !dstStarts || !srcStarts) return;
    std::shared_lock<std::shared_mutex> lock(g_m);
    Heap* first = HeapAt(dstStarts[0].ptr);
    if (!first) return;
    uint32_t inc = first->inc;
    UINT di = 0, dOff = 0, si = 0, sOff = 0;
    while (di < numDst && si < numSrc) {
        UINT dSize = dstSizes ? dstSizes[di] : 1, sSize = srcSizes ? srcSizes[si] : 1;
        if (dOff >= dSize) { ++di; dOff = 0; continue; }
        if (sOff >= sSize) { ++si; sOff = 0; continue; }
        DescInfo* d = SlotAt(dstStarts[di].ptr + (uintptr_t)dOff * inc);
        DescInfo* s = SlotAt(srcStarts[si].ptr + (uintptr_t)sOff * inc);
        if (d) *d = s ? *s : DescInfo{};
        ++dOff;
        ++sOff;
    }
}

void DescStats(size_t* heaps, size_t* visible, size_t* retired, size_t* bytes, size_t* created) {
    std::shared_lock<std::shared_mutex> lock(g_m);
    *heaps = g_heaps.size();
    *visible = g_visible.size();
    *retired = g_replaced;
    *created = g_created;
    size_t b = 0;
    for (auto& h : g_heaps) b += h.second->slots.size() * sizeof(DescInfo);
    *bytes = b;
}

bool DescAt(uint64_t table, UINT offset, DescInfo* out) {
    std::shared_lock<std::shared_mutex> lock(g_m);
    for (Heap* h : g_visible) {
        if (table < h->gpu || table >= h->gpu + (uint64_t)h->num * h->inc) continue;
        size_t i = (table - h->gpu) / h->inc + offset;
        if (i >= h->num) return false;
        *out = h->slots[i];
        return true;
    }
    return false;
}

}  // namespace hf
