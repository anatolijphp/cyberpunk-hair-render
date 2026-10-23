// CBV/SRV/UAV descriptor tracking: which resource every descriptor in every heap points at, so a
// bound descriptor table can be mapped back to resources (the built-in FSR 2.1 has no API to hook).
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace hf {

struct DescInfo {
    void* res = nullptr;   // ID3D12Resource*, valid while the descriptor is (the game keeps it alive)
    uint32_t w = 0, h = 0;
    uint16_t fmt = 0;      // view format (resource format if the view has none)
    uint8_t uav = 0;
    uint8_t dim = 0;       // D3D12_RESOURCE_DIMENSION, 0 = empty / null descriptor
};

void DescOnHeapCreated(void* heap);   // ID3D12DescriptorHeap*
void DescOnView(void* resource, D3D12_CPU_DESCRIPTOR_HANDLE dst, bool uav, DXGI_FORMAT viewFormat);
void DescOnCopySimple(UINT n, D3D12_CPU_DESCRIPTOR_HANDLE dst, D3D12_CPU_DESCRIPTOR_HANDLE src);
void DescOnCopy(UINT numDst, const D3D12_CPU_DESCRIPTOR_HANDLE* dstStarts, const UINT* dstSizes, UINT numSrc,
                const D3D12_CPU_DESCRIPTOR_HANDLE* srcStarts, const UINT* srcSizes);

// descriptor `offset` of the table at GPU handle `table` (shader-visible heaps only)
bool DescAt(uint64_t table, UINT offset, DescInfo* out);

}  // namespace hf
