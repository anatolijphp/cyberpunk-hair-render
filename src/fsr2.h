// Built-in FSR 2.1. The game runs it as its own compute passes, so there is no FFX call to hook:
// its passes are recognised by shader hash when their PSOs are created, and the mod's work is
// recorded into the game's command list around them:
// - before the first pass (luminance pyramid): resolve the hair, copy the result over the input colour
// - after the last pass (RCAS, or accumulate when FSR sharpening is off): unsharp mask, banner
// The game's compute bindings are tracked so they can be restored after each insertion.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace hf {

void Fsr2PsoCreated(void* pso, uint64_t csHash);   // every PSO (0 = graphics): classifies / forgets

// compute binding tracking (vtable hooks, recording thread)
void TrackDescriptorHeaps(void* list, UINT n, ID3D12DescriptorHeap* const* heaps);
void TrackComputeRootSignature(void* list, ID3D12RootSignature* rootSig);
void TrackPipelineState(void* list, ID3D12PipelineState* pso);
void TrackComputeTable(void* list, UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table);
void TrackComputeConstants(void* list, UINT index, UINT n, const void* data, UINT offset);
void TrackComputeRootView(void* list, UINT index, int kind, D3D12_GPU_VIRTUAL_ADDRESS va);   // 3 CBV, 4 SRV, 5 UAV

// injection points
void Fsr2BeforeDispatch(void* list);
void Fsr2AfterDispatch(void* list);
void Fsr2OnBarriers(void* list, UINT n, const D3D12_RESOURCE_BARRIER* barriers);
void Fsr2BeforeClose(void* list);

}  // namespace hf
