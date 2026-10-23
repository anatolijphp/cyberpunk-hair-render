// Upscaler probe ([Debug] ProbeUpscaler): logs the compute work of a few frames, to find the
// game's built-in FSR 2.1 passes and their inputs. Diagnostics only, changes nothing.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace hf {

bool ProbeEnabled();   // [Debug] ProbeUpscaler (fixed at startup)

void ProbeComputePso(void* pso, uint64_t csHash);
void ProbeListCreated(void* list, D3D12_COMMAND_LIST_TYPE type);

// command recording (logged only during a capture)
bool ProbeCapturing();
void ProbeSetPso(void* list, void* pso);
void ProbeSetComputeTable(void* list, UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table);
void ProbeDispatch(void* list, UINT x, UINT y, UINT z);
void ProbeBarriers(void* list, UINT n, const D3D12_RESOURCE_BARRIER* barriers);
void ProbeExecute(void* queue, UINT n, void* const* lists);
void ProbeFfxUpscale(void* list, void* color, void* depth, void* motion, void* output);

void ProbePresent();   // frame boundary: starts / ends captures

}  // namespace hf
