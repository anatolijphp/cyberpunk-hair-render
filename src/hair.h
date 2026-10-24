// Hair fix: side coverage pass + guided resolve (phase 2).
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace hf {

// A game alpha_accum PSO was created; builds the matching side PSO (desc or stream, one non-null).
void RegisterAlphaAccumPso(void* gamePso, uint64_t psHash, const D3D12_GRAPHICS_PIPELINE_STATE_DESC* desc,
                           const D3D12_PIPELINE_STATE_STREAM_DESC* stream);
bool HasSidePso(void* gamePso);
void ForgetGamePso(void* gamePso);   // a non-hair PSO now lives at this address

// command-list tracking (called from the vtable hooks, on the recording thread)
void TrackRenderTargets(void* list, UINT n, const D3D12_CPU_DESCRIPTOR_HANDLE* rts, BOOL single,
                        const D3D12_CPU_DESCRIPTOR_HANDLE* dsv);
void TrackRenderPass(void* list, bool inside);
void AfterAlphaAccumDraw(void* list, void* gamePso, bool indexed, UINT a, UINT b, UINT c, INT d, UINT e);

// Upscaler inputs; on return the non-null fields of `out` replace them (all in
// NON_PIXEL_SHADER_RESOURCE state). Returns false when nothing was replaced.
struct UpscaleInputs {
    void* color = nullptr;
    unsigned colorState = 0;
    void* depth = nullptr;
    unsigned depthState = 0;
    void* motion = nullptr;
    unsigned motionState = 0;
};
bool ResolveBeforeUpscale(void* commandList, const UpscaleInputs& in, UpscaleInputs& out);
// After the upscaler: the coverage target (render resolution, NON_PIXEL_SHADER_RESOURCE) of the
// resolve that ran in this dispatch, for the unsharp mask; then FinishUpscale clears the targets.
bool PendingCoverage(void** coverage, unsigned* width, unsigned* height);
// The Area map of the same frame (RGBA16F, .y = 1 inside the area; same size as the coverage,
// readable by compute), or false when the unsharp mask covers all hair.
bool PendingArea(void** area);
// All targets of that resolve (NON_PIXEL_SHADER_RESOURCE), for the training capture.
bool PendingTargets(void** coverage, void** kept, void** resolved);
void FinishUpscale(void* commandList);
void OnHairFrameBoundary();

// unsharp mask on the upscaler output (usm.cpp)
void ApplyUnsharpMask(void* commandList, void* output, unsigned ffxOutputState, unsigned renderW, unsigned renderH);

// helpers (overlay.cpp)
DXGI_FORMAT TypedFormatForView(DXGI_FORMAT f);      // typeless -> typed; sRGB -> UNKNOWN (no UAV)
unsigned D3D12StateFromFfx(unsigned ffxState);      // FFX_API_RESOURCE_STATE_* -> D3D12_RESOURCE_STATES

}  // namespace hf
