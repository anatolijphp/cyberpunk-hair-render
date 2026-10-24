// Hair fix: side coverage pass + guided resolve (phase 2).
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace hf {

struct ViewSettings;   // common.h

// The coverage target: hair layers multiply into it one by one, and an 8-bit target would round
// each layer's (1 - a) back to 1 when a < 1/510, so thin hair would vanish from the coverage.
constexpr DXGI_FORMAT kCoverageFormat = DXGI_FORMAT_R16_UNORM;

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
uint64_t HairFrame();   // frames counted by OnHairFrameBoundary
void MirrorStats(char* buf, size_t size);   // counters since the last call, for the perf log

// mirrors (a second view without upscaler; [Hair] Mirror): depth views tell a hair draw's view
void HairOnDepthView(void* resource, D3D12_CPU_DESCRIPTOR_HANDLE dst);
// width << 16 | height of the view other than the main one whose hair was drawn last (this or the
// last frame: a mirror's), else 0
uint32_t MirrorSize();
// Right after the mirror's final pass on `commandList`: the hair resolve of its output image, the
// descriptor `offset` of the shader-visible table `table` (a UAV). That descriptor may still change
// until the list is submitted, so the image is reached through descriptors of the mod's own, written
// at MirrorBeforeExecute; the recorded work touches the game's image only through them.
void MirrorAfterPass(void* commandList, uint64_t table, unsigned offset);
// command lists about to be submitted: their mirror descriptors, from the game's final ones
void MirrorBeforeExecute(unsigned n, void* const* lists);
void MirrorListReset(void* list);   // a list is reset: its pending mirror descriptors are dropped

// unsharp mask on the upscaler output (usm.cpp)
void ApplyUnsharpMask(void* commandList, void* output, unsigned ffxOutputState, unsigned renderW, unsigned renderH);
// ... and on a mirror's resolve: the top-left width x height of `image` (NON_PIXEL_SHADER_RESOURCE,
// UAV-capable), its hair coverage target and area map (null: all hair)
void UnsharpMaskMirror(void* commandList, void* image, unsigned width, unsigned height, void* coverage, void* area,
                       const ViewSettings& settings);

// helpers (overlay.cpp)
DXGI_FORMAT TypedFormatForView(DXGI_FORMAT f);      // typeless -> typed; sRGB -> UNKNOWN (no UAV)
unsigned D3D12StateFromFfx(unsigned ffxState);      // FFX_API_RESOURCE_STATE_* -> D3D12_RESOURCE_STATES

}  // namespace hf
