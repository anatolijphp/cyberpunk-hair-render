// FSR 3 / 4 through the FFX API: see fsr3.h.
#include "fsr3.h"

#include <map>
#include <mutex>

#include "capture.h"
#include "common.h"
#include "hair.h"
#include "probe.h"

namespace hf {

namespace {

// how many calls of each descriptor type were seen; the first few and every 1000th get logged
bool ShouldLog(uint64_t type) {
    static std::mutex m;
    static std::map<uint64_t, uint64_t> counts;
    std::lock_guard<std::mutex> lock(m);
    uint64_t n = ++counts[type];
    return n <= 5 || n % 1000 == 0;
}

}  // namespace

void Fsr3LogCreate(const ffxCreateContextDescHeader* desc) {
    for (auto* h = desc; h; h = h->pNext)
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
            auto* c = reinterpret_cast<const ffxCreateContextDescUpscale*>(h);
            Log("ffx: upscale context flags=0x%x maxRender=%ux%u maxUpscale=%ux%u", c->flags,
                c->maxRenderSize.width, c->maxRenderSize.height, c->maxUpscaleSize.width, c->maxUpscaleSize.height);
        }
}

void Fsr3LogQuery(const ffxQueryDescHeader* desc, ffxReturnCode_t result) {
    for (auto* h = desc; h; h = h->pNext) {
        if (!ShouldLog(h->type)) continue;
        switch (h->type) {
            case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
                auto* q = reinterpret_cast<const ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(h);
                Log("ffx query: render resolution for display %ux%u quality %u -> %ux%u (r=%u)", q->displayWidth,
                    q->displayHeight, q->qualityMode, q->pOutRenderWidth ? *q->pOutRenderWidth : 0,
                    q->pOutRenderHeight ? *q->pOutRenderHeight : 0, result);
                break;
            }
            case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
                auto* q = reinterpret_cast<const ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(h);
                Log("ffx query: upscale ratio for quality %u -> %.3f (r=%u)", q->qualityMode,
                    q->pOutUpscaleRatio ? *q->pOutUpscaleRatio : 0.0f, result);
                break;
            }
            case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
                auto* q = reinterpret_cast<const ffxQueryDescUpscaleGetJitterPhaseCount*>(h);
                Log("ffx query: jitter phase count render %u display %u -> %d (r=%u)", q->renderWidth,
                    q->displayWidth, q->pOutPhaseCount ? *q->pOutPhaseCount : 0, result);
                break;
            }
            default:
                Log("ffx query: type 0x%llx (r=%u)", (unsigned long long)h->type, result);
        }
    }
}

void Fsr3LogConfigure(const ffxConfigureDescHeader* desc) {
    for (auto* h = desc; h; h = h->pNext)
        if (ShouldLog(h->type | (1ull << 63))) Log("ffx configure: type 0x%llx", (unsigned long long)h->type);
}

ffxReturnCode_t Fsr3Upscale(PfnFfxDispatch dispatch, ffxContext* context, const ffxDispatchDescUpscale* up) {
    CaptureNoteRenderSize(up->renderSize.width, up->renderSize.height);
    static uint32_t lastW = 0, lastH = 0;
    if (up->renderSize.width != lastW || up->renderSize.height != lastH) {
        lastW = up->renderSize.width;
        lastH = up->renderSize.height;
        Log("ffx: render size %ux%u -> %ux%u, colour texture %ux%u", lastW, lastH, up->upscaleSize.width,
            up->upscaleSize.height, up->color.description.width, up->color.description.height);
    }
    static int logged = 0;
    if (logged < 1) {
        ++logged;
        Log("upscale: color=%p fmt=%u %ux%u state=%u depth=%p (fmt=%u state=%u) mv=%p (fmt=%u state=%u) out=%p (fmt=%u state=%u) render=%ux%u -> %ux%u",
            up->color.resource, up->color.description.format, up->color.description.width,
            up->color.description.height, up->color.state, up->depth.resource,
            up->depth.description.format, up->depth.state, up->motionVectors.resource,
            up->motionVectors.description.format, up->motionVectors.state, up->output.resource, up->output.description.format, up->output.state,
            up->renderSize.width,
            up->renderSize.height, up->upscaleSize.width, up->upscaleSize.height);
    }
    if (ProbeCapturing())
        ProbeFfxUpscale(up->commandList, up->color.resource, up->depth.resource, up->motionVectors.resource,
                        up->output.resource);
    ffxDispatchDescUpscale patched = *up;
    UpscaleInputs in, out;
    in.color = up->color.resource;
    in.colorState = up->color.state;
    in.depth = up->depth.resource;
    in.depthState = up->depth.state;
    in.motion = up->motionVectors.resource;
    in.motionState = up->motionVectors.state;
    bool resolved = ResolveBeforeUpscale(up->commandList, in, out);
    CaptureFrame(up->commandList, up->color.resource, D3D12StateFromFfx(up->color.state), &up->jitterOffset.x, "fsr3");
    if (resolved) {
        patched.color.resource = out.color;
        patched.color.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
        if (out.depth) {
            patched.depth.resource = out.depth;
            patched.depth.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
            patched.depth.description.format = FFX_API_SURFACE_FORMAT_R32_FLOAT;
        }
        if (out.motion) {
            patched.motionVectors.resource = out.motion;
            patched.motionVectors.state = FFX_API_RESOURCE_STATE_COMPUTE_READ;
        }
    }
    ffxReturnCode_t r = dispatch(context, &patched.header);
    if (r == 0) {
        CaptureOutput(up->commandList, up->output.resource, D3D12StateFromFfx(up->output.state));
        ApplyUnsharpMask(up->commandList, up->output.resource, up->output.state, up->renderSize.width,
                         up->renderSize.height);
        AfterUpscale(up->commandList, up->output.resource, up->output.state);
    }
    FinishUpscale(up->commandList);
    OnFrameBoundary();
    return r;
}

}  // namespace hf
