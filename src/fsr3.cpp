// FSR 3 / 4 through the FFX API: see fsr3.h.
#include "fsr3.h"

#include "common.h"
#include "hair.h"
#include "probe.h"

namespace hf {

ffxReturnCode_t Fsr3Upscale(PfnFfxDispatch dispatch, ffxContext* context, const ffxDispatchDescUpscale* up) {
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
    if (ResolveBeforeUpscale(up->commandList, in, out)) {
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
        ApplyUnsharpMask(up->commandList, up->output.resource, up->output.state, up->renderSize.width,
                         up->renderSize.height);
        AfterUpscale(up->commandList, up->output.resource, up->output.state);
    }
    FinishUpscale(up->commandList);
    OnFrameBoundary();
    return r;
}

}  // namespace hf
