// FSR 3 / 4: the game calls the upscaler through the FFX API (amd_fidelityfx_loader_dx12.dll, see
// proxy.cpp), so the inputs are handed over directly: the resolved colour (and optionally stabilised
// depth / motion) replace the game's in the dispatch description, the unsharp mask runs on the output.
// The built-in FSR 2.1 has no such call; see fsr2.h.
#pragma once

#include "../third_party/FidelityFX/api/include/ffx_api.h"
#include "../third_party/FidelityFX/upscalers/include/ffx_upscale.h"

namespace hf {

// Runs an upscale dispatch through `dispatch` (the original loader) with the hair fix around it.
ffxReturnCode_t Fsr3Upscale(PfnFfxDispatch dispatch, ffxContext* context, const ffxDispatchDescUpscale* up);

}  // namespace hf
