// Hair box for the network: the display-pixel bounding box of all render pixels with hair.
// mode 0: reset (one thread), mode 1: accumulate over the render-resolution coverage.
#include "nn_common.hlsli"

// scalars only: root constants are packed tightly, as the C++ structs in neural.cpp
cbuffer Params : register(b0) {
    uint mode;
    uint renderW, renderH;
    float scale;          // display / render
    uint displayW, displayH;
    uint pad0, pad1, pad2, pad3, pad4, pad5, pad6, pad7, pad8, pad9;
};
static const uint2 renderSize = uint2(renderW, renderH);
static const uint2 displaySize = uint2(displayW, displayH);

Texture2D<float> coverage : register(t0);
Texture2D<float> kept : register(t1);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (mode == 0) {
        if (all(id == 0)) {
            hairBox[0] = 0xffffffff;
            hairBox[1] = 0xffffffff;
            hairBox[2] = 0;
            hairBox[3] = 0;
        }
        return;
    }
    if (any(id >= renderSize)) return;
    if (coverage[id] > 0.98 && kept[id] < 0.02) return;
    uint2 lo = uint2(floor(float2(id) * scale));
    uint2 hi = min(uint2(ceil(float2(id + 1) * scale)), displaySize) - 1;
    InterlockedMin(hairBox[0], lo.x);
    InterlockedMin(hairBox[1], lo.y);
    InterlockedMax(hairBox[2], hi.x);
    InterlockedMax(hairBox[3], hi.y);
}
