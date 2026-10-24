// Writes the network's correction into the FSR output inside one chunk:
//   tone-mapped base image + hair mask * correction, back to linear.
// debugView 5: the hair mask; 6: the correction (grey = none).
#include "nn_common.hlsli"

// scalars only: root constants are packed tightly, as the C++ structs in neural.cpp
cbuffer Params : register(b0) {
    int originX, originY;       // display pixel of buffer pixel (0, 0)
    uint size;
    uint displayW, displayH;
    float uvScaleX, uvScaleY;
    float exposure;
    int chunkX0, chunkY0, chunkX1, chunkY1;   // display pixels this chunk writes
    uint debugView;
    uint widen;                 // hair mask widening in pixels (as in training)
    uint pad0, pad1;
};
static const int2 origin = int2(originX, originY);
static const uint2 displaySize = uint2(displayW, displayH);
static const float2 uvScale = float2(uvScaleX, uvScaleY);
static const int4 chunk = int4(chunkX0, chunkY0, chunkX1, chunkY1);

Texture2D<float4> source : register(t0);
Texture2D<float> coverage : register(t1);
Texture2D<float> kept : register(t2);
SamplerState linearClamp : register(s0);
RWTexture2D<float4> output : register(u0);
RWBuffer<float> correction : register(u1);   // 3 planes

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    int2 q = chunk.xy + int2(id);
    if (any(q >= chunk.zw) || any(q >= int2(displaySize)) || !ChunkHasHair(chunk)) return;
    float mask = 0;
    const int wr = int(widen);
    for (int y = -wr; y <= wr; ++y)
        for (int x = -wr; x <= wr; ++x) {
            float2 uv = (float2(clamp(q + int2(x, y), int2(0, 0), int2(displaySize) - 1)) + 0.5) * uvScale;
            mask = max(mask, RawHair(coverage.SampleLevel(linearClamp, uv, 0), kept.SampleLevel(linearClamp, uv, 0)));
        }
    float4 c = source[q];
    if (debugView == 5) {
        output[q] = float4(mask.xxx, c.a);
        return;
    }
    if (mask <= 0) return;   // output already holds the unmodified pixel
    uint2 p = uint2(q - origin);
    uint plane = size * size, i = p.y * size + p.x;
    float3 r = float3(correction[i], correction[plane + i], correction[2 * plane + i]);
    if (debugView == 6) {
        output[q] = float4(0.5 + r * mask * 4, c.a);
        return;
    }
    float3 t = ToneMap(c.rgb, exposure);
    output[q] = float4(InverseToneMap(t + r * mask, exposure), c.a);
}
