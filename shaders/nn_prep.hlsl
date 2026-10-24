// Network input for one chunk (as training/train.py builds it): the tone-mapped base image (the
// mod's resolve; 3 planes), hair coverage and hair-surface mask (sampled bilinearly at each pixel),
// the local line direction: the structure tensor of the luminance over 5x5 pixels as
// (cos 2a, sin 2a, coherence), and the tone-mapped raw frame (11 inputs). "Display" below is the
// base image's resolution. Pixels outside the screen repeat the edge.
#include "nn_common.hlsli"

// scalars only: root constants are packed tightly, as the C++ structs in neural.cpp
cbuffer Params : register(b0) {
    int originX, originY;       // display pixel of buffer pixel (0, 0)
    uint size;                  // buffer size (square)
    uint displayW, displayH;
    float uvScaleX, uvScaleY;   // display pixel centre -> coverage uv: (p + 0.5) * uvScale
    float exposure;
    int chunkX0, chunkY0, chunkX1, chunkY1;   // display rect of the chunk, for the hair test
    uint inputs;                // 5: no line features (older networks), 8: with them, 11: + raw frame
    uint pad0, pad1, pad2;
};
static const int2 origin = int2(originX, originY);
static const uint2 displaySize = uint2(displayW, displayH);
static const float2 uvScale = float2(uvScaleX, uvScaleY);
static const int4 chunk = int4(chunkX0, chunkY0, chunkX1, chunkY1);

Texture2D<float4> source : register(t0);   // base image (copy)
Texture2D<float> coverage : register(t1);
Texture2D<float> kept : register(t2);
SamplerState linearClamp : register(s0);
Texture2D<float4> raw : register(t3);     // the game's raw frame (11 inputs)
RWBuffer<float> input : register(u0);

float3 Tone(int2 q) { return ToneMap(source[clamp(q, int2(0, 0), int2(displaySize) - 1)].rgb, exposure); }
float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// luminance of the group's 8x8 tile plus 3 pixels around it, read once per group
groupshared float sLum[14][14];

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID, uint2 group : SV_GroupID, uint2 local : SV_GroupThreadID,
          uint index : SV_GroupIndex) {
    if (!TileNearHair(origin, 0, group, 8)) return;   // whole group
    const int2 base = origin + int2(group * 8) - 3;
    if (inputs >= 8) {
        for (uint i = index; i < 14 * 14; i += 64) sLum[i / 14][i % 14] = Luma(Tone(base + int2(i % 14, i / 14)));
        GroupMemoryBarrierWithGroupSync();
    }
    if (any(id >= size)) return;
    int2 q = clamp(origin + int2(id), int2(0, 0), int2(displaySize) - 1);
    float3 t = Tone(q);
    float2 uv = (float2(q) + 0.5) * uvScale;
    uint plane = size * size, i = id.y * size + id.x;
    input[i] = t.r;
    input[plane + i] = t.g;
    input[2 * plane + i] = t.b;
    input[3 * plane + i] = coverage.SampleLevel(linearClamp, uv, 0);
    input[4 * plane + i] = kept.SampleLevel(linearClamp, uv, 0);
    if (inputs < 8) return;

    // structure tensor over the 5x5 pixels around (local + 3) in the tile
    float jxx = 0, jyy = 0, jxy = 0;
    [unroll] for (int y = 1; y < 6; ++y)
        [unroll] for (int x = 1; x < 6; ++x) {
            int2 c = int2(local) + int2(x, y);
            float gx = (sLum[c.y][c.x + 1] - sLum[c.y][c.x - 1]) * 0.5;
            float gy = (sLum[c.y + 1][c.x] - sLum[c.y - 1][c.x]) * 0.5;
            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
        }
    jxx /= 25;
    jyy /= 25;
    jxy /= 25;
    float a = jxx - jyy, b = 2 * jxy, inv = 1 / (jxx + jyy + 1e-5);
    input[5 * plane + i] = a * inv;
    input[6 * plane + i] = b * inv;
    input[7 * plane + i] = sqrt(a * a + b * b + 1e-12) * inv;
    if (inputs < 11) return;
    float3 r = ToneMap(raw[q].rgb, exposure);
    input[8 * plane + i] = r.r;
    input[9 * plane + i] = r.g;
    input[10 * plane + i] = r.b;
}
