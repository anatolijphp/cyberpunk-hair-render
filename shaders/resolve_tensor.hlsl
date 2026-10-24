// Resolve pass 1: local strand direction from the structure tensor of the hair coverage.
// out = (tangent.x, tangent.y, coherence, 0); coherence 0 where there is no clear direction.
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(SRV(t0, numDescriptors=6), UAV(u0, numDescriptors=4))"

cbuffer Params : register(b0) {
    uint2 size;
    uint mode;        // 0 box, 1 strand
    float strength;
    uint debugView;
    uint stabilize;   // unused here
    uint2 pad;
};

Texture2D<float> transmittance : register(t1);   // our coverage target (R16): 1 = no hair
RWTexture2D<float4> tensorOut : register(u0);

static const int kTile = 16;
static const int kBox = 3;                   // 7x7 tensor smoothing
static const int kApron = kBox + 1;          // + 1 for the Sobel taps
static const int kIn = kTile + 2 * kApron;   // 24
static const int kGrad = kTile + 2 * kBox;   // 22

groupshared float s_alpha[kIn][kIn];
groupshared float3 s_j[kGrad][kGrad];        // (gx*gx, gy*gy, gx*gy)

float AlphaAt(int2 p) {
    p = clamp(p, int2(0, 0), int2(size) - 1);
    return 1.0 - transmittance[p];
}

[RootSignature(RS)]
[numthreads(kTile, kTile, 1)]
void main(uint2 gid : SV_GroupID, uint2 tid : SV_GroupThreadID) {
    int2 base = int2(gid) * kTile - kApron;
    for (int i = tid.y * kTile + tid.x; i < kIn * kIn; i += kTile * kTile)
        s_alpha[i / kIn][i % kIn] = AlphaAt(base + int2(i % kIn, i / kIn));
    GroupMemoryBarrierWithGroupSync();

    for (int i = tid.y * kTile + tid.x; i < kGrad * kGrad; i += kTile * kTile) {
        int x = i % kGrad + 1, y = i / kGrad + 1;
        float gx = (s_alpha[y - 1][x + 1] + 2 * s_alpha[y][x + 1] + s_alpha[y + 1][x + 1]) -
                   (s_alpha[y - 1][x - 1] + 2 * s_alpha[y][x - 1] + s_alpha[y + 1][x - 1]);
        float gy = (s_alpha[y + 1][x - 1] + 2 * s_alpha[y + 1][x] + s_alpha[y + 1][x + 1]) -
                   (s_alpha[y - 1][x - 1] + 2 * s_alpha[y - 1][x] + s_alpha[y - 1][x + 1]);
        s_j[i / kGrad][i % kGrad] = float3(gx * gx, gy * gy, gx * gy);
    }
    GroupMemoryBarrierWithGroupSync();

    int2 p = int2(gid) * kTile + int2(tid);
    if (any(p >= int2(size))) return;
    float3 j = 0;
    for (int y = 0; y <= 2 * kBox; ++y)
        for (int x = 0; x <= 2 * kBox; ++x) j += s_j[tid.y + y][tid.x + x];
    j /= float((2 * kBox + 1) * (2 * kBox + 1));

    float tr = j.x + j.y;
    float diff = sqrt((j.x - j.y) * (j.x - j.y) + 4 * j.z * j.z);
    float coh = tr > 1e-6 ? diff / tr : 0;
    float theta = 0.5 * atan2(2 * j.z, j.x - j.y) + 1.5707963;   // along the strand
    tensorOut[p] = float4(cos(theta), sin(theta), coh * coh, 0);
}
