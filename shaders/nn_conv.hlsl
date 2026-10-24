// 3x3 convolution (zero padding) + bias + optional LeakyReLU(0.1). A group computes 8 output channels
// (z of the dispatch picks which) for an 8x8 pixel tile: input channels go through groupshared memory
// 8 at a time (the 10x10 tile they need, and their weights), so every value is read from memory once
// per group instead of once per thread. Groups far from the hair box skip (TileNearHair).
// The input is the concatenation of A (optionally at half resolution, upsampled x2 nearest) and B;
// weights as PyTorch stores them: [out][in][3][3], bias[out].
// HALF (cs_6_4, -enable-16bit-types): inputs and weights go through groupshared memory as 16-bit
// pairs of input channels and are multiplied with dot2add (two 16-bit products added to a 32-bit
// sum), twice the math per instruction; the sums stay 32-bit.
#include "nn_common.hlsli"

// scalars only: root constants are packed tightly, as the C++ structs in neural.cpp
cbuffer Params : register(b0) {
    uint size;            // output and B resolution (square)
    uint cinA;
    uint cinB;
    uint cout;
    uint upA;             // 1: A is at size / 2
    uint wOffset;         // first weight of this layer in the weights buffer
    uint act;             // 1: LeakyReLU
    int originX, originY; // display pixel of full-resolution buffer pixel (0, 0)
    uint level;           // this layer's resolution: full >> level
    uint round16;         // 16-bit buffers: round to nearest here (stores may round toward zero, which
                          // drifts over the layers)
    uint pad1, pad2, pad3, pad4, pad5;
};

// x rounded to the nearest 16-bit float mantissa (ties to even) with integer maths: f32tof16 may round
// toward zero too. Exact for the 16-bit normal range, which is all that matters here.
float RoundToHalf(float x) {
    uint u = asuint(x);
    return asfloat((u + 0xfffu + ((u >> 13) & 1u)) & ~0x1fffu);
}

Buffer<float> weights : register(t3);
RWBuffer<float> dst : register(u0);
RWBuffer<float> srcA : register(u1);
RWBuffer<float> srcB : register(u2);

#define TILE 8
#ifdef HALF
#define BLOCK 4           // input channels per groupshared pass (more is slower: registers)
groupshared half2 sIn[BLOCK / 2][TILE + 2][TILE + 2];   // [input channel pair][y][x]
groupshared half2 sW[8][BLOCK / 2][9];                  // [output channel][input channel pair][tap]
#else
#define BLOCK 8           // input channels per groupshared pass
groupshared float sIn[BLOCK][TILE + 2][TILE + 2];
groupshared float sW[8][BLOCK][9];   // [output channel][input channel][tap]
#endif

float LoadIn(uint ci, int2 p) {
    if (any(p < 0) || any(p >= int(size))) return 0;
    if (ci < cinA) {
        uint sa = upA ? size / 2 : size;
        uint2 a = upA ? uint2(p) / 2 : uint2(p);
        return srcA[(ci * sa + a.y) * sa + a.x];
    }
    return srcB[((ci - cinA) * size + p.y) * size + p.x];
}

[RootSignature(RS)]
[numthreads(TILE, TILE, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID,
          uint index : SV_GroupIndex) {
    if (!TileNearHair(int2(originX, originY), level, group.xy, TILE)) return;   // whole group
    const uint cin = cinA + cinB, co0 = group.z * 8;
    const int2 base = int2(group.xy * TILE) - 1;
    float acc[8];
    [unroll] for (uint k = 0; k < 8; ++k) acc[k] = 0;

    for (uint c0 = 0; c0 < cin; c0 += BLOCK) {
#ifdef HALF
        for (uint i = index; i < BLOCK / 2 * (TILE + 2) * (TILE + 2); i += TILE * TILE) {
            uint ch = i / ((TILE + 2) * (TILE + 2)), r = i % ((TILE + 2) * (TILE + 2));
            uint y = r / (TILE + 2), x = r % (TILE + 2);
            uint c = c0 + 2 * ch;
            int2 p = base + int2(x, y);
            sIn[ch][y][x] = half2(c < cin ? LoadIn(c, p) : 0, c + 1 < cin ? LoadIn(c + 1, p) : 0);
        }
        for (uint i = index; i < 8 * BLOCK / 2 * 9; i += TILE * TILE) {
            uint co = i / (BLOCK / 2 * 9), ci = c0 + 2 * ((i / 9) % (BLOCK / 2)), t = i % 9;
            uint w = wOffset + ((co0 + co) * cin + ci) * 9 + t;
            bool ok = co0 + co < cout;
            sW[co][(i / 9) % (BLOCK / 2)][t] = half2(ok && ci < cin ? weights[w] : 0, ok && ci + 1 < cin ? weights[w + 9] : 0);
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint cp = 0; cp < BLOCK / 2; ++cp) {
            half2 v[9];
            [unroll] for (uint j = 0; j < 9; ++j) v[j] = sIn[cp][local.y + j / 3][local.x + j % 3];
            [unroll] for (uint k = 0; k < 8; ++k)
                [unroll] for (uint j = 0; j < 9; ++j) acc[k] = dot2add(sW[k][cp][j], v[j], acc[k]);
        }
#else
        for (uint i = index; i < BLOCK * (TILE + 2) * (TILE + 2); i += TILE * TILE) {
            uint ch = i / ((TILE + 2) * (TILE + 2)), r = i % ((TILE + 2) * (TILE + 2));
            uint y = r / (TILE + 2), x = r % (TILE + 2);
            sIn[ch][y][x] = c0 + ch < cin ? LoadIn(c0 + ch, base + int2(x, y)) : 0;
        }
        for (uint i = index; i < 8 * BLOCK * 9; i += TILE * TILE) {
            uint co = i / (BLOCK * 9), ci = (i / 9) % BLOCK, t = i % 9;
            sW[co][ci][t] = co0 + co < cout && c0 + ci < cin ? weights[wOffset + ((co0 + co) * cin + c0 + ci) * 9 + t] : 0;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint ci = 0; ci < BLOCK; ++ci) {
            float v[9];
            [unroll] for (uint j = 0; j < 9; ++j) v[j] = sIn[ci][local.y + j / 3][local.x + j % 3];
            [unroll] for (uint k = 0; k < 8; ++k)
                [unroll] for (uint j = 0; j < 9; ++j) acc[k] += sW[k][ci][j] * v[j];
        }
#endif
        GroupMemoryBarrierWithGroupSync();
    }

    if (any(id.xy >= size)) return;
    uint biasOffset = wOffset + cout * cin * 9;
    [unroll] for (uint k = 0; k < 8; ++k) {
        uint co = co0 + k;
        if (co >= cout) break;
        float x = acc[k] + weights[biasOffset + co];
        if (act) x = x > 0 ? x : 0.1 * x;
        if (round16) x = RoundToHalf(x);
        dst[(co * size + id.y) * size + id.x] = x;
    }
}
