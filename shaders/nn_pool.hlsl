// 2x2 average pooling of all channels (z of the dispatch = channel). Groups far from the hair box skip.
#include "nn_common.hlsli"

// scalars only: root constants are packed tightly, as the C++ structs in neural.cpp
cbuffer Params : register(b0) {
    uint size;            // output resolution (square); the input is 2 * size
    uint channels;
    int originX, originY; // display pixel of full-resolution buffer pixel (0, 0)
    uint level;           // the output's resolution: full >> level
    uint pad0, pad1, pad2, pad3, pad4, pad5, pad6, pad7, pad8, pad9, pad10;
};

RWBuffer<float> dst : register(u0);
RWBuffer<float> src : register(u1);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    if (any(id.xy >= size) || id.z >= channels || !TileNearHair(int2(originX, originY), level, group.xy, 8)) return;
    uint s = size * 2;
    uint base = id.z * s * s + id.y * 2 * s + id.x * 2;
    dst[(id.z * size + id.y) * size + id.x] = 0.25 * (src[base] + src[base + 1] + src[base + s] + src[base + s + 1]);
}
