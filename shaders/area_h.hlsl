// Mod area pass 1 ([Hair] Area): per pixel, the horizontal distance to the nearest
// strand edge in its row, i.e. a pixel where the background shows through the hair (or no hair).
// Pass 2 (area.hlsl) turns it into a distance to the nearest edge in any direction.
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(SRV(t0, numDescriptors=6), UAV(u0, numDescriptors=4))"

cbuffer Params : register(b0) {
    uint2 size;
    float width;      // pixels from an edge that keep the resolve; < 0: all hair
    float feather;    // pixels over which it fades out beyond that
    uint debugView;
    uint3 pad;
};

Texture2D<float> transmittance : register(t1);   // our coverage target (R8): 1 = no hair
RWTexture2D<float4> distOut : register(u0);      // .x: row distance (the tensor target, free by now)

static const float kEdge = 0.05;   // transmittance above this: background shows through

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= size) || width < 0) return;
    int r = int(ceil(width + feather));
    float d = r + 1;
    for (int x = -r; x <= r; ++x) {
        int2 p = int2(clamp(int(id.x) + x, 0, int(size.x) - 1), id.y);
        if (transmittance[p] > kEdge) d = min(d, abs(x));
    }
    distOut[id] = float4(d, 0, 0, 0);
}
