// Mod area pass 2 ([Hair] Area): distance to the nearest strand edge (the larger of the vertical
// offset and that row's horizontal distance, from pass 1) and from it the area: 1 within `width`
// pixels of an edge, fading to 0 over `feather` pixels further into fully filled hair. Outside the
// area FSR gets the game's own colour (and depth / motion) back, as without the mod; the map goes
// to .y next to the row distance so that the unsharp mask (usm.hlsl) stays inside the area too.
// Runs after the resolve (and the network), on its output.
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(SRV(t0, numDescriptors=6), UAV(u0, numDescriptors=4))"

cbuffer Params : register(b0) {
    uint2 size;
    float width;      // < 0: all hair (debug view only)
    float feather;
    uint debugView;
    uint stabilize;   // 1: the resolve wrote depth / motion for FSR too
    uint2 pad;
};

Texture2D<float4> color : register(t0);         // the game's colour (before the resolve)
Texture2D<float> depth : register(t4);
Texture2D<float2> motion : register(t5);
RWTexture2D<float4> area : register(u0);        // .x: row distance (pass 1); .y: the area
RWTexture2D<float4> output : register(u1);      // the resolve
RWTexture2D<float> depthOut : register(u2);
RWTexture2D<float2> motionOut : register(u3);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= size)) return;
    float keep = 1;
    float own = area[id].x;
    if (width >= 0) {
        int r = int(ceil(width + feather));
        float d = r + 1;
        for (int y = -r; y <= r; ++y) {
            int2 p = int2(id.x, clamp(int(id.y) + y, 0, int(size.y) - 1));
            d = min(d, max(abs(y), area[p].x));
        }
        keep = saturate((width + feather - d) / feather);   // 1 within width, 0 beyond width + feather
        keep = keep * keep * (3 - 2 * keep);
    }
    area[id] = float4(own, keep, 0, 0);   // .x unchanged: other threads still read it
    if (keep >= 1) return;
    output[id] = float4(lerp(color[id].rgb, output[id].rgb, keep), output[id].a);
    if (stabilize && keep < 0.5) {
        depthOut[id] = depth[id];
        motionOut[id] = motion[id];
    }
}
