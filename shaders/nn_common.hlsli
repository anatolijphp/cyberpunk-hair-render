// Shared by the hair-detail network passes (src/neural.cpp). One root signature for all of them:
// 16 root constants, SRV t0..t3 (textures / weights), UAV u0..u3 (network buffers, hair box).
// Network buffers are typed 16-bit float buffers, plane-major: index = (channel * size + y) * size + x.
#define RS "RootConstants(num32BitConstants=16, b0), " \
           "DescriptorTable(SRV(t0, numDescriptors=4), UAV(u0, numDescriptors=4)), " \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, " \
           "addressV=TEXTURE_ADDRESS_CLAMP)"

RWBuffer<uint> hairBox : register(u3);   // display pixels: min x, min y, max x, max y (inclusive)

// false when the chunk [rect.xy, rect.zw) has no hair: every pass of that chunk does nothing
bool ChunkHasHair(int4 rect) {
    int4 box = int4(hairBox[0], hairBox[1], hairBox[2], hairBox[3]);
    return box.x <= box.z && rect.x <= box.z && rect.z > box.x && rect.y <= box.w && rect.w > box.y;
}

// false when the tile of a thread group (tileSize buffer pixels at level `level`, buffer pixel 0 at display
// pixel `origin`) is farther than HAIR_MARGIN display pixels from the hair box: nothing it computes
// reaches a hair pixel (the network sees about 24 pixels around a pixel), so the whole group skips
#define HAIR_MARGIN 32
bool TileNearHair(int2 origin, uint level, uint2 group, uint tileSize) {
    int4 box = int4(hairBox[0], hairBox[1], hairBox[2], hairBox[3]);
    if (box.x > box.z) return false;
    int2 lo = origin + int2(group * tileSize) * int(1u << level);
    int2 hi = lo + int(tileSize) * int(1u << level) - 1;
    return lo.x <= box.z + HAIR_MARGIN && hi.x >= box.x - HAIR_MARGIN && lo.y <= box.w + HAIR_MARGIN &&
           hi.y >= box.y - HAIR_MARGIN;
}

float3 ToneMap(float3 c, float exposure) {
    float3 x = max(c * exposure, 0);
    return x / (1 + x);
}

float3 InverseToneMap(float3 t, float exposure) {
    t = clamp(t, 0, 0.999);
    return t / (1 - t) / exposure;
}

// what the network may change (as in training: hair coverage or surface, x4, widened by 3 px)
float RawHair(float coverage, float kept) { return saturate(max(1 - coverage, kept) * 4); }
