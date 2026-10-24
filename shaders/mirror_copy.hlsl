// The mirror's final image in and out of the mod's resolve (src/hair.cpp: mirrors). `image` is the
// game's texture, the output of the mirror's final pass, still in UAV state after it. Its descriptor
// is written only when the game submits the command list (the game's own descriptor may be written
// after the pass was recorded): null when it did not check out, then both modes do nothing.
//   mode 0: image -> work (the mod's copy, w x h; past the image's edge its edge repeats)
//   mode 1: result -> image (inside the image)
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0), UAV(u0, numDescriptors=2))"

cbuffer Params : register(b0) {
    uint mode;
    uint w, h;   // the mod's textures (at least the image's size)
    uint pad;
};

Texture2D<float4> result : register(t0);
RWTexture2D<float4> image : register(u0);
RWTexture2D<float4> work : register(u1);

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    uint iw, ih;
    image.GetDimensions(iw, ih);
    if (iw == 0 || ih == 0 || any(id >= uint2(w, h))) return;
    if (mode == 0) {
        work[id] = image[min(id, uint2(iw, ih) - 1)];
    } else if (all(id < uint2(iw, ih))) {
        image[id] = result[id];
    }
}
