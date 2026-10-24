// Unsharp mask on hair only, on the upscaler output (display resolution):
//   out = c + amount * (c - gaussian(c, radius)), weighted by hair coverage and a soft threshold
// The hair coverage comes from our render-resolution coverage target, sampled bilinearly, limited to
// [Hair] Area pixels from strand edges when the area map is given (area.hlsl).
#define RS "RootConstants(num32BitConstants=12, b0), DescriptorTable(SRV(t0, numDescriptors=3), UAV(u0)), " \
           "StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, " \
           "addressV=TEXTURE_ADDRESS_CLAMP)"

cbuffer Params : register(b0) {
    uint2 displaySize;
    float2 renderSize;      // part of the coverage texture that holds this frame
    float2 coverageTexSize;
    float amount;
    float radius;           // gaussian sigma, display pixels
    float threshold;        // relative to local brightness; smaller differences are left alone
    uint debugView;         // 4: show the sharpening mask, 7: the area
    uint useArea;           // 1: limit to the area map
    uint pad;
};

Texture2D<float4> source : register(t0);          // copy of the upscaler output
Texture2D<float> transmittance : register(t1);    // our coverage target: 1 = no hair
Texture2D<float4> areaMap : register(t2);         // .y: 1 inside the area (same size as the coverage)
SamplerState linearClamp : register(s0);
RWTexture2D<float4> output : register(u0);

float Luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= displaySize)) return;
    float2 uv = (float2(id) + 0.5) / float2(displaySize) * renderSize / coverageTexSize;
    float hair = 1.0 - transmittance.SampleLevel(linearClamp, uv, 0);
    float area = useArea ? areaMap.SampleLevel(linearClamp, uv, 0).y : 1;
    float4 c = source[id];
    if (debugView == 7) {   // red: hair outside the area, white: inside
        output[id] = float4(hair.xxx * float3(1, area, area), c.a);
        return;
    }
    hair *= area;
    if (debugView == 4) {
        output[id] = float4(hair.xxx, c.a);
        return;
    }
    if (hair < 0.01) return;   // output already holds the unmodified pixel

    int r = clamp(int(ceil(radius * 2.0)), 1, 4);
    float inv2s2 = 1.0 / (2.0 * radius * radius);
    float3 blur = 0;
    float wsum = 0;
    for (int y = -r; y <= r; ++y)
        for (int x = -r; x <= r; ++x) {
            float w = exp(-(x * x + y * y) * inv2s2);
            int2 q = clamp(int2(id) + int2(x, y), int2(0, 0), int2(displaySize) - 1);
            blur += source[q].rgb * w;
            wsum += w;
        }
    blur /= wsum;

    float3 detail = c.rgb - blur;
    float rel = abs(Luma(detail)) / (Luma(blur) + 1e-3);
    float gate = smoothstep(threshold, threshold * 2.0, rel);
    float3 sharpened = max(c.rgb + amount * detail * gate, 0);
    output[id] = float4(lerp(c.rgb, sharpened, saturate(hair)), c.a);
}
