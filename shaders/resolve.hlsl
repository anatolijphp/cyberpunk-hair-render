// Resolve pass 2: guided hair resolve on the upscaler's colour input.
//   out = a * hairColour + (1 - a) * backgroundColour
// a = true coverage (1 - transmittance); hair/background colours come from neighbours that hold
// a hair surface (game's dither kept it) or not, averaged in a 5x5 box or along the strand.
// Partially covered pixels always use the averaged hair colour (independent of the per-frame
// dither), and optionally get the nearest hair depth / motion vector in 5x5, so the upscaler's
// disocclusion test sees stable hair instead of the dither flipping hair and background.
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(SRV(t0, numDescriptors=6), UAV(u0, numDescriptors=4))"

cbuffer Params : register(b0) {
    uint2 size;
    uint mode;        // 0 box, 1 strand
    float strength;   // k: 0 = keep own hair sample, 1 = fully averaged
    uint debugView;   // 0 off, 1 coverage, 2 hair-surface mask, 3 strand coherence
    uint stabilize;   // 1: hand FSR the nearest hair depth / motion on touched pixels
    uint2 pad;
};

Texture2D<float4> color : register(t0);
Texture2D<float> transmittance : register(t1);
Texture2D<float> keptMask : register(t2);
Texture2D<float4> tensor : register(t3);
Texture2D<float> depth : register(t4);          // reversed Z: larger = nearer
Texture2D<float2> motion : register(t5);
RWTexture2D<float4> output : register(u1);
RWTexture2D<float> depthOut : register(u2);
RWTexture2D<float2> motionOut : register(u3);

int2 Clamp(int2 p) { return clamp(p, int2(0, 0), int2(size) - 1); }

// premultiplied hair / background layers at a texel: (rgb * w, w)
void Layers(int2 p, out float4 hair, out float4 bg) {
    p = Clamp(p);
    float3 c = color[p].rgb;
    float w = keptMask[p];
    hair = float4(c * w, w);
    bg = float4(c * (1 - w), 1 - w);
}

void LayersBilinear(float2 p, out float4 hair, out float4 bg) {
    float2 f = frac(p);
    int2 i = int2(floor(p));
    float4 h00, b00, h10, b10, h01, b01, h11, b11;
    Layers(i, h00, b00);
    Layers(i + int2(1, 0), h10, b10);
    Layers(i + int2(0, 1), h01, b01);
    Layers(i + int2(1, 1), h11, b11);
    hair = lerp(lerp(h00, h10, f.x), lerp(h01, h11, f.x), f.y);
    bg = lerp(lerp(b00, b10, f.x), lerp(b01, b11, f.x), f.y);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= size)) return;
    int2 p = int2(id);
    float4 own = color[p];
    float a = 1.0 - transmittance[p];
    bool hairHere = keptMask[p] > 0.5;

    if (debugView != 0) {
        depthOut[p] = depth[p];
        motionOut[p] = motion[p];
        if (debugView == 1) output[p] = float4(a.xxx, own.a);
        else if (debugView == 2) output[p] = float4(hairHere ? 1 : 0, a, 0, own.a);
        else output[p] = float4(tensor[p].zzz * a, own.a);
        return;
    }

    bool touch = (a > 1.0 / 255 && a < 254.0 / 255) || hairHere;
    float ownDepth = depth[p];
    float2 ownMotion = motion[p];
    if (!touch) {
        output[p] = own;
        depthOut[p] = ownDepth;
        motionOut[p] = ownMotion;
        return;
    }

    // nearest surface (hair, being in front) in 5x5: stable depth / motion for the upscaler
    float nearDepth = ownDepth;
    int2 nearAt = p;

    float4 hairSum = 0, bgSum = 0;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x) {
            float4 h, b;
            Layers(p + int2(x, y), h, b);
            hairSum += h;
            bgSum += b;
            int2 q = Clamp(p + int2(x, y));
            float dq = depth[q];
            if (h.a > 0.5 && dq > nearDepth) { nearDepth = dq; nearAt = q; }
        }
    if (stabilize != 0 && a > 0.02) {
        depthOut[p] = nearDepth;
        motionOut[p] = motion[nearAt];
    } else {
        depthOut[p] = ownDepth;
        motionOut[p] = ownMotion;
    }

    if (mode == 1) {
        float4 t = tensor[p];
        float coh = t.z;
        if (coh > 0.01) {
            float4 hairDir = 0, bgDir = 0;
            for (int s = -3; s <= 3; ++s) {
                float4 h, b;
                LayersBilinear(float2(p) + t.xy * s, h, b);
                hairDir += h;
                bgDir += b;
            }
            hairDir *= 25.0 / 7.0;   // same weight as the 25-tap box
            bgDir *= 25.0 / 7.0;
            hairSum = lerp(hairSum, hairDir, coh);
            bgSum = lerp(bgSum, bgDir, coh);
        }
    }

    float3 hairAvg = hairSum.a > 1e-3 ? hairSum.rgb / hairSum.a : own.rgb;
    float3 bgAvg = bgSum.a > 1e-3 ? bgSum.rgb / bgSum.a : own.rgb;
    // the own sample only counts where hair fully covers the pixel; in the translucent fringe the
    // dither decides which pixels hold hair, so only the neighbourhood average is stable
    float k = lerp(1.0, strength, smoothstep(0.9, 0.98, a));
    float3 hairC = hairHere ? lerp(own.rgb, hairAvg, k) : hairAvg;
    float3 bgC = hairHere ? bgAvg : own.rgb;
    output[p] = float4(a * hairC + (1 - a) * bgC, own.a);
}
