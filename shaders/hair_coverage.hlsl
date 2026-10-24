// Side coverage pass for the game's hair alpha_accum draws (re-issued with the game's VS and
// root signature). Re-implements the four game variants without the stochastic dither and
// without the K-buffer writes:
//   SV_Target0.a  coverage, blended dst * (1 - a) into an R16 target -> transmittance
//   SV_Target1.r  1 where the game's own dither keeps the fragment (blend MAX) -> hair-surface mask
//
// Variants (game PS hash): base 23568b87da57f655, DISSOLVE ad72537bdaf26928,
// FADE c9c37ca4835cb163, DISSOLVE+FADE d6fb2e5e5c795ce9.
// SHARPEN (+ SHARPEN_THRESHOLD, SHARPEN_WIDTH): coverage = sharpened normalised alpha instead of raw.

cbuffer GlobalShaderConsts : register(b0) { uint4 g_global[31]; };
cbuffer CameraShaderConsts : register(b1) { uint4 g_camera[52]; };
cbuffer ShaderSpecificConsts : register(b4) { float4 g_specific[12]; };
cbuffer MaterialModifiersConsts : register(b7) { float4 g_modifiers[3]; };

Texture2D<float4> g_textures[] : register(t0, space1);
Texture2DArray<float4> g_dissolveNoise : register(t67);
SamplerState g_sampler : register(s0);

#if FADE
static const uint kTexIndexReg = 8;
static const uint kCutoffReg = 11;
#else
static const uint kTexIndexReg = 2;
static const uint kCutoffReg = 5;
#endif

struct PSIn {
    float4 pos : SV_Position;
    float4 tc0 : TEXCOORD0;
    float4 tc1 : TEXCOORD1;
    float4 tc2 : TEXCOORD2;
    float4 tc3 : TEXCOORD3;
#if FADE
    float3 tc4 : TEXCOORD4;
#else
    float2 tc4 : TEXCOORD4;
#endif
    bool front : SV_IsFrontFace;
};

struct PSOut {
    float4 coverage : SV_Target0;
    float4 kept : SV_Target1;
};

PSOut main(PSIn i) {
    float2 xy = i.pos.xy;
#if DISSOLVE
    uint3 nc = uint3(uint(xy.x) & 63, uint(xy.y) & 63, g_global[30].y & 63);
    float n = g_dissolveNoise.Load(int4(nc, 0)).x;
    if (n * g_modifiers[0].z + g_modifiers[0].w < 0) discard;
#endif

    uint tex = asuint(g_specific[kTexIndexReg].x);
    float alpha = g_textures[NonUniformResourceIndex(tex)].Sample(g_sampler, float2(i.tc0.w, i.tc1.x)).x;

#if FADE
    // 4x4 ordered fade (distance / LOD fade), kept as in the game
    uint ux = uint(xy.x), uy = uint(xy.y);
    uint b = ((((uy & 1) ^ (ux & 1)) << 1 | (uy & 1)) << 2) | ((((uy >> 1) & 1) << 1) ^ (ux & 2)) | ((uy >> 1) & 1);
    float bayer = saturate(float(b) * (1.0 / 64.0) + (1.0 / 128.0));
    float fade = max(g_modifiers[2].x + i.tc4.z, 0);
    alpha = fade <= 0 ? alpha : alpha - bayer - fade;
#endif

    // the game's dither test, used only to rebuild its hair-surface mask
    float cutoff = g_specific[kCutoffReg].x;
    float frame = float(g_camera[51].z);
    float noise = (frac((xy.x - 1.5 + xy.y * 2.0 + frame) * 0.2) * 5.0 + frac(dot(float2(2.4084506, 3.2535212), xy))) *
                  0.16535948;
    float norm = max(alpha - cutoff, 0) / (1 - cutoff) * (g_camera[51].w != 0 ? 1.33 : 1.0);
    bool kept = norm - 0.0078431 - noise > 0;

    PSOut o;
#if SHARPEN
    // anti-aliased alpha test (Golus): edge where the game's normalised alpha crosses the
    // threshold, about SHARPEN_WIDTH pixels wide
    float cov = saturate((norm - SHARPEN_THRESHOLD) / max(fwidth(norm) * SHARPEN_WIDTH, 1e-4) + 0.5);
    o.coverage = float4(0, 0, 0, cov);
#else
    o.coverage = float4(0, 0, 0, saturate(alpha));
#endif
    o.kept = float4(kept ? 1 : 0, 0, 0, 0);
    return o;
}
