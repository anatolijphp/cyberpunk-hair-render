// On-screen status banner ("HAIR FIX ON" / "HAIR FIX OFF"), drawn into the upscaler output.
// 5x7 bitmap glyphs, one thread per output pixel of the banner rectangle.
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(UAV(u0))"

cbuffer Params : register(b0) {
    uint2 origin;       // top-left of the banner, in output pixels
    uint scale;         // pixels per glyph dot
    uint numChars;
    uint4 text;         // glyph ids, 4 per uint (8 bits each)
};

RWTexture2D<float4> output : register(u0);

// glyph rows, 5 bits each (MSB = leftmost column); ids: 0 ' ' 1 H 2 A 3 I 4 R 5 F 6 X 7 O 8 N
static const uint kGlyphs[9 * 7] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,   // ' '
    0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11,   // H
    0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11,   // A
    0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E,   // I
    0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11,   // R
    0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10,   // F
    0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11,   // X
    0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E,   // O
    0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11,   // N
};

static const uint kPad = 2;       // dots of padding around the text
static const uint kCellW = 6;     // 5 dots + 1 spacing

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID) {
    uint w = (numChars * kCellW - 1 + 2 * kPad) * scale;
    uint h = (7 + 2 * kPad) * scale;
    if (id.x >= w || id.y >= h) return;

    uint2 dot = id / scale;                     // position in glyph dots, including padding
    bool lit = false;
    if (dot.x >= kPad && dot.y >= kPad && dot.y < kPad + 7) {
        uint cx = dot.x - kPad;
        uint c = cx / kCellW, col = cx % kCellW;
        if (c < numChars && col < 5) {
            uint g = (text[c / 4] >> ((c % 4) * 8)) & 0xFF;
            uint row = kGlyphs[g * 7 + (dot.y - kPad)];
            lit = (row >> (4 - col)) & 1;
        }
    }

    uint2 p = origin + id;
    float4 src = output[p];
    float3 rgb = lit ? float3(1.0, 0.85, 0.2) : src.rgb * 0.25;   // amber text on a darkened box
    output[p] = float4(rgb, src.a);
}
