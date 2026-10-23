// Hair PSO classification.
#pragma once

#include <d3d12.h>

namespace hf {

enum class HairKind { None, AlphaAccum, BasecolorBlend, GBuffer };

struct PsoInfo {
    D3D12_SHADER_BYTECODE vs{}, ps{}, cs{};
    ID3D12RootSignature* rootSig = nullptr;
    UINT numRT = 0;
    DXGI_FORMAT rtv[8]{};
    DXGI_FORMAT dsv = DXGI_FORMAT_UNKNOWN;
    D3D12_RENDER_TARGET_BLEND_DESC blend0{};
    bool depthWrite = false;
    bool stencilEnable = false;
    D3D12_COMPARISON_FUNC stencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    UINT8 stencilReadMask = 0xff;
    UINT8 stencilWriteMask = 0xff;
};

const char* KindName(HairKind k);
HairKind Classify(const PsoInfo& p);
PsoInfo FromGraphicsDesc(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* d);
bool FromStream(const D3D12_PIPELINE_STATE_STREAM_DESC* d, PsoInfo& p);
// Full graphics desc from a stream (false if it uses mesh shaders, view instancing or newer types).
bool StreamToGraphicsDesc(const D3D12_PIPELINE_STATE_STREAM_DESC* s, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d);

}  // namespace hf
