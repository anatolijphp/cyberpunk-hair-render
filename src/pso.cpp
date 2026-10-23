// Hair PSO classification from graphics PSO descs and pipeline state streams.
#define CINTERFACE
#define COBJMACROS
#include "pso.h"

#include <climits>
#include <cstdint>
#include <cstring>

#include "common.h"

namespace hf {

const char* KindName(HairKind k) {
    switch (k) {
        case HairKind::AlphaAccum: return "alpha_accum";
        case HairKind::BasecolorBlend: return "basecolor_blend";
        case HairKind::GBuffer: return "gbuffer_solid";
        default: return "none";
    }
}

HairKind Classify(const PsoInfo& p) {
    const auto& b = p.blend0;
    if (p.numRT == 1 && p.rtv[0] == DXGI_FORMAT_A8_UNORM && b.BlendEnable &&
        b.SrcBlendAlpha == D3D12_BLEND_ZERO && b.DestBlendAlpha == D3D12_BLEND_INV_SRC_ALPHA &&
        b.RenderTargetWriteMask == D3D12_COLOR_WRITE_ENABLE_ALPHA)
        return HairKind::AlphaAccum;
    if (p.numRT == 1 && p.rtv[0] == DXGI_FORMAT_R16G16B16A16_FLOAT && b.BlendEnable &&
        b.SrcBlend == D3D12_BLEND_ONE && b.DestBlend == D3D12_BLEND_ONE && p.stencilEnable &&
        p.stencilFunc == D3D12_COMPARISON_FUNC_EQUAL && p.stencilReadMask == 0xE0 && !p.depthWrite)
        return HairKind::BasecolorBlend;
    if (p.numRT >= 3 && p.rtv[0] == DXGI_FORMAT_R10G10B10A2_UNORM &&
        p.rtv[1] == DXGI_FORMAT_R10G10B10A2_UNORM && p.rtv[2] == DXGI_FORMAT_R8G8B8A8_UNORM &&
        p.stencilEnable && p.stencilFunc == D3D12_COMPARISON_FUNC_EQUAL &&
        p.stencilReadMask == 0xE0 && p.depthWrite)
        return HairKind::GBuffer;
    return HairKind::None;
}

static void FillFromDepthStencil(PsoInfo& p, BOOL depthEnable, D3D12_DEPTH_WRITE_MASK write, BOOL stencil,
                          D3D12_COMPARISON_FUNC func, UINT8 readMask, UINT8 writeMask) {
    p.depthWrite = depthEnable && write == D3D12_DEPTH_WRITE_MASK_ALL;
    p.stencilEnable = stencil;
    p.stencilFunc = func;
    p.stencilReadMask = readMask;
    p.stencilWriteMask = writeMask;
}

PsoInfo FromGraphicsDesc(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* d) {
    PsoInfo p;
    p.vs = d->VS;
    p.ps = d->PS;
    p.rootSig = d->pRootSignature;
    p.numRT = d->NumRenderTargets;
    for (UINT i = 0; i < 8; ++i) p.rtv[i] = d->RTVFormats[i];
    p.dsv = d->DSVFormat;
    p.blend0 = d->BlendState.RenderTarget[0];
    const auto& ds = d->DepthStencilState;
    FillFromDepthStencil(p, ds.DepthEnable, ds.DepthWriteMask, ds.StencilEnable, ds.FrontFace.StencilFunc,
                         ds.StencilReadMask, ds.StencilWriteMask);
    return p;
}

// Pipeline state streams: each subobject is `alignas(void*) { TYPE; Inner; }` (d3dx12 layout).
// Subobject types newer than the mingw headers, with sizes from the D3D12 spec:
constexpr UINT kDepthStencil2 = 0x1a;   // D3D12_DEPTH_STENCIL_DESC2: 60 bytes, align 4
constexpr UINT kRasterizer1 = 0x1b;     // D3D12_RASTERIZER_DESC1: 44 bytes, align 4
constexpr UINT kRasterizer2 = 0x1c;     // D3D12_RASTERIZER_DESC2: 40 bytes, align 4
constexpr UINT kSerializedRootSig = 0x1d;  // D3D12_SERIALIZED_ROOT_SIGNATURE_DESC: 16, align 8

static bool InnerLayout(UINT type, size_t& size, size_t& align) {
#define L(T) size = sizeof(T); align = alignof(T); return true
    switch (type) {
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE: L(ID3D12RootSignature*);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS:
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS: L(D3D12_SHADER_BYTECODE);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT: L(D3D12_STREAM_OUTPUT_DESC);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND: L(D3D12_BLEND_DESC);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK: L(UINT);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER: L(D3D12_RASTERIZER_DESC);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL: L(D3D12_DEPTH_STENCIL_DESC);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT: L(D3D12_INPUT_LAYOUT_DESC);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE: L(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY: L(D3D12_PRIMITIVE_TOPOLOGY_TYPE);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS: L(D3D12_RT_FORMAT_ARRAY);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT: L(DXGI_FORMAT);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC: L(DXGI_SAMPLE_DESC);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK: L(UINT);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO: L(D3D12_CACHED_PIPELINE_STATE);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS: L(D3D12_PIPELINE_STATE_FLAGS);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1: L(D3D12_DEPTH_STENCIL_DESC1);
        case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING: L(D3D12_VIEW_INSTANCING_DESC);
        case kDepthStencil2: size = 60; align = 4; return true;
        case kRasterizer1: size = 44; align = 4; return true;
        case kRasterizer2: size = 40; align = 4; return true;
        case kSerializedRootSig: size = 16; align = 8; return true;
        default: return false;
    }
#undef L
}

bool FromStream(const D3D12_PIPELINE_STATE_STREAM_DESC* d, PsoInfo& p) {
    auto* cur = static_cast<const uint8_t*>(d->pPipelineStateSubobjectStream);
    auto* end = cur + d->SizeInBytes;
    bool haveDS = false;
    while (cur < end) {
        UINT type = *reinterpret_cast<const UINT*>(cur);
        size_t size, align;
        if (!InnerLayout(type, size, align)) {
            Log("stream: unknown subobject type 0x%x, skipping PSO", type);
            return false;
        }
        size_t innerOff = (sizeof(UINT) + align - 1) & ~(align - 1);
        const uint8_t* in = cur + innerOff;
        switch (type) {
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
                p.rootSig = *reinterpret_cast<ID3D12RootSignature* const*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS: p.vs = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS: p.ps = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND:
                p.blend0 = reinterpret_cast<const D3D12_BLEND_DESC*>(in)->RenderTarget[0];
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS: {
                auto* a = reinterpret_cast<const D3D12_RT_FORMAT_ARRAY*>(in);
                p.numRT = a->NumRenderTargets;
                for (UINT i = 0; i < 8; ++i) p.rtv[i] = a->RTFormats[i];
                break;
            }
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT:
                p.dsv = *reinterpret_cast<const DXGI_FORMAT*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL: {
                auto* ds = reinterpret_cast<const D3D12_DEPTH_STENCIL_DESC*>(in);
                FillFromDepthStencil(p, ds->DepthEnable, ds->DepthWriteMask, ds->StencilEnable,
                                     ds->FrontFace.StencilFunc, ds->StencilReadMask, ds->StencilWriteMask);
                haveDS = true;
                break;
            }
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1: {
                auto* ds = reinterpret_cast<const D3D12_DEPTH_STENCIL_DESC1*>(in);
                FillFromDepthStencil(p, ds->DepthEnable, ds->DepthWriteMask, ds->StencilEnable,
                                     ds->FrontFace.StencilFunc, ds->StencilReadMask, ds->StencilWriteMask);
                haveDS = true;
                break;
            }
            case kDepthStencil2: {
                // {BOOL DepthEnable; WRITE_MASK; FUNC; BOOL StencilEnable;
                //  STENCILOP_DESC1 FrontFace {4 x enum, UINT8 read, UINT8 write}; ...}
                auto* u = reinterpret_cast<const UINT*>(in);
                FillFromDepthStencil(p, u[0], (D3D12_DEPTH_WRITE_MASK)u[1], u[3], (D3D12_COMPARISON_FUNC)u[7],
                                     in[32], in[33]);
                haveDS = true;
                break;
            }
            default: break;
        }
        size_t total = (innerOff + size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
        cur += total;
    }
    if (!haveDS) {  // stream default: depth on, write all, stencil off
        p.depthWrite = true;
        p.stencilEnable = false;
    }
    return true;
}

bool StreamToGraphicsDesc(const D3D12_PIPELINE_STATE_STREAM_DESC* sd, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) {
    // stream defaults (D3D12 spec): everything zero except the members below
    d = {};
    d.SampleMask = UINT_MAX;
    d.SampleDesc.Count = 1;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    d.RasterizerState.DepthClipEnable = TRUE;
    d.DepthStencilState.DepthEnable = TRUE;
    d.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    d.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    d.DepthStencilState.StencilReadMask = 0xff;
    d.DepthStencilState.StencilWriteMask = 0xff;
    const D3D12_DEPTH_STENCILOP_DESC defOp = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                              D3D12_COMPARISON_FUNC_ALWAYS};
    d.DepthStencilState.FrontFace = defOp;
    d.DepthStencilState.BackFace = defOp;
    for (auto& rt : d.BlendState.RenderTarget) {
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
        rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
    }

    auto* cur = static_cast<const uint8_t*>(sd->pPipelineStateSubobjectStream);
    auto* end = cur + sd->SizeInBytes;
    while (cur < end) {
        UINT type = *reinterpret_cast<const UINT*>(cur);
        size_t size, align;
        if (!InnerLayout(type, size, align)) return false;
        size_t innerOff = (sizeof(UINT) + align - 1) & ~(align - 1);
        const uint8_t* in = cur + innerOff;
        switch (type) {
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
                d.pRootSignature = *reinterpret_cast<ID3D12RootSignature* const*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS: d.VS = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS: d.PS = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS: d.DS = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS: d.HS = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS: d.GS = *reinterpret_cast<const D3D12_SHADER_BYTECODE*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT:
                d.StreamOutput = *reinterpret_cast<const D3D12_STREAM_OUTPUT_DESC*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND: d.BlendState = *reinterpret_cast<const D3D12_BLEND_DESC*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK: d.SampleMask = *reinterpret_cast<const UINT*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER:
                d.RasterizerState = *reinterpret_cast<const D3D12_RASTERIZER_DESC*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL:
                d.DepthStencilState = *reinterpret_cast<const D3D12_DEPTH_STENCIL_DESC*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1: {
                auto* ds = reinterpret_cast<const D3D12_DEPTH_STENCIL_DESC1*>(in);
                memcpy(&d.DepthStencilState, ds, sizeof(D3D12_DEPTH_STENCIL_DESC));   // same prefix layout
                break;
            }
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT:
                d.InputLayout = *reinterpret_cast<const D3D12_INPUT_LAYOUT_DESC*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE:
                d.IBStripCutValue = *reinterpret_cast<const D3D12_INDEX_BUFFER_STRIP_CUT_VALUE*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY:
                d.PrimitiveTopologyType = *reinterpret_cast<const D3D12_PRIMITIVE_TOPOLOGY_TYPE*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS: {
                auto* a = reinterpret_cast<const D3D12_RT_FORMAT_ARRAY*>(in);
                d.NumRenderTargets = a->NumRenderTargets;
                for (UINT i = 0; i < 8; ++i) d.RTVFormats[i] = a->RTFormats[i];
                break;
            }
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT:
                d.DSVFormat = *reinterpret_cast<const DXGI_FORMAT*>(in);
                break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC: d.SampleDesc = *reinterpret_cast<const DXGI_SAMPLE_DESC*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK: d.NodeMask = *reinterpret_cast<const UINT*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS: d.Flags = *reinterpret_cast<const D3D12_PIPELINE_STATE_FLAGS*>(in); break;
            case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO: break;   // never reused for our PSOs
            default: return false;   // mesh shaders, view instancing, DEPTH_STENCIL2, RASTERIZER1/2, ...
        }
        cur += (innerOff + size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
    }
    return true;
}

}  // namespace hf
