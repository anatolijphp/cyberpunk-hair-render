// Hair resolve network: see neural.h.
//
// The network (training/train.py) is a small U-Net: it takes the mod's tone-mapped resolve, the
// hair coverage / hair-surface mask, the resolve's line direction and the game's raw frame, and
// predicts a correction of the resolve, applied only on hair. It runs in chunks of kChunk pixels plus
// a kHalo border (what the network sees around a pixel); thread groups far from the hair box
// (computed on the GPU) skip their work. Intermediate data lives in typed 16-bit float buffers
// (shaders/nn_*.hlsl), one layer per dispatch. The Network class takes any of train.py's networks.
#include "neural.h"

#include <d3d12.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "descriptors.h"
#include "fsr2.h"
#include "hair.h"
#include "../build/gen/nn_apply_cso.h"
#include "../build/gen/nn_bbox_cso.h"
#include "../build/gen/nn_conv16_cso.h"
#include "../build/gen/nn_conv_cso.h"
#include "../build/gen/nn_pool_cso.h"
#include "../build/gen/nn_prep_cso.h"

namespace hf {
namespace {

constexpr UINT kChunk = 960, kHalo = 32, kSize = kChunk + 2 * kHalo;   // kSize: multiple of 4 (two poolings)
constexpr UINT kSetSize = 8;      // t0..t3, u0..u3
constexpr UINT kRing = 8;         // per-frame descriptor sets / timestamps in flight
constexpr UINT kFrameSets = 3;    // hair box, input, apply

// network buffers: channels and level (0 full, 1 half, 2 quarter of kSize)
enum Buffer { kIn, kF0, kF1, kE1, kH0, kH1, kE2, kQ0, kQ1, kNumBuffers };

struct Step {
    bool pool;
    Buffer dst, a, b;   // b == kNumBuffers: none
    UINT cinA, cinB, cout, level;
    bool upA, act;
    UINT wOffset;
};

// shared by both networks
ID3D12Device* g_device = nullptr;
ID3D12RootSignature* g_rootSig = nullptr;
ID3D12PipelineState *g_psoBox = nullptr, *g_psoPrep = nullptr, *g_psoConv = nullptr, *g_psoPool = nullptr,
                    *g_psoApply = nullptr;
UINT g_inc = 0;
DXGI_FORMAT g_bufFormat = DXGI_FORMAT_R16_FLOAT;
bool g_halfMath = false;   // convolutions with 16-bit math (nn_conv.hlsl HALF)
UINT g_bufBytes = 2;
UINT64 g_frequency = 0;   // GPU timestamps per second
bool g_sharedTried = false, g_sharedOk = false;

ID3D12Resource* MakeBuffer(UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                           D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    ID3D12Resource* r = nullptr;
    if (FAILED(g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

ID3D12PipelineState* Pso(const unsigned char* cso, size_t size) {
    D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
    d.pRootSignature = g_rootSig;
    d.CS = {cso, size};
    ID3D12PipelineState* p = nullptr;
    return SUCCEEDED(g_device->CreateComputePipelineState(&d, IID_PPV_ARGS(&p))) ? p : nullptr;
}

bool InitShared(ID3D12Device* dev) {
    if (g_sharedTried) return g_sharedOk;
    g_sharedTried = true;
    g_device = dev;
    if (FAILED(dev->CreateRootSignature(0, kNn_convCso, sizeof(kNn_convCso), IID_PPV_ARGS(&g_rootSig)))) {
        Log("neural: root signature failed");
        return false;
    }
    g_psoBox = Pso(kNn_bboxCso, sizeof(kNn_bboxCso));
    g_psoPrep = Pso(kNn_prepCso, sizeof(kNn_prepCso));
    // 16-bit math needs shader model 6.4 and native 16-bit ops (HAIRFIX_NN_FP32=1 turns it off, for tests)
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_4};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
    const char* fp32 = getenv("HAIRFIX_NN_FP32");
    if (!(fp32 && *fp32 == '1') &&
        SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) &&
        sm.HighestShaderModel >= D3D_SHADER_MODEL_6_4 &&
        SUCCEEDED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4))) &&
        o4.Native16BitShaderOpsSupported)
        g_psoConv = Pso(kNn_conv16Cso, sizeof(kNn_conv16Cso));
    g_halfMath = g_psoConv != nullptr;
    if (!g_psoConv) g_psoConv = Pso(kNn_convCso, sizeof(kNn_convCso));
    Log("neural: %s-bit math", g_halfMath ? "16" : "32");
    g_psoPool = Pso(kNn_poolCso, sizeof(kNn_poolCso));
    g_psoApply = Pso(kNn_applyCso, sizeof(kNn_applyCso));
    if (!g_psoBox || !g_psoPrep || !g_psoConv || !g_psoPool || !g_psoApply) {
        Log("neural: pipelines failed");
        return false;
    }
    // 16-bit buffers need typed UAV loads of R16_FLOAT; otherwise 32-bit ones
    D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
    if (FAILED(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o))) ||
        !o.TypedUAVLoadAdditionalFormats) {
        g_bufFormat = DXGI_FORMAT_R32_FLOAT;
        g_bufBytes = 4;
    }
    g_inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (auto* q = static_cast<ID3D12CommandQueue*>(OverlayQueue())) q->GetTimestampFrequency(&g_frequency);
    g_sharedOk = true;
    return true;
}

void Transition(std::vector<D3D12_RESOURCE_BARRIER>& v, ID3D12Resource* r, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    v.push_back(b);
}

void Submit(ID3D12GraphicsCommandList* cl, std::vector<D3D12_RESOURCE_BARRIER>& v) {
    if (!v.empty()) cl->ResourceBarrier((UINT)v.size(), v.data());
    v.clear();
}

void UavBarrier(ID3D12GraphicsCommandList* cl) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // all UAV accesses
    cl->ResourceBarrier(1, &b);
}

template <typename T>
void Constants(ID3D12GraphicsCommandList* cl, const T& c) {
    static_assert(sizeof(T) == 64, "16 root constants");
    cl->SetComputeRoot32BitConstants(0, 16, &c, 0);
}

UINT LevelSize(UINT level) { return kSize >> level; }

class Network {
public:
    Network(const wchar_t* file, const char* label, UINT widen) : file_(file), label_(label), widen_(widen) {}

    // Runs the network on `out` in place (its current content is the base image). coverage / kept:
    // hair targets (NON_PIXEL_SHADER_RESOURCE) covering renderW x renderH of their texture; extra:
    // the raw game frame for networks with 11 inputs (readable by compute), else null. imageW / H: the
    // image's part of `out` (0: all of it; a mirror's targets are larger than its image).
    bool Run(ID3D12GraphicsCommandList* cl, ID3D12Resource* out, D3D12_RESOURCE_STATES outState, ID3D12Resource* cov,
             ID3D12Resource* kept, ID3D12Resource* extra, UINT renderW, UINT renderH, UINT imageW = 0, UINT imageH = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        ID3D12Device* dev = nullptr;
        out->GetDevice(IID_PPV_ARGS(&dev));
        bool ok = dev && InitShared(dev) && Init();
        if (dev) dev->Release();
        if (!ok || (inputs_ == 11 && !extra)) return false;
        return Record(cl, out, outState, cov, kept, extra, renderW, renderH, imageW, imageH);
    }

    void TakeTime(double* ms, UINT* samples) {
        *ms = gpuMs_.exchange(0);
        *samples = gpuSamples_.exchange(0);
    }

private:
    bool LoadWeights() {
        std::wstring path = ModuleDir() + file_;
        FILE* f = _wfopen(path.c_str(), L"rb");
        if (!f) {
            Log("neural %s: %ls not found", label_, path.c_str());
            return false;
        }
        char magic[4];
        uint32_t version = 0, channels = 0, inputs = 0, count = 0;
        float exposure = 0;
        bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "HFNN", 4) == 0 && fread(&version, 4, 1, f) == 1 &&
                  version == 2 && fread(&channels, 4, 1, f) == 1 && fread(&inputs, 4, 1, f) == 1 &&
                  fread(&exposure, 4, 1, f) == 1 && fread(&count, 4, 1, f) == 1 && channels >= 3 &&
                  channels <= 64 && (inputs == 5 || inputs == 8 || inputs == 11) && count < (64u << 20);
        if (ok) {
            weightData_.resize(count);
            ok = fread(weightData_.data(), 4, count, f) == count;
        }
        fclose(f);
        if (!ok) {
            Log("neural %s: %ls is not a hair network file", label_, path.c_str());
            return false;
        }
        channels_ = channels;
        inputs_ = inputs;
        exposure_ = exposure;
        return true;
    }

    // the U-Net of training/train.py, in the order of its weights
    bool BuildSteps() {
        const UINT c = channels_;
        const Buffer none = kNumBuffers;
        struct { Buffer buf; UINT channels, level; } layout[] = {
            {kIn, inputs_, 0}, {kF0, c, 0}, {kF1, c, 0}, {kE1, c, 0}, {kH0, 2 * c, 1}, {kH1, 2 * c, 1},
            {kE2, 2 * c, 1}, {kQ0, 3 * c, 2}, {kQ1, 3 * c, 2}};
        for (auto& l : layout) {
            bufChannels_[l.buf] = l.channels;
            bufLevel_[l.buf] = l.level;
        }
        UINT w = 0;
        auto conv = [&](Buffer dst, Buffer a, UINT cinA, Buffer b, UINT cinB, UINT cout, UINT level, bool upA, bool act) {
            steps_.push_back({false, dst, a, b, cinA, cinB, cout, level, upA, act, w});
            w += cout * (cinA + cinB) * 9 + cout;
        };
        auto pool = [&](Buffer dst, Buffer src, UINT channels, UINT level) {
            steps_.push_back({true, dst, src, none, channels, 0, channels, level, false, false, 0});
        };
        conv(kF0, kIn, inputs_, none, 0, c, 0, false, true);        // e1
        conv(kE1, kF0, c, none, 0, c, 0, false, true);
        pool(kH0, kE1, c, 1);
        conv(kH1, kH0, c, none, 0, 2 * c, 1, false, true);          // e2
        conv(kE2, kH1, 2 * c, none, 0, 2 * c, 1, false, true);
        pool(kQ0, kE2, 2 * c, 2);
        conv(kQ1, kQ0, 2 * c, none, 0, 3 * c, 2, false, true);      // e3
        conv(kQ0, kQ1, 3 * c, none, 0, 3 * c, 2, false, true);
        conv(kH0, kQ0, 3 * c, kE2, 2 * c, 2 * c, 1, true, true);    // d2: up(e3) + e2
        conv(kH1, kH0, 2 * c, none, 0, 2 * c, 1, false, true);
        conv(kF0, kH1, 2 * c, kE1, c, c, 0, true, true);            // d1: up(d2) + e1
        conv(kF1, kF0, c, none, 0, c, 0, false, true);
        conv(kF0, kF1, c, none, 0, 3, 0, false, false);             // out: the correction
        if (w != weightData_.size()) {
            Log("neural %s: weight file has %zu values, a network with %u channels and %u inputs needs %u", label_,
                weightData_.size(), channels_, inputs_, w);
            return false;
        }
        return true;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE Cpu(UINT set, UINT slot) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap_->GetCPUDescriptorHandleForHeapStart();
        h.ptr += (SIZE_T)(set * kSetSize + slot) * g_inc;
        return h;
    }

    D3D12_GPU_DESCRIPTOR_HANDLE Gpu(UINT set) {
        D3D12_GPU_DESCRIPTOR_HANDLE h = heap_->GetGPUDescriptorHandleForHeapStart();
        h.ptr += (UINT64)set * kSetSize * g_inc;
        return h;
    }

    void TextureSrv(UINT set, UINT slot, ID3D12Resource* r, DXGI_FORMAT f) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = f;
        d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Texture2D.MipLevels = 1;
        g_device->CreateShaderResourceView(r, &d, Cpu(set, slot));
    }

    void BufferSrv(UINT set, UINT slot, ID3D12Resource* r, DXGI_FORMAT f, UINT elements) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = f;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = elements;
        g_device->CreateShaderResourceView(r, &d, Cpu(set, slot));
    }

    void BufferUav(UINT set, UINT slot, ID3D12Resource* r, DXGI_FORMAT f, UINT elements) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = f;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements = elements;
        g_device->CreateUnorderedAccessView(r, nullptr, &d, Cpu(set, 4 + slot));
    }

    void NetworkUav(UINT set, UINT slot, Buffer b) {
        if (b == kNumBuffers) {
            BufferUav(set, slot, nullptr, g_bufFormat, 1);
            return;
        }
        UINT n = LevelSize(bufLevel_[b]);
        BufferUav(set, slot, buffers_[b], g_bufFormat, bufChannels_[b] * n * n);
    }

    bool Init() {
        if (tried_) return ok_;
        tried_ = true;
        if (!LoadWeights() || !BuildSteps()) return false;
        UINT64 total = 0;
        for (int b = 0; b < kNumBuffers; ++b) {
            UINT n = LevelSize(bufLevel_[b]);
            UINT64 bytes = (UINT64)bufChannels_[b] * n * n * g_bufBytes;
            buffers_[b] = MakeBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            if (!buffers_[b]) {
                Log("neural %s: buffer %d (%llu bytes) failed", label_, b, (unsigned long long)bytes);
                return false;
            }
            total += bytes;
        }
        box_ = MakeBuffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        UINT64 wbytes = weightData_.size() * 4;
        weights_ = MakeBuffer(wbytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
        upload_ = MakeBuffer(wbytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
        void* p = nullptr;
        if (!box_ || !weights_ || !upload_ || FAILED(upload_->Map(0, nullptr, &p))) {
            Log("neural %s: weight / box buffers failed", label_);
            return false;
        }
        memcpy(p, weightData_.data(), wbytes);
        upload_->Unmap(0, nullptr);

        UINT sets = (UINT)steps_.size() + kRing * kFrameSets;
        D3D12_DESCRIPTOR_HEAP_DESC h{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, sets * kSetSize,
                                     D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        if (FAILED(g_device->CreateDescriptorHeap(&h, IID_PPV_ARGS(&heap_)))) {
            Log("neural %s: descriptor heap failed", label_);
            return false;
        }
        // the steps' sets never change
        for (UINT s = 0; s < steps_.size(); ++s) {
            const Step& st = steps_[s];
            for (UINT t = 0; t < 3; ++t) TextureSrv(s, t, nullptr, DXGI_FORMAT_R8_UNORM);
            BufferSrv(s, 3, weights_, DXGI_FORMAT_R32_FLOAT, (UINT)weightData_.size());
            NetworkUav(s, 0, st.dst);
            NetworkUav(s, 1, st.a);
            NetworkUav(s, 2, st.b);
            BufferUav(s, 3, box_, DXGI_FORMAT_R32_UINT, 4);
        }
        if (g_frequency) {
            D3D12_QUERY_HEAP_DESC q{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2 * kRing, 0};
            void* m = nullptr;
            if (SUCCEEDED(g_device->CreateQueryHeap(&q, IID_PPV_ARGS(&queries_))) &&
                (queryRead_ = MakeBuffer(16 * kRing, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                                         D3D12_RESOURCE_FLAG_NONE)) &&
                SUCCEEDED(queryRead_->Map(0, nullptr, &m)))
                queryData_ = static_cast<const UINT64*>(m);   // stays mapped (readback heap)
            else
                queries_ = nullptr;
        }
        Log("neural %s: %u channels, %u inputs, exposure %.2f, %zu weights, buffers %.0f MB (%s)", label_, channels_,
            inputs_, exposure_, weightData_.size(), total / 1048576.0, g_bufBytes == 2 ? "16-bit" : "32-bit");
        ok_ = true;
        return true;
    }

    // at least the output's size (only grows: a mirror's image changes size every frame)
    ID3D12Resource* CopyTarget(const D3D12_RESOURCE_DESC& out) {
        if (copy_) {
            D3D12_RESOURCE_DESC d = copy_->GetDesc();
            if (d.Width >= out.Width && d.Height >= out.Height && d.Format == out.Format) return copy_;
        }
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = out.Width;
        d.Height = out.Height;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = out.Format;
        d.SampleDesc.Count = 1;
        ID3D12Resource* r = nullptr;   // the old copy is leaked: it may still be in flight
        if (FAILED(g_device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&r))))
            return nullptr;
        copy_ = r;
        copyState_ = D3D12_RESOURCE_STATE_COPY_DEST;
        return copy_;
    }

    bool Record(ID3D12GraphicsCommandList* cl, ID3D12Resource* out, D3D12_RESOURCE_STATES outState,
                ID3D12Resource* cov, ID3D12Resource* kept, ID3D12Resource* extra, UINT renderW, UINT renderH,
                UINT imageW, UINT imageH) {
        D3D12_RESOURCE_DESC od = out->GetDesc();
        DXGI_FORMAT fmt = TypedFormatForView(od.Format);
        if (!(od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || fmt == DXGI_FORMAT_UNKNOWN) return false;
        ID3D12Resource* copy = CopyTarget(od);
        if (!copy) return false;
        // the base image (display or render resolution), the top-left part of `out`
        const UINT dw = imageW ? imageW : (UINT)od.Width, dh = imageH ? imageH : od.Height;
        D3D12_RESOURCE_DESC cd = cov->GetDesc();
        const float uvx = (float)renderW / (float)cd.Width / (float)dw, uvy = (float)renderH / (float)cd.Height / (float)dh;

        // per-frame sets: hair box, input, apply
        UINT base = (UINT)steps_.size() + (ring_++ % kRing) * kFrameSets;
        const UINT setBox = base, setPrep = base + 1, setApply = base + 2;
        for (UINT s : {setBox, setPrep, setApply}) {
            TextureSrv(s, 0, s == setBox ? cov : copy, s == setBox ? kCoverageFormat : fmt);
            TextureSrv(s, 1, s == setBox ? kept : cov, s == setBox ? DXGI_FORMAT_R8_UNORM : kCoverageFormat);
            TextureSrv(s, 2, s == setBox ? nullptr : kept, DXGI_FORMAT_R8_UNORM);
            for (UINT u = 0; u < 3; ++u) BufferUav(s, u, nullptr, g_bufFormat, 1);
            BufferUav(s, 3, box_, DXGI_FORMAT_R32_UINT, 4);
        }
        BufferSrv(setBox, 3, nullptr, DXGI_FORMAT_R32_FLOAT, 1);
        BufferSrv(setApply, 3, nullptr, DXGI_FORMAT_R32_FLOAT, 1);
        if (extra)
            TextureSrv(setPrep, 3, extra, TypedFormatForView(extra->GetDesc().Format));
        else
            TextureSrv(setPrep, 3, nullptr, fmt);
        NetworkUav(setPrep, 0, kIn);
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
        ud.Format = fmt;
        ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        g_device->CreateUnorderedAccessView(out, nullptr, &ud, Cpu(setApply, 4));
        NetworkUav(setApply, 1, kF0);

        std::vector<D3D12_RESOURCE_BARRIER> bars;
        Transition(bars, out, outState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(bars, copy, copyState_, D3D12_RESOURCE_STATE_COPY_DEST);
        Submit(cl, bars);
        D3D12_TEXTURE_COPY_LOCATION to{copy, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}},
            from{out, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        to.SubresourceIndex = from.SubresourceIndex = 0;
        const D3D12_BOX image{0, 0, 0, dw, dh, 1};
        cl->CopyTextureRegion(&to, 0, 0, 0, &from, &image);
        if (!weightsReady_) {
            cl->CopyBufferRegion(weights_, 0, upload_, 0, weightData_.size() * 4);
            Transition(bars, weights_, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            weightsReady_ = true;
        }
        Transition(bars, out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Transition(bars, copy, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Submit(cl, bars);
        copyState_ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        // GPU time: the pair written kRing runs ago is certainly resolved by now
        const UINT ts = (UINT)(runs_++ % kRing);
        if (queries_) {
            if (runs_ > kRing && queryData_[2 * ts + 1] > queryData_[2 * ts]) {
                gpuMs_ = gpuMs_.load() + (double)(queryData_[2 * ts + 1] - queryData_[2 * ts]) * 1000.0 / (double)g_frequency;
                ++gpuSamples_;
            }
            cl->EndQuery(queries_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * ts);
        }

        cl->SetDescriptorHeaps(1, &heap_);
        cl->SetComputeRootSignature(g_rootSig);

        // hair box
        struct { UINT mode, rw, rh; float scale; UINT dw, dh, pad[10]; } box{0, renderW, renderH, (float)dw / renderW, dw, dh, {}};
        cl->SetComputeRootDescriptorTable(1, Gpu(setBox));
        cl->SetPipelineState(g_psoBox);
        Constants(cl, box);
        cl->Dispatch(1, 1, 1);
        UavBarrier(cl);
        box.mode = 1;
        Constants(cl, box);
        cl->Dispatch((renderW + 7) / 8, (renderH + 7) / 8, 1);
        UavBarrier(cl);

        const UINT debug = g_config.debugView == 5 || g_config.debugView == 6 ? (UINT)g_config.debugView : 0;
        for (UINT cy = 0; cy * kChunk < dh; ++cy)
            for (UINT cx = 0; cx * kChunk < dw; ++cx) {
                const int x0 = (int)(cx * kChunk), y0 = (int)(cy * kChunk);
                const int chunk[4] = {x0, y0, (int)std::min(dw, cx * kChunk + kChunk), (int)std::min(dh, cy * kChunk + kChunk)};
                const int ox = x0 - (int)kHalo, oy = y0 - (int)kHalo;

                struct { int ox, oy; UINT size, dw, dh; float uvx, uvy, exposure; int chunk[4]; UINT inputs, pad[3]; } prep{
                    ox, oy, kSize, dw, dh, uvx, uvy, exposure_, {chunk[0], chunk[1], chunk[2], chunk[3]}, inputs_, {}};
                cl->SetComputeRootDescriptorTable(1, Gpu(setPrep));
                cl->SetPipelineState(g_psoPrep);
                Constants(cl, prep);
                cl->Dispatch(kSize / 8, kSize / 8, 1);
                UavBarrier(cl);

                for (UINT s = 0; s < steps_.size(); ++s) {
                    const Step& st = steps_[s];
                    const UINT n = LevelSize(st.level);
                    cl->SetComputeRootDescriptorTable(1, Gpu(s));
                    if (st.pool) {
                        struct { UINT size, channels; int ox, oy; UINT level, pad[11]; } c{n, st.cout, ox, oy, st.level, {}};
                        cl->SetPipelineState(g_psoPool);
                        Constants(cl, c);
                        cl->Dispatch((n + 7) / 8, (n + 7) / 8, st.cout);
                    } else {
                        struct { UINT size, cinA, cinB, cout, upA, wOffset, act; int ox, oy; UINT level, round16, pad[5]; } c{
                            n, st.cinA, st.cinB, st.cout, st.upA, st.wOffset, st.act, ox, oy, st.level, g_bufBytes == 2, {}};
                        cl->SetPipelineState(g_psoConv);
                        Constants(cl, c);
                        cl->Dispatch((n + 7) / 8, (n + 7) / 8, (st.cout + 7) / 8);
                    }
                    UavBarrier(cl);
                }

                struct { int ox, oy; UINT size, dw, dh; float uvx, uvy, exposure; int chunk[4]; UINT debug, widen, pad[2]; } apply{
                    ox, oy, kSize, dw, dh, uvx, uvy, exposure_, {chunk[0], chunk[1], chunk[2], chunk[3]}, debug, widen_, {}};
                cl->SetComputeRootDescriptorTable(1, Gpu(setApply));
                cl->SetPipelineState(g_psoApply);
                Constants(cl, apply);
                cl->Dispatch(kChunk / 8, kChunk / 8, 1);
                UavBarrier(cl);
            }

        Transition(bars, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, outState);
        Submit(cl, bars);
        if (queries_) {
            cl->EndQuery(queries_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * ts + 1);
            cl->ResolveQueryData(queries_, D3D12_QUERY_TYPE_TIMESTAMP, 2 * ts, 2, queryRead_, 16 * ts);
        }
        if (!logged_) {
            logged_ = true;
            Log("neural %s: first run %ux%u (hair targets %ux%u), %u chunks", label_, dw, dh, renderW, renderH,
                ((dw + kChunk - 1) / kChunk) * ((dh + kChunk - 1) / kChunk));
        }
        return true;
    }

    const wchar_t* file_;
    const char* label_;
    UINT widen_;   // hair mask widening in pixels of the base image (as in training)
    std::mutex mutex_;
    bool tried_ = false, ok_ = false, logged_ = false;
    UINT channels_ = 0, inputs_ = 8;
    float exposure_ = 2.0f;
    std::vector<float> weightData_;
    std::vector<Step> steps_;
    ID3D12DescriptorHeap* heap_ = nullptr;   // the steps' sets, then kRing x kFrameSets per-frame sets
    UINT ring_ = 0;
    ID3D12Resource* buffers_[kNumBuffers]{};
    UINT bufChannels_[kNumBuffers]{}, bufLevel_[kNumBuffers]{};
    ID3D12Resource *box_ = nullptr, *weights_ = nullptr, *upload_ = nullptr;
    bool weightsReady_ = false;
    ID3D12Resource* copy_ = nullptr;   // copy of the base image (the network reads around each pixel)
    D3D12_RESOURCE_STATES copyState_ = D3D12_RESOURCE_STATE_COPY_DEST;
    ID3D12QueryHeap* queries_ = nullptr;
    ID3D12Resource* queryRead_ = nullptr;
    const UINT64* queryData_ = nullptr;
    UINT64 runs_ = 0;
    std::atomic<double> gpuMs_{0};
    std::atomic<UINT> gpuSamples_{0};
};

Network g_resolve(L"hairfix_resolve.bin", "resolve", 1);   // render resolution: mask widened by 1 pixel
Network g_mirror(L"hairfix_resolve.bin", "mirror", 1);     // the same network on mirrors (own buffers)

}  // namespace

bool NeuralResolve(void* commandList, void* resolved, unsigned resolvedState, void* raw, void* coverage, void* kept,
                   unsigned renderW, unsigned renderH) {
    if (g_config.resolveMode != 2 || !commandList || !resolved || !raw || !coverage || !kept) return false;
    return g_resolve.Run(static_cast<ID3D12GraphicsCommandList*>(commandList), static_cast<ID3D12Resource*>(resolved),
                         (D3D12_RESOURCE_STATES)resolvedState, static_cast<ID3D12Resource*>(coverage),
                         static_cast<ID3D12Resource*>(kept), static_cast<ID3D12Resource*>(raw), renderW, renderH);
}

bool NeuralResolveMirror(void* commandList, void* resolved, unsigned resolvedState, void* raw, void* coverage,
                         void* kept, unsigned width, unsigned height) {
    if (!commandList || !resolved || !raw || !coverage || !kept) return false;   // the caller checks the setting
    return g_mirror.Run(static_cast<ID3D12GraphicsCommandList*>(commandList), static_cast<ID3D12Resource*>(resolved),
                        (D3D12_RESOURCE_STATES)resolvedState, static_cast<ID3D12Resource*>(coverage),
                        static_cast<ID3D12Resource*>(kept), static_cast<ID3D12Resource*>(raw), width, height, width,
                        height);
}

void PerfOnPresent() {
    static LARGE_INTEGER freq{}, start{};
    static UINT frames = 0;
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (!freq.QuadPart) {
        QueryPerformanceFrequency(&freq);
        start = now;
    }
    ++frames;
    double seconds = (double)(now.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    if (seconds < 10) return;
    double msR, msM;
    UINT nR, nM;
    g_resolve.TakeTime(&msR, &nR);
    g_mirror.TakeTime(&msM, &nM);
    PROCESS_MEMORY_COUNTERS_EX pm{};
    pm.cb = sizeof(pm);
    K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pm, sizeof(pm));
    size_t heaps, visible, replaced, bytes, created;
    DescStats(&heaps, &visible, &replaced, &bytes, &created);
    Log("perf: %.1f fps, network %.2f ms (%u frames), mirror network %.2f ms (%u frames), private %.0f MB, "
        "descriptor heaps %zu (%zu visible, %zu replaced, %zu created, %.0f MB tracked)",
        frames / seconds, nR ? msR / nR : 0.0, nR, nM ? msM / nM : 0.0, nM, pm.PrivateUsage / 1048576.0, heaps, visible,
        replaced, created, bytes / 1048576.0);
    if (g_config.mirror) {
        char a[320], b[160];
        MirrorStats(a, sizeof(a));
        MirrorPassStats(b, sizeof(b));
        Log("mirror: %s; %s", a, b);
    }
    frames = 0;
    start = now;
}

bool NeuralTestResolve(void* commandList, void* resolved, unsigned d3d12State, void* raw, void* coverage, void* kept,
                       unsigned renderW, unsigned renderH) {
    return g_resolve.Run(static_cast<ID3D12GraphicsCommandList*>(commandList), static_cast<ID3D12Resource*>(resolved),
                         (D3D12_RESOURCE_STATES)d3d12State, static_cast<ID3D12Resource*>(coverage),
                         static_cast<ID3D12Resource*>(kept), static_cast<ID3D12Resource*>(raw), renderW, renderH);
}

}  // namespace hf
