// Runs the mod's hair network (src/neural.cpp, shaders/nn_*.hlsl) on one frame without the game, to
// compare with PyTorch (engine_test.py writes the input and checks the result).
//
// usage: engine_test <dir>   (dir: input.bin, hairfix_resolve.bin; writes output.bin)
// input.bin: u32 width, u32 height, u32 raw (must be 1), RGBA16F colour (linear, the mod's resolve),
//            RGBA16F raw frame, R8 coverage, R8 kept (all width x height)
// output.bin: RGBA16F colour after the network
#include <d3d12.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/common.h"
#include "../src/hair.h"
#include "../src/neural.h"

// what neural.cpp uses from the rest of the mod
namespace hf {
Config g_config;
std::wstring g_dir;
std::wstring ModuleDir() { return g_dir; }
void Log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}
bool PendingTargets(void**, void**, void**) { return false; }
void ApplyUnsharpMask(void*, void*, unsigned, unsigned, unsigned) {}
DXGI_FORMAT TypedFormatForView(DXGI_FORMAT f) { return f; }
unsigned D3D12StateFromFfx(unsigned s) { return s; }
void* OverlayQueue() { return nullptr; }   // no GPU timing here
void DescStats(size_t* a, size_t* b, size_t* c, size_t* d, size_t* e) { *a = *b = *c = *d = *e = 0; }
}  // namespace hf

namespace {

ID3D12Device* dev;

ID3D12Resource* Make(D3D12_RESOURCE_DESC d, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) {
        printf("resource creation failed\n");
        exit(1);
    }
    return r;
}

D3D12_RESOURCE_DESC Tex(UINT w, UINT h, DXGI_FORMAT f, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = f;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC Buf(UINT64 size) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

void Barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    cl->ResourceBarrier(1, &x);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string dir = argv[1];
    hf::g_dir = std::wstring(dir.begin(), dir.end()) + L"\\";
    FILE* f = fopen((dir + "\\input.bin").c_str(), "rb");
    if (!f) return 1;
    uint32_t w, h, hasRaw;
    fread(&w, 4, 1, f);
    fread(&h, 4, 1, f);
    fread(&hasRaw, 4, 1, f);
    std::vector<uint8_t> color(w * h * 8), rawColor(hasRaw ? w * h * 8 : 0), cov(w * h), kept(w * h);
    fread(color.data(), 1, color.size(), f);
    fread(rawColor.data(), 1, rawColor.size(), f);
    fread(cov.data(), 1, cov.size(), f);
    fread(kept.data(), 1, kept.size(), f);
    fclose(f);

    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)))) return 1;
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    ID3D12CommandQueue* q;
    ID3D12CommandAllocator* alloc;
    ID3D12GraphicsCommandList* cl;
    ID3D12Fence* fence;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cl));
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    ID3D12Resource* out = Make(Tex(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* covTex = Make(Tex(w, h, DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* keptTex = Make(Tex(w, h, DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE), D3D12_HEAP_TYPE_DEFAULT,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* rawTex = Make(Tex(w, h, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE),
                                  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!hasRaw) rawColor.assign(w * h * 8, 0);

    // upload the three textures
    ID3D12Resource* textures[4] = {out, covTex, keptTex, rawTex};
    const uint8_t* data[4] = {color.data(), cov.data(), kept.data(), rawColor.data()};
    const UINT bpp[4] = {8, 1, 1, 8};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp[4];
    UINT64 offset = 0;
    for (int t = 0; t < 4; ++t) {
        D3D12_RESOURCE_DESC d = textures[t]->GetDesc();
        UINT64 size;
        dev->GetCopyableFootprints(&d, 0, 1, offset, &fp[t], nullptr, nullptr, &size);
        offset = (fp[t].Offset + size + 511) & ~511ull;
    }
    ID3D12Resource* upload = Make(Buf(offset), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    uint8_t* p;
    upload->Map(0, nullptr, (void**)&p);
    for (int t = 0; t < 4; ++t)
        for (UINT y = 0; y < h; ++y)
            memcpy(p + fp[t].Offset + (size_t)y * fp[t].Footprint.RowPitch, data[t] + (size_t)y * w * bpp[t], w * bpp[t]);
    upload->Unmap(0, nullptr);
    for (int t = 0; t < 4; ++t) {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = textures[t];
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = upload;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp[t];
        cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    Barrier(cl, out, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Barrier(cl, covTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(cl, keptTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Barrier(cl, rawTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto run = [&] {
        return hasRaw && hf::NeuralTestResolve(cl, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, rawTex, covTex, keptTex,
                                               w, h);
    };

    if (!run()) {
        printf("network did not run\n");
        return 1;
    }

    // optional timing: argv[2] = number of extra runs (each its own submission, waited for)
    if (argc > 2) {
        cl->Close();
        ID3D12CommandList* first[] = {cl};
        q->ExecuteCommandLists(1, first);
        UINT64 v = 0;
        auto wait = [&] {
            q->Signal(fence, ++v);
            HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            fence->SetEventOnCompletion(v, e);
            WaitForSingleObject(e, INFINITE);
            CloseHandle(e);
        };
        wait();
        int runs = atoi(argv[2]);
        LARGE_INTEGER f, t0, t1;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t0);
        for (int r = 0; r < runs; ++r) {
            alloc->Reset();
            cl->Reset(alloc, nullptr);
            run();
            cl->Close();
            q->ExecuteCommandLists(1, first);
            wait();
        }
        QueryPerformanceCounter(&t1);
        printf("%.2f ms per run (%d runs, %ux%u)\n", (t1.QuadPart - t0.QuadPart) * 1000.0 / f.QuadPart / runs, runs, w, h);
        fence->Signal(0);
        alloc->Reset();
        cl->Reset(alloc, nullptr);
        fence = nullptr;
        dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    }

    // read the result back
    ID3D12Resource* readback = Make(Buf(fp[0].Footprint.RowPitch * (UINT64)h), D3D12_HEAP_TYPE_READBACK,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
    Barrier(cl, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp[0];
    dst.PlacedFootprint.Offset = 0;
    src.pResource = out;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    cl->Close();
    ID3D12CommandList* lists[] = {cl};
    q->ExecuteCommandLists(1, lists);
    q->Signal(fence, 1);
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    fence->SetEventOnCompletion(1, ev);
    WaitForSingleObject(ev, INFINITE);
    if (dev->GetDeviceRemovedReason() != S_OK) {
        printf("device removed 0x%lx\n", dev->GetDeviceRemovedReason());
        return 1;
    }
    readback->Map(0, nullptr, (void**)&p);
    f = fopen((dir + "\\output.bin").c_str(), "wb");
    for (UINT y = 0; y < h; ++y) fwrite(p + (size_t)y * fp[0].Footprint.RowPitch, 1, w * 8, f);
    fclose(f);
    printf("output.bin written (%ux%u)\n", w, h);
    return 0;
}
