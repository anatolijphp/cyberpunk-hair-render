// Training capture: see capture.h.
//
// Each recorded frame goes into its own readback buffer (CopyTextureRegion on the game's list). A
// writer thread maps a buffer once the GPU is surely done with it (300 ms later), crops it to the
// hair box of the first frame and writes raw files, so the render thread never waits on the disk.
#include "capture.h"

#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "hair.h"

namespace hf {
namespace {

struct Phase {
    bool fixOn;
    int warmup;   // frames not recorded while FSR's history settles
    int record;
};
constexpr Phase kPhases[] = {{true, 16, 64}, {false, 16, 64}};
constexpr int kNumPhases = 2;
constexpr int kOutputFrames = 8;      // the last frames of each phase also keep the FSR output
constexpr int kMinSlots = 8, kMaxSlots = 48;
constexpr UINT64 kSlotMemory = 768ull << 20;   // readback memory for the render-resolution buffers
constexpr uint64_t kGpuDoneMs = 300;  // a frame's copies have certainly executed by then
constexpr uint64_t kStaleMs = 1000;   // a frame whose output never came is written without it
constexpr uint64_t kArmTimeoutMs = 3000, kStallMs = 5000;
constexpr int kMargin = 32;

enum Buf { kColor, kCoverage, kKept, kResolved, kOutput, kBufs };
const char* const kNames[kBufs] = {"color", "coverage", "kept", "resolved", "output"};

struct Slot {
    ID3D12Resource* buf = nullptr;      // render-resolution buffers
    ID3D12Resource* outBuf = nullptr;   // FSR output (allocated when first needed)
    int state = 0;   // 0 free, 1 recording (output may follow this frame), 2 waiting for the GPU, 3 writing
    uint64_t ms = 0;
    int index = 0;
    int phase = 0;
    bool targets = false;   // coverage / kept / resolved copied
    bool wantOutput = false, output = false;
};

struct FrameMeta {
    int index;
    uint64_t ms;
    int phase;
    bool targets;
    bool hasJitter;
    float jitter[2];
    bool output;
};

std::mutex g_m;
std::atomic<bool> g_armed{false}, g_running{false};
uint64_t g_armedMs = 0;
bool g_keyWasDown = false;

// Bridge to the CET script training/cet/hairfix_capture (it switches the FSR quality between captures).
// CET mods may only write inside their own folder, so both sides talk through files there:
//   pair.txt     mod -> script  "<n>": the capture key was pressed, record a pair
//   request.txt  script -> mod  "<id> <tag>": record one capture, tag goes into the folder name
//   status.txt   mod -> script  "<id> done|failed <folder>"
// Counters instead of deleting files: the script cannot delete them.
std::wstring BridgeDir() { return ModuleDir() + L"plugins\\cyber_engine_tweaks\\mods\\hairfix_capture\\"; }
long long g_lastRequest = -1, g_requestId = -1;   // request being served (-1: started by the key)
bool g_requestSeen = false;                      // request.txt read once since the game started
std::atomic<uint32_t> g_renderSize{0};           // width << 16 | height of the last upscaler frame
uint32_t g_reportedSize = ~0u;
uint64_t g_lastPollMs = 0;
std::string g_tag;

void WriteText(const std::wstring& path, const std::string& text) {
    FILE* f = _wfopen(path.c_str(), L"w");
    if (!f) return;
    fputs(text.c_str(), f);
    fclose(f);
}

void ReportStatus(long long id, bool ok, const std::wstring& dir) {
    if (id < 0) return;
    char buf[64];
    snprintf(buf, sizeof(buf), "%lld %s ", id, ok ? "done" : "failed");
    std::string s = buf;
    s.append(dir.begin(), dir.end());
    WriteText(BridgeDir() + L"status.txt", s + "\n");
}

// current capture (g_m)
std::wstring g_dir;
const char* g_path = "";
UINT g_w = 0, g_h = 0, g_outW = 0, g_outH = 0;
D3D12_PLACED_SUBRESOURCE_FOOTPRINT g_fp[kBufs]{};
UINT g_bpp[kBufs]{};
UINT64 g_bufSize = 0, g_outSize = 0;
int g_phase = 0, g_phaseFrame = 0, g_recorded = 0, g_written = 0, g_waited = 0;
bool g_done = false, g_wasActive = true;
uint64_t g_startMs = 0, g_lastFrameMs = 0;
int g_box[4]{};   // x, y, w, h at render resolution
bool g_haveBox = false;
Slot g_slots[kMaxSlots];
Slot* g_current = nullptr;
std::vector<FrameMeta> g_frames;

UINT BytesPerPixel(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_A8_UNORM: return 1;
        case DXGI_FORMAT_R16_UNORM: return 2;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_TYPELESS: return 8;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_TYPELESS: return 16;
        case DXGI_FORMAT_R11G11B10_FLOAT: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM:
            return 4;
        default: return 0;
    }
}

D3D12_RESOURCE_DESC Texture(UINT w, UINT h, DXGI_FORMAT f) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = f;
    d.SampleDesc.Count = 1;
    return d;
}

ID3D12Resource* ReadbackBuffer(ID3D12Device* dev, UINT64 size) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                            IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}

void Copy(ID3D12GraphicsCommandList* cl, ID3D12Resource* dstBuf, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,
          ID3D12Resource* src, D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = src;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = state;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    bool transition = state != D3D12_RESOURCE_STATE_COPY_SOURCE;
    if (transition) cl->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION dst{}, from{};
    dst.pResource = dstBuf;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    from.pResource = src;
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    from.SubresourceIndex = 0;
    cl->CopyTextureRegion(&dst, 0, 0, 0, &from, nullptr);
    if (transition) {
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        cl->ResourceBarrier(1, &b);
    }
}

void ReleaseSlots() {
    for (Slot& s : g_slots) {
        if (s.buf) s.buf->Release();
        if (s.outBuf) s.outBuf->Release();
        s = Slot{};
    }
    g_current = nullptr;
}

// back to the user's on/off state (g_m held)
void Finish() {
    g_done = true;
    g_current = nullptr;
    if (IsActive() != g_wasActive) SetActive(g_wasActive);
}

// hair box from the coverage / kept planes of a mapped frame, with a margin (g_m held)
void FindBox(const uint8_t* data) {
    const auto& cov = g_fp[kCoverage];
    const auto& kept = g_fp[kKept];
    int x0 = (int)g_w, y0 = (int)g_h, x1 = -1, y1 = -1;
    for (UINT y = 0; y < g_h; ++y) {
        const uint16_t* c = reinterpret_cast<const uint16_t*>(data + cov.Offset + (size_t)y * cov.Footprint.RowPitch);
        const uint8_t* k = data + kept.Offset + (size_t)y * kept.Footprint.RowPitch;
        for (UINT x = 0; x < g_w; ++x) {
            if (c[x] < 64250 || k[x] > 0) {
                x0 = std::min(x0, (int)x), x1 = std::max(x1, (int)x);
                y0 = std::min(y0, (int)y), y1 = std::max(y1, (int)y);
            }
        }
    }
    if (x1 < 0) {   // no hair in view: keep the whole frame
        g_box[0] = 0, g_box[1] = 0, g_box[2] = (int)g_w, g_box[3] = (int)g_h;
    } else {
        x0 = std::max(0, x0 - kMargin), y0 = std::max(0, y0 - kMargin);
        x1 = std::min((int)g_w - 1, x1 + kMargin), y1 = std::min((int)g_h - 1, y1 + kMargin);
        g_box[0] = x0, g_box[1] = y0, g_box[2] = x1 - x0 + 1, g_box[3] = y1 - y0 + 1;
    }
    g_haveBox = true;
    Log("capture: hair box %d,%d %dx%d", g_box[0], g_box[1], g_box[2], g_box[3]);
}

// the render box scaled to the output resolution (rounded outwards)
void OutputBox(const int box[4], int out[4]) {
    out[0] = (int)((uint64_t)box[0] * g_outW / g_w);
    out[1] = (int)((uint64_t)box[1] * g_outH / g_h);
    int x1 = (int)(((uint64_t)(box[0] + box[2]) * g_outW + g_w - 1) / g_w);
    int y1 = (int)(((uint64_t)(box[1] + box[3]) * g_outH + g_h - 1) / g_h);
    out[2] = std::min(x1, (int)g_outW) - out[0];
    out[3] = std::min(y1, (int)g_outH) - out[1];
}

void WritePlane(int index, int b, const uint8_t* base, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, UINT bpp,
                const int box[4]) {
    wchar_t name[64];
    swprintf(name, 64, L"%04d_%hs.raw", index, kNames[b]);
    FILE* f = _wfopen((g_dir + name).c_str(), L"wb");
    if (!f) return;
    for (int y = box[1]; y < box[1] + box[3]; ++y)
        fwrite(base + fp.Offset + (size_t)y * fp.Footprint.RowPitch + (size_t)box[0] * bpp, 1, (size_t)box[2] * bpp, f);
    fclose(f);
}

void WriteFrame(Slot& s) {
    void* p = nullptr;
    D3D12_RANGE all{0, (SIZE_T)g_bufSize};
    if (FAILED(s.buf->Map(0, &all, &p)) || !p) {
        Log("capture: map failed for frame %d", s.index);
        return;
    }
    auto* data = static_cast<const uint8_t*>(p);
    int box[4];
    {
        std::lock_guard<std::mutex> lock(g_m);
        if (!g_haveBox && s.targets) FindBox(data);
        if (!g_haveBox) g_box[0] = 0, g_box[1] = 0, g_box[2] = (int)g_w, g_box[3] = (int)g_h, g_haveBox = true;
        memcpy(box, g_box, sizeof(box));
    }
    WritePlane(s.index, kColor, data, g_fp[kColor], g_bpp[kColor], box);
    if (s.targets)
        for (int b : {kCoverage, kKept, kResolved}) WritePlane(s.index, b, data, g_fp[b], g_bpp[b], box);
    D3D12_RANGE none{0, 0};
    s.buf->Unmap(0, &none);
    if (s.output && s.outBuf) {
        D3D12_RANGE outAll{0, (SIZE_T)g_outSize};
        if (SUCCEEDED(s.outBuf->Map(0, &outAll, &p)) && p) {
            int obox[4];
            OutputBox(box, obox);
            WritePlane(s.index, kOutput, static_cast<const uint8_t*>(p), g_fp[kOutput], g_bpp[kOutput], obox);
            s.outBuf->Unmap(0, &none);
        }
    }
}

void WriteMeta() {
    FILE* f = _wfopen((g_dir + L"meta.txt").c_str(), L"w");
    if (!f) return;
    int obox[4];
    OutputBox(g_box, obox);
    fprintf(f, "path %s\ntag %s\nrender %u %u\noutput %u %u\nbox %d %d %d %d\noutput_box %d %d %d %d\n", g_path,
            g_tag.empty() ? "-" : g_tag.c_str(), g_w, g_h, g_outW, g_outH, g_box[0], g_box[1], g_box[2], g_box[3], obox[0], obox[1], obox[2], obox[3]);
    for (int b = 0; b < kBufs; ++b)
        fprintf(f, "buffer %s format %d bpp %u\n", kNames[b], (int)g_fp[b].Footprint.Format, g_bpp[b]);
    for (const FrameMeta& m : g_frames) {
        fprintf(f, "frame %d ms %llu fix %s targets %d output %d", m.index, (unsigned long long)m.ms,
                kPhases[m.phase].fixOn ? "on" : "off", m.targets, m.output);
        if (m.hasJitter) fprintf(f, " jitter %.6f %.6f", m.jitter[0], m.jitter[1]);
        fputc('\n', f);
    }
    fclose(f);
}

void WriterThread() {
    for (;;) {
        Sleep(50);
        uint64_t now = GetTickCount64();
        Slot* ready = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_m);
            if (!g_done && now - g_lastFrameMs > kStallMs) {
                Log("capture: no frames for %llu ms, stopping early", (unsigned long long)kStallMs);
                Finish();
            }
            for (Slot& s : g_slots) {
                if (s.state == 1 && (g_done || now - s.ms >= kStaleMs)) {   // output never came
                    s.state = 2;
                    if (g_current == &s) g_current = nullptr;
                }
                if (s.state == 2 && now - s.ms >= kGpuDoneMs && (!ready || s.index < ready->index)) ready = &s;
            }
            if (ready) ready->state = 3;
        }
        if (ready) {
            WriteFrame(*ready);
            std::lock_guard<std::mutex> lock(g_m);
            for (FrameMeta& m : g_frames)
                if (m.index == ready->index) m.output = ready->output;
            ready->state = 0;
            ++g_written;
            continue;
        }
        std::lock_guard<std::mutex> lock(g_m);
        bool busy = false;
        for (const Slot& s : g_slots) busy |= s.state != 0;
        if (!g_done || busy) continue;
        WriteMeta();
        ReleaseSlots();
        Log("capture: done, %d frames (waited %d frames for buffers) in %.1f s -> %ls", g_written, g_waited,
            (GetTickCount64() - g_startMs) / 1000.0, g_dir.c_str());
        ReportStatus(g_requestId, g_written > 0, g_dir);
        g_running = false;
        return;
    }
}

// first frame after the key press: buffers, folder, writer thread (g_m held)
bool Start(ID3D12GraphicsCommandList* cl, ID3D12Resource* color, const char* path) {
    ID3D12Device* dev = nullptr;
    if (FAILED(cl->GetDevice(IID_PPV_ARGS(&dev))) || !dev) return false;
    D3D12_RESOURCE_DESC cd = color->GetDesc();
    g_w = (UINT)cd.Width;
    g_h = cd.Height;
    const D3D12_RESOURCE_DESC descs[kResolved + 1] = {cd, Texture(g_w, g_h, kCoverageFormat),
                                                      Texture(g_w, g_h, DXGI_FORMAT_R8_UNORM),
                                                      Texture(g_w, g_h, cd.Format)};
    UINT64 offset = 0;
    bool ok = true;
    for (int b = kColor; b <= kResolved; ++b) {
        UINT64 size = 0;
        dev->GetCopyableFootprints(&descs[b], 0, 1, offset, &g_fp[b], nullptr, nullptr, &size);
        g_bpp[b] = BytesPerPixel(descs[b].Format);
        ok &= g_bpp[b] != 0;
        offset = (g_fp[b].Offset + size + 511) & ~511ull;   // placement alignment
    }
    g_bufSize = offset;
    int slots = (int)std::min<UINT64>(kMaxSlots, std::max<UINT64>(kMinSlots, kSlotMemory / g_bufSize));
    for (int i = 0; i < slots && ok; ++i)
        if (!(g_slots[i].buf = ReadbackBuffer(dev, g_bufSize))) ok = false;
    dev->Release();
    if (!ok) {
        Log("capture: colour format %d / readback buffers not usable, not capturing", (int)cd.Format);
        ReleaseSlots();
        return false;
    }
    g_outW = g_outH = 0;
    g_outSize = 0;
    SYSTEMTIME t;
    GetLocalTime(&t);
    wchar_t name[128];
    swprintf(name, 128, L"%04d%02d%02d_%02d%02d%02d%hs%hs\\", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute,
             t.wSecond, g_tag.empty() ? "" : "_", g_tag.c_str());
    std::wstring root = ModuleDir() + L"hairfix_capture\\";
    CreateDirectoryW(root.c_str(), nullptr);
    g_dir = root + name;
    CreateDirectoryW(g_dir.c_str(), nullptr);
    g_path = path;
    g_phase = g_phaseFrame = g_recorded = g_written = g_waited = 0;
    g_done = false;
    g_haveBox = false;
    g_current = nullptr;
    g_frames.clear();
    g_startMs = g_lastFrameMs = GetTickCount64();
    g_wasActive = IsActive();
    if (!g_wasActive) SetActive(true);
    Log("capture: started (%s, render %ux%u) -> %ls", path, g_w, g_h, g_dir.c_str());
    std::thread(WriterThread).detach();
    return true;
}

}  // namespace

void CapturePollKey() {
    if (!g_config.captureKey) return;
    bool down = (GetAsyncKeyState(g_config.captureKey) & 0x8000) != 0;
    uint64_t now = GetTickCount64();
    bool idle = !g_armed && !g_running;
    if (down && !g_keyWasDown && idle) {
        if (GetFileAttributesW(BridgeDir().c_str()) != INVALID_FILE_ATTRIBUTES) {
            // the CET script records the pair (it switches the FSR quality in between)
            WriteText(BridgeDir() + L"pair.txt", std::to_string(now) + "\n");
            Log("capture: key pressed, pair requested from the CET script");
        } else {
            std::lock_guard<std::mutex> lock(g_m);
            g_tag.clear();
            g_requestId = -1;
            g_armedMs = now;
            g_armed = true;
            Log("capture: key pressed, waiting for the next upscaler frame");
        }
    }
    g_keyWasDown = down;

    // capture requests from the CET script
    if (now - g_lastPollMs >= 250) {
        g_lastPollMs = now;
        // the render resolution, so the script can tell when a quality switch took effect
        uint32_t size = g_renderSize.load(std::memory_order_relaxed);
        if (size != g_reportedSize && GetFileAttributesW(BridgeDir().c_str()) != INVALID_FILE_ATTRIBUTES) {
            g_reportedSize = size;
            WriteText(BridgeDir() + L"render.txt", std::to_string(size >> 16) + " " + std::to_string(size & 0xffff) + "\n");
        }
        long long id = -1;
        char tag[64] = "";
        int n = 0;
        if (FILE* f = _wfopen((BridgeDir() + L"request.txt").c_str(), L"r")) {
            n = fscanf(f, "%lld %63s", &id, tag);
            fclose(f);
        }
        if (!g_requestSeen) {   // first look: a request left over from an earlier session is not served
            g_requestSeen = true;
            if (n >= 1) g_lastRequest = id;
        } else if (idle) {
            if (n >= 1 && id != g_lastRequest) {
                g_lastRequest = id;
                std::lock_guard<std::mutex> lock(g_m);
                g_tag = n >= 2 ? tag : "";
                g_requestId = id;
                g_armedMs = now;
                g_armed = true;
                Log("capture: request %lld (%s) from the CET script", id, g_tag.c_str());
            }
        }
    }

    if (g_armed && now - g_armedMs > kArmTimeoutMs) {
        g_armed = false;
        Log("capture: no upscaler frame within %llu ms - the game is not rendering the scene now",
            (unsigned long long)kArmTimeoutMs);
        std::lock_guard<std::mutex> lock(g_m);
        ReportStatus(g_requestId, false, L"");
    }
}

bool CaptureActive() { return g_armed.load(std::memory_order_relaxed) || g_running.load(std::memory_order_relaxed); }

void CaptureNoteRenderSize(unsigned width, unsigned height) {
    g_renderSize.store((width & 0xffff) << 16 | (height & 0xffff), std::memory_order_relaxed);
}

void CaptureFrame(void* commandList, void* color, unsigned colorState, const float* jitter, const char* path) {
    if (!CaptureActive() || !commandList || !color) return;
    auto* cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    std::lock_guard<std::mutex> lock(g_m);
    if (g_armed) {
        g_armed = false;
        if (g_running) return;
        if (!Start(cl, static_cast<ID3D12Resource*>(color), path)) {
            ReportStatus(g_requestId, false, L"");
            return;
        }
        g_running = true;
    }
    if (g_done) return;
    uint64_t now = GetTickCount64();
    g_lastFrameMs = now;
    if (g_current) {   // last frame had no output step
        g_current->state = 2;
        g_current = nullptr;
    }
    const Phase& ph = kPhases[g_phase];
    int r = g_phaseFrame - ph.warmup;
    if (r >= ph.record) {
        if (++g_phase == kNumPhases) {
            Finish();
            return;
        }
        g_phaseFrame = 0;
        SetActive(kPhases[g_phase].fixOn);
        return;
    }
    void *cov = nullptr, *kept = nullptr, *resolved = nullptr;
    bool targets = PendingTargets(&cov, &kept, &resolved);
    if (ph.fixOn && !targets) return;   // the resolve did not run this frame: not counted
    if (r < 0) {   // warm-up
        ++g_phaseFrame;
        return;
    }
    Slot* slot = nullptr;
    for (Slot& s : g_slots)
        if (s.buf && s.state == 0) {
            slot = &s;
            break;
        }
    if (!slot) {   // writer behind: wait for a free buffer, the scene is still
        ++g_waited;
        return;
    }
    ++g_phaseFrame;
    const auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;   // the resolve leaves its targets so
    Copy(cl, slot->buf, g_fp[kColor], static_cast<ID3D12Resource*>(color), (D3D12_RESOURCE_STATES)colorState);
    slot->targets = ph.fixOn;
    if (ph.fixOn) {
        Copy(cl, slot->buf, g_fp[kCoverage], static_cast<ID3D12Resource*>(cov), kRead);
        Copy(cl, slot->buf, g_fp[kKept], static_cast<ID3D12Resource*>(kept), kRead);
        Copy(cl, slot->buf, g_fp[kResolved], static_cast<ID3D12Resource*>(resolved), kRead);
    }
    slot->index = g_recorded++;
    slot->phase = g_phase;
    slot->ms = now;
    slot->wantOutput = r >= ph.record - kOutputFrames;
    slot->output = false;
    slot->state = 1;
    g_current = slot;
    FrameMeta m{slot->index, now - g_startMs, g_phase, ph.fixOn, jitter != nullptr, {0, 0}, false};
    if (jitter) m.jitter[0] = jitter[0], m.jitter[1] = jitter[1];
    g_frames.push_back(m);
}

void CaptureOutput(void* commandList, void* output, unsigned outputState) {
    if (!g_running.load(std::memory_order_relaxed) || !commandList || !output) return;
    std::lock_guard<std::mutex> lock(g_m);
    Slot* s = g_current;
    g_current = nullptr;
    if (!s || s->state != 1) return;
    s->ms = GetTickCount64();
    if (s->wantOutput) {
        auto* out = static_cast<ID3D12Resource*>(output);
        D3D12_RESOURCE_DESC od = out->GetDesc();
        ID3D12Device* dev = nullptr;
        if (!g_outSize && SUCCEEDED(out->GetDevice(IID_PPV_ARGS(&dev))) && dev) {
            g_outW = (UINT)od.Width;
            g_outH = od.Height;
            dev->GetCopyableFootprints(&od, 0, 1, 0, &g_fp[kOutput], nullptr, nullptr, &g_outSize);
            g_bpp[kOutput] = BytesPerPixel(od.Format);
            if (!g_bpp[kOutput]) Log("capture: output format %d not supported", (int)od.Format);
        }
        if (dev) dev->Release();
        if (g_bpp[kOutput] && (UINT)od.Width == g_outW && od.Height == g_outH) {
            if (!s->outBuf && SUCCEEDED(out->GetDevice(IID_PPV_ARGS(&dev))) && dev) {
                s->outBuf = ReadbackBuffer(dev, g_outSize);
                dev->Release();
            }
            if (s->outBuf) {
                Copy(static_cast<ID3D12GraphicsCommandList*>(commandList), s->outBuf, g_fp[kOutput], out,
                     (D3D12_RESOURCE_STATES)outputState);
                s->output = true;
            }
        }
    }
    s->state = 2;
}

}  // namespace hf
