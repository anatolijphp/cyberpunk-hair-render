// Upscaler probe: see probe.h.
//
// A capture logs two frames of compute dispatches (with the resources their descriptor tables
// point at, see descriptors.h), texture barriers and queue submissions; one starts every 1200
// presents, at most 8 per run.
#include "probe.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "common.h"
#include "descriptors.h"

namespace hf {
namespace {

struct ListState {
    int type = -1;
    uint64_t cs = 0;   // hash of the bound compute shader, 0 = none / graphics
    uint64_t tables[8]{};
};

std::mutex g_m;
std::unordered_map<void*, uint64_t> g_csHash;   // compute PSO -> shader hash
std::unordered_map<void*, ListState> g_lists;

std::atomic<bool> g_capturing{false};
uint64_t g_presents = 0, g_captureEnd = 0;
int g_captures = 0;
constexpr int kMaxCaptures = 8;
constexpr uint64_t kCaptureEvery = 1200, kCaptureFrames = 2;

std::string Format(const DescInfo& s) {
    char buf[96];
    if (!s.dim)
        snprintf(buf, sizeof(buf), "null");
    else if (s.dim == D3D12_RESOURCE_DIMENSION_BUFFER)
        snprintf(buf, sizeof(buf), "%p:buf%u%s", s.res, s.w, s.uav ? ":U" : "");
    else
        snprintf(buf, sizeof(buf), "%p:%ux%u:f%u%s", s.res, s.w, s.h, s.fmt, s.uav ? ":U" : ":S");
    return buf;
}

std::string FormatResource(void* resource) {
    DescInfo s;
    if (resource) {
        D3D12_RESOURCE_DESC d = static_cast<ID3D12Resource*>(resource)->GetDesc();
        s.res = resource;
        s.dim = (uint8_t)d.Dimension;
        s.w = (uint32_t)d.Width;
        s.h = d.Height;
        s.fmt = (uint16_t)d.Format;
    }
    return Format(s);
}

}  // namespace

bool ProbeEnabled() { return g_config.probe; }

bool ProbeCapturing() { return g_capturing.load(std::memory_order_relaxed); }

void ProbeComputePso(void* pso, uint64_t csHash) {
    std::lock_guard<std::mutex> lock(g_m);
    g_csHash[pso] = csHash;
}

void ProbeListCreated(void* list, D3D12_COMMAND_LIST_TYPE type) {
    std::lock_guard<std::mutex> lock(g_m);
    g_lists[list] = ListState{};
    g_lists[list].type = (int)type;
}

void ProbeSetPso(void* list, void* pso) {
    if (!ProbeCapturing()) return;
    std::lock_guard<std::mutex> lock(g_m);
    auto it = g_csHash.find(pso);
    g_lists[list].cs = it == g_csHash.end() ? 0 : it->second;
}

void ProbeSetComputeTable(void* list, UINT index, D3D12_GPU_DESCRIPTOR_HANDLE table) {
    if (!ProbeCapturing() || index >= 8) return;
    std::lock_guard<std::mutex> lock(g_m);
    g_lists[list].tables[index] = table.ptr;
}

void ProbeDispatch(void* list, UINT x, UINT y, UINT z) {
    if (!ProbeCapturing()) return;
    ListState ls;
    {
        std::lock_guard<std::mutex> lock(g_m);
        ls = g_lists[list];
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "D list=%p t%d cs=%016llx groups=%u,%u,%u", list, ls.type, (unsigned long long)ls.cs,
             x, y, z);
    std::string line = buf;
    for (int t = 0; t < 8; ++t) {
        if (!ls.tables[t]) continue;
        snprintf(buf, sizeof(buf), " | T%d", t);
        line += buf;
        DescInfo d;
        if (!DescAt(ls.tables[t], 0, &d)) {
            line += " ?";
            continue;
        }
        for (UINT i = 0; i < 8 && DescAt(ls.tables[t], i, &d); ++i) line += " " + Format(d);
    }
    Log("%s", line.c_str());
}

void ProbeBarriers(void* list, UINT n, const D3D12_RESOURCE_BARRIER* b) {
    if (!ProbeCapturing() || !b) return;
    for (UINT i = 0; i < n; ++i) {
        if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) {
            const auto& t = b[i].Transition;
            if (!t.pResource || t.pResource->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) continue;
            Log("B list=%p %s sub=%u 0x%x->0x%x", list, FormatResource(t.pResource).c_str(), t.Subresource,
                t.StateBefore, t.StateAfter);
        } else if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && b[i].UAV.pResource) {
            Log("B list=%p %s uav", list, FormatResource(b[i].UAV.pResource).c_str());
        }
    }
}

void ProbeExecute(void* queue, UINT n, void* const* lists) {
    if (!ProbeCapturing()) return;
    std::string line;
    char buf[48];
    for (UINT i = 0; i < n; ++i) {
        snprintf(buf, sizeof(buf), " %p", lists[i]);
        line += buf;
    }
    Log("X queue=%p lists:%s", queue, line.c_str());
}

void ProbeFfxUpscale(void* list, void* color, void* depth, void* motion, void* output) {
    if (!ProbeCapturing()) return;
    Log("FSR3 upscale list=%p color=%s depth=%s motion=%s output=%s", list, FormatResource(color).c_str(),
        FormatResource(depth).c_str(), FormatResource(motion).c_str(), FormatResource(output).c_str());
}

void ProbePresent() {
    if (!g_config.probe) return;
    uint64_t f = ++g_presents;
    if (g_capturing.load()) {
        if (f >= g_captureEnd) {
            g_capturing = false;
            Log("=== probe capture %d end ===", g_captures);
        }
    } else if (f % kCaptureEvery == kCaptureEvery / 2 && g_captures < kMaxCaptures) {
        ++g_captures;
        g_captureEnd = f + kCaptureFrames;
        Log("=== probe capture %d begin (present %llu) ===", g_captures, (unsigned long long)f);
        g_capturing = true;
    }
}

}  // namespace hf
