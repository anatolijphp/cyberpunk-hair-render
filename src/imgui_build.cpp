// Builds Dear ImGui's DX12 backend (the core and the Win32 backend are compiled from third_party).
// The DX12 backend compiles its two shaders with D3DCompile at runtime; here D3DCompile is
// redirected to return our precompiled DXIL (shaders/imgui_vs.hlsl, imgui_ps.hlsl), so there's
// no dependency on d3dcompiler_47.dll.
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_5.h>

#include <cstring>

#include "../build/gen/imgui_ps_cso.h"
#include "../build/gen/imgui_vs_cso.h"

namespace {

// minimal ID3DBlob over static data
class StaticBlob final : public ID3DBlob {
public:
    StaticBlob(const void* data, SIZE_T size) : data_(data), size_(size) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D10Blob)) {
            *out = this;
            AddRef();
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG r = --refs_;
        if (!r) delete this;
        return r;
    }
    LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return const_cast<void*>(data_); }
    SIZE_T STDMETHODCALLTYPE GetBufferSize() override { return size_; }

private:
    const void* data_;
    SIZE_T size_;
    ULONG refs_ = 1;
};

HRESULT PrecompiledD3DCompile(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR,
                              LPCSTR target, UINT, UINT, ID3DBlob** code, ID3DBlob** errors) {
    if (errors) *errors = nullptr;
    if (!strncmp(target, "vs_", 3)) *code = new StaticBlob(kImgui_vsCso, sizeof(kImgui_vsCso));
    else if (!strncmp(target, "ps_", 3)) *code = new StaticBlob(kImgui_psCso, sizeof(kImgui_psCso));
    else return E_FAIL;
    return S_OK;
}

}  // namespace

// The backend calls ::CreateDXGIFactory1 once at init; resolve it from the already loaded dxgi.dll
// instead of importing dxgi.dll into this proxy (which loads before the game sets up its graphics).
HRESULT WINAPI LazyCreateDXGIFactory1(REFIID riid, void** factory) {
    HMODULE dxgi = GetModuleHandleW(L"dxgi.dll");
    if (!dxgi) dxgi = LoadLibraryW(L"dxgi.dll");
    using Fn = HRESULT(WINAPI*)(REFIID, void**);
    auto fn = dxgi ? (Fn)(void*)GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    return fn ? fn(riid, factory) : E_FAIL;
}

#define D3DCompile PrecompiledD3DCompile
#define CreateDXGIFactory1 LazyCreateDXGIFactory1
#include "../third_party/imgui/backends/imgui_impl_dx12.cpp"
#undef CreateDXGIFactory1
#undef D3DCompile
