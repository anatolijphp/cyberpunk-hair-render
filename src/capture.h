// Training capture ([Debug] CaptureKey). With the scene still (pause menu, photo mode), a key press
// records two halves of a sequence into hairfix_capture\<date_time>\ next to the DLL, cropped to the
// hair:
// - fix on:  game colour before the mod and FSR, hair coverage, hair-surface mask, resolved colour;
//            FSR output (display resolution, before the unsharp mask) for the last frames
// - fix off (switched automatically, restored afterwards): game colour; FSR output for the last frames
// Each half starts with a few unrecorded frames so FSR's history settles. Averaging the frames of a
// half gives the reference: dither and FSR jitter average out.
#pragma once

namespace hf {

void CapturePollKey();   // once per present
void CaptureNoteRenderSize(unsigned width, unsigned height);   // render resolution of this upscaler frame
bool CaptureActive();    // armed or recording (keeps the banner up)

// Before the upscaler, after the mod's resolve (if it ran) and before the game's colour is
// changed. colorState: the colour's D3D12 state; jitter: FSR jitter in pixels, or null.
void CaptureFrame(void* commandList, void* color, unsigned colorState, const float* jitter, const char* path);
// After the upscaler, before the unsharp mask; outputState: D3D12 state of the output.
void CaptureOutput(void* commandList, void* output, unsigned outputState);

}  // namespace hf
