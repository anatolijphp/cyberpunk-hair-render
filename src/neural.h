// Hair resolve network: a small network that refines the mod's resolve from the game's raw frame
// before the upscaler ([Hair] Resolve = neural, weights in hairfix_resolve.bin, see training/).
#pragma once

namespace hf {

// Before the upscaler, right after the mod's resolve ([Hair] Resolve = neural, hairfix_resolve.bin):
// the resolve network rewrites `resolved` (render resolution, in resolvedState) from it, the raw game
// frame and the hair targets. False when it did not run (the plain resolve stays).
bool NeuralResolve(void* commandList, void* resolved, unsigned resolvedState, void* raw, void* coverage, void* kept,
                   unsigned renderW, unsigned renderH);

// The same on a mirror's resolve (hair.h): `resolved` holds the image in its top-left width x height.
bool NeuralResolveMirror(void* commandList, void* resolved, unsigned resolvedState, void* raw, void* coverage,
                         void* kept, unsigned width, unsigned height);

// training/engine_test.cpp: the network on given textures (resolved in d3d12State, raw / coverage /
// kept in NON_PIXEL_SHADER_RESOURCE), without the game.
bool NeuralTestResolve(void* commandList, void* resolved, unsigned d3d12State, void* raw, void* coverage, void* kept,
                       unsigned renderW, unsigned renderH);

}  // namespace hf
