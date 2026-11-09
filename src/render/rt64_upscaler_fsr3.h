//
// RT64
//

#pragma once

#include <memory>
#include <string>

#include "rt64_upscaler.h"

namespace RT64 {
    // AMD FSR 3.1 temporal upscaling through the AMD FSR SDK API (amd_fidelityfx_loader_dx12.dll, loaded the first time
    // it's selected). D3D12 only for now. Returns null and fills the reason when it can't be used: built without it,
    // another graphics API, the DLLs are missing or the upscaler context can't be created.
    std::unique_ptr<Upscaler> createFSR3Upscaler(RenderDevice *device, RenderShaderFormat shaderFormat, std::string &failureReason);
};
