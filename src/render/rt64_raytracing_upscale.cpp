//
// RT64
//
// Upscaling of the ray traced image. The path tracer renders at a lower resolution, the composed image is upscaled to
// the display resolution (FSR 1 spatially, or FSR 3 temporally with motion vectors, depth and a reactive mask), and the
// post process pass blends it over the rasterized frame by primary hit coverage, so the sky and background keep the
// display resolution. Everything drawn after the path traced pass (effects, the HUD) is rasterized at the display
// resolution as before. Also the GPU timing of the ray tracing passes.

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "rt64_descriptor_sets.h"
#include "rt64_shader_library.h"
#include "rt64_upscaler_fsr1.h"
#include "rt64_upscaler_fsr3.h"

#include "shaders/UpscalerInputsCS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/UpscalerInputsCS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/UpscalerInputsCS.hlsl.metal.h"
#endif

namespace RT64 {
    static std::atomic<uint32_t> upscalerModeSetting{ uint32_t(RaytracingUpscaler::Off) };
    static std::atomic<uint32_t> upscalerQualitySetting{ 1u };
    static std::atomic<float> upscalerSharpnessSetting{ 0.5f };

    void setRaytracingUpscaler(uint32_t mode, uint32_t quality, float sharpness) {
        upscalerModeSetting.store(std::min(mode, uint32_t(RaytracingUpscaler::FSR3)));
        upscalerQualitySetting.store(std::min(quality, 4u));
        upscalerSharpnessSetting.store(std::clamp(sharpness, 0.0f, 1.0f));
    }

    static Upscaler::QualityMode qualityFromSetting(uint32_t quality) {
        switch (quality) {
        case 0:
            return Upscaler::QualityMode::Native;
        case 2:
            return Upscaler::QualityMode::Balanced;
        case 3:
            return Upscaler::QualityMode::Performance;
        case 4:
            return Upscaler::QualityMode::UltraPerformance;
        case 1:
        default:
            return Upscaler::QualityMode::Quality;
        }
    }

    static const char *upscalerKindName(RaytracingUpscaler kind) {
        switch (kind) {
        case RaytracingUpscaler::FSR1:
            return "FSR 1";
        case RaytracingUpscaler::FSR3:
            return "FSR 3";
        case RaytracingUpscaler::Off:
        default:
            return "Off";
        }
    }

    static const char *passNames[RaytracingResources::PassCount] = {
        "primary", "emissive", "direct", "indirect", "refraction/water", "reflection", "temporal", "denoise", "compose", "upscale", "post"
    };

    void RaytracingResources::resolveUpscaler(RenderWorker *worker, uint32_t displayWidth, uint32_t displayHeight, uint32_t &renderWidth, uint32_t &renderHeight) {
        requestedUpscaler = RaytracingUpscaler(upscalerModeSetting.load());
        upscalerQuality = qualityFromSetting(upscalerQualitySetting.load());
        upscalerSharpness = upscalerSharpnessSetting.load();
        renderWidth = displayWidth;
        renderHeight = displayHeight;

        Upscaler *upscaler = nullptr;
        RaytracingUpscaler kind = RaytracingUpscaler::Off;
        int renderW = int(displayWidth);
        int renderH = int(displayHeight);
        if (requestedUpscaler == RaytracingUpscaler::FSR3) {
            if ((fsr3Upscaler == nullptr) && !fsr3Unavailable) {
                fsr3Upscaler = createFSR3Upscaler(device, shaderFormat, fsr3FailureReason);
                fsr3Unavailable = (fsr3Upscaler == nullptr);
            }

            if (fsr3Upscaler != nullptr) {
                fsr3Upscaler->getQualityInformation(upscalerQuality, int(displayWidth), int(displayHeight), renderW, renderH);
                fsr3Upscaler->set(worker, upscalerQuality, renderW, renderH, int(displayWidth), int(displayHeight));
                if (fsr3Upscaler->isInitialized()) {
                    upscaler = fsr3Upscaler.get();
                    kind = RaytracingUpscaler::FSR3;
                }
                else if (fsr3FailureReason.empty()) {
                    fsr3FailureReason = "the upscaler failed to initialize or dispatch (see the FSR 3 lines above)";
                }
            }

            if ((upscaler == nullptr) && !fsr3FallbackLogged) {
                rtDiagLog("Upscaler: FSR 3 is unavailable (%s); using FSR 1 instead", fsr3FailureReason.c_str());
                fsr3FallbackLogged = true;
            }
        }

        if ((upscaler == nullptr) && (requestedUpscaler != RaytracingUpscaler::Off) && (shaderLibrary != nullptr)) {
            if (fsr1Upscaler == nullptr) {
                fsr1Upscaler = std::make_unique<FSR1Upscaler>(device, shaderFormat, &shaderLibrary->samplerLibrary);
            }

            fsr1Upscaler->getQualityInformation(upscalerQuality, int(displayWidth), int(displayHeight), renderW, renderH);
            fsr1Upscaler->set(worker, upscalerQuality, renderW, renderH, int(displayWidth), int(displayHeight));
            if (fsr1Upscaler->isInitialized()) {
                upscaler = fsr1Upscaler.get();
                kind = RaytracingUpscaler::FSR1;
            }
        }

        if (upscaler != nullptr) {
            renderWidth = uint32_t(std::max(renderW, 1));
            renderHeight = uint32_t(std::max(renderH, 1));
        }

        activeUpscaler = upscaler;
        activeUpscalerKind = kind;
        upscaleActive = (upscaler != nullptr);
        if (kind != previousUpscalerKind) {
            rtDiagLog("Upscaler: %s (requested %s), %s, ray tracing %u x %u for display %u x %u", upscalerKindName(kind), upscalerKindName(requestedUpscaler),
                Upscaler::getQualityName(upscalerQuality), renderWidth, renderHeight, displayWidth, displayHeight);
            previousUpscalerKind = kind;
            upscalerResetPending = true;
        }
    }

    void RaytracingResources::updateUpscalerFrame(uint64_t frameCount) {
        // Jitter the camera rays for the temporal upscaler, so it gathers the detail of the display resolution over a
        // few frames. pixelJitter is the sample position relative to the pixel center.
        const bool temporal = upscaleActive && (activeUpscaler != nullptr) && activeUpscaler->isTemporal();
        if (temporal) {
            jitterPhaseCount = std::max(activeUpscaler->getJitterPhaseCount(int(textureWidth), int(displayWidth)), 1);
            jitterIndex = int(frameCount % uint64_t(jitterPhaseCount));
            float jitterX = 0.0f;
            float jitterY = 0.0f;
            activeUpscaler->getJitterOffset(jitterIndex, jitterPhaseCount, jitterX, jitterY);
            rtParams.pixelJitter = { jitterX, jitterY };
        }
        else {
            jitterPhaseCount = 1;
            jitterIndex = 0;
            rtParams.pixelJitter = { 0.0f, 0.0f };
        }

        rtParams.displayScaleX = float(displayWidth) / float(std::max(textureWidth, 1u));
        rtParams.displayScaleY = float(displayHeight) / float(std::max(textureHeight, 1u));
        rtParams.upscaleCoverage = upscaleActive ? 1u : 0u;

        // The upscaler's history is dropped when the image it accumulates isn't a continuation of the last one: the
        // first ray traced frame after rasterized ones, a new size or upscaler, lighting that changed at once, or a
        // camera cut.
        bool cameraCut = false;
        {
            const hlslpp::float4x4 viewI = rtParams.viewI;
            const hlslpp::float4 originH = hlslpp::mul(hlslpp::float4(0.0f, 0.0f, 1.0f, 0.0f), viewI);
            const hlslpp::float4 midH = hlslpp::mul(hlslpp::float4(0.0f, 0.0f, 0.0f, 1.0f), viewI);
            if ((std::fabs(float(originH.w)) > 1e-12f) && (std::fabs(float(midH.w)) > 1e-12f)) {
                const hlslpp::float3 origin = hlslpp::float3(originH.xyz) / float(originH.w);
                const hlslpp::float3 mid = hlslpp::float3(midH.xyz) / float(midH.w);
                const hlslpp::float3 delta = mid - origin;
                const float length = float(hlslpp::length(delta));
                if (length > 1e-6f) {
                    const hlslpp::float3 forward = delta / length;
                    const float o[3] = { float(origin.x), float(origin.y), float(origin.z) };
                    const float f[3] = { float(forward.x), float(forward.y), float(forward.z) };
                    if (lastCameraValid) {
                        const float facing = f[0] * lastCameraForward[0] + f[1] * lastCameraForward[1] + f[2] * lastCameraForward[2];
                        const float dx = o[0] - lastCameraOrigin[0], dy = o[1] - lastCameraOrigin[1], dz = o[2] - lastCameraOrigin[2];
                        const float moved = std::sqrt(dx * dx + dy * dy + dz * dz);
                        // About 25 degrees of turn, or a jump far beyond any camera move between two frames.
                        cameraCut = (facing < 0.9f) || (moved > 400.0f);
                    }

                    memcpy(lastCameraOrigin, o, sizeof(o));
                    memcpy(lastCameraForward, f, sizeof(f));
                    lastCameraValid = true;
                }
            }
        }

        const bool firstRaytracedFrame = !lastRaytracedFrameValid || (frameCount != (lastRaytracedFrame + 1));
        lastRaytracedFrame = frameCount;
        lastRaytracedFrameValid = true;
        upscalerResetThisFrame = upscalerResetPending || firstRaytracedFrame || (rtParams.historyReset != 0) || cameraCut;
        upscalerResetPending = false;
        if (upscalerResetThisFrame && temporal) {
            upscalerResets++;
        }
    }

    void RaytracingResources::submitUpscalerInputs(RenderWorker *worker) {
        if (!upscalerInputsPipelineCreated) {
            upscalerInputsPipelineCreated = true;
            const void *blob = nullptr;
            uint64_t blobSize = 0;
#       ifdef _WIN32
            if (shaderFormat == RenderShaderFormat::DXIL) {
                blob = UpscalerInputsCSBlobDXIL;
                blobSize = sizeof(UpscalerInputsCSBlobDXIL);
            }
#       elif defined(__APPLE__)
            if (shaderFormat == RenderShaderFormat::METAL) {
                blob = UpscalerInputsCSBlobMSL;
                blobSize = sizeof(UpscalerInputsCSBlobMSL);
            }
#       endif
            if (shaderFormat == RenderShaderFormat::SPIRV) {
                blob = UpscalerInputsCSBlobSPIRV;
                blobSize = sizeof(UpscalerInputsCSBlobSPIRV);
            }

            if ((blob != nullptr) && (shaderLibrary != nullptr)) {
                std::unique_ptr<RenderShader> shader = device->createShader(blob, blobSize, "CSMain", shaderFormat);
                UpscalerInputsDescriptorSet layoutSet(shaderLibrary->samplerLibrary);
                RenderPipelineLayoutBuilder layoutBuilder;
                layoutBuilder.begin();
                layoutBuilder.addDescriptorSet(layoutSet);
                layoutBuilder.end();
                upscalerInputsPipelineLayout = layoutBuilder.create(device);
                upscalerInputsPipeline = device->createComputePipeline(RenderComputePipelineDesc(upscalerInputsPipelineLayout.get(), shader.get(), 8, 8, 1));
            }
        }

        if ((upscalerInputsPipeline == nullptr) || (shaderLibrary == nullptr)) {
            return;
        }

        if (upscalerInputsSet == nullptr) {
            upscalerInputsSet = std::make_unique<UpscalerInputsDescriptorSet>(shaderLibrary->samplerLibrary, device);
        }

        const int current = swapBuffers ? 1 : 0;
        const int previous = swapBuffers ? 0 : 1;
        UpscalerInputsDescriptorSet &set = *upscalerInputsSet;
        set.setBuffer(set.RtParams, rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        set.setTexture(set.gDepth, depthTexture[current].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gShadingPosition, shadingPositionTexture.get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gDirectLightAccum, directLightTexture[current].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gPrevNormalRoughness, normalRoughnessTexture[previous].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gDiffuse, diffuseTexture.get(), RenderTextureLayout::SHADER_READ);
        set.setTexture(set.gRefraction, refractionTexture.get(), RenderTextureLayout::SHADER_READ);
        set.setTexture(set.gColor, outputTexture[current].get(), RenderTextureLayout::SHADER_READ);
        set.setTexture(set.gPrevColor, outputTexture[previous].get(), RenderTextureLayout::SHADER_READ);
        set.setTexture(set.gMotionVectors, motionVectorTexture.get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gDeviceDepth, deviceDepthTexture.get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gReactiveMask, reactiveMaskTexture.get(), RenderTextureLayout::GENERAL);

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, {
            RenderTextureBarrier(diffuseTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(refractionTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(outputTexture[current].get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(outputTexture[previous].get(), RenderTextureLayout::SHADER_READ)
        });

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
            RenderTextureBarrier(depthTexture[current].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(directLightTexture[current].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(normalRoughnessTexture[previous].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(motionVectorTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(deviceDepthTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(reactiveMaskTexture.get(), RenderTextureLayout::GENERAL)
        });

        worker->commandList->setPipeline(upscalerInputsPipeline.get());
        worker->commandList->setComputePipelineLayout(upscalerInputsPipelineLayout.get());
        worker->commandList->setComputeDescriptorSet(set.get(), 0);
        worker->commandList->dispatch((textureWidth + 7) / 8, (textureHeight + 7) / 8, 1);

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, {
            RenderTextureBarrier(motionVectorTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(deviceDepthTexture.get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(reactiveMaskTexture.get(), RenderTextureLayout::SHADER_READ)
        });
    }

    void RaytracingResources::submitUpscale(RenderWorker *worker, float deltaTimeMs) {
        if (!upscaleActive || (activeUpscaler == nullptr)) {
            return;
        }

        const int current = swapBuffers ? 1 : 0;
        RenderTexture *color = outputTexture[current].get();
        const bool temporal = activeUpscaler->isTemporal();
        if (temporal) {
            submitUpscalerInputs(worker);
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(color, RenderTextureLayout::SHADER_READ));
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderTextureBarrier(upscaledOutputTexture.get(), RenderTextureLayout::GENERAL));

        Upscaler::UpscaleParameters params;
        params.inRect = { 0, 0, int(textureWidth), int(textureHeight) };
        params.inDiffuseAlbedo = diffuseTexture.get();
        params.inSpecularAlbedo = shadingSpecularTexture.get();
        params.inNormalRoughness = normalRoughnessTexture[current].get();
        params.inColor = color;
        params.inDepth = temporal ? deviceDepthTexture.get() : nullptr;
        params.inMotionVectors = temporal ? motionVectorTexture.get() : nullptr;
        params.inReactiveMask = temporal ? reactiveMaskTexture.get() : nullptr;
        params.outColor = upscaledOutputTexture.get();
        params.outWidth = displayWidth;
        params.outHeight = displayHeight;
        params.jitterX = -float(rtParams.pixelJitter.x);
        params.jitterY = -float(rtParams.pixelJitter.y);
        params.deltaTime = deltaTimeMs;
        params.nearPlane = cameraNearPlane;
        params.farPlane = cameraFarPlane;
        params.fovY = rtParams.fovRadians;
        params.sharpness = upscalerSharpness;
        params.resetAccumulation = upscalerResetThisFrame;
        activeUpscaler->upscale(worker, params);

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(upscaledOutputTexture.get(), RenderTextureLayout::SHADER_READ));
    }

    void RaytracingResources::beginPassTiming(RenderWorker *worker) {
        const uint32_t slot = passQueryFrame % PassTimingDepth;
        passQueryFrame++;
        std::unique_ptr<RenderQueryPool> &pool = passQueryPools[slot];
        if (pool == nullptr) {
            pool = worker->device->createQueryPool(uint32_t(PassTimestamp::Count));
        }
        else if (passQueryWritten[slot]) {
            pool->queryResults();
            const uint64_t *timestamps = pool->getResults();
            if (timestamps != nullptr) {
                auto average = [](float &value, float sample) {
                    value = (value > 0.0f) ? (value * 0.95f + sample * 0.05f) : sample;
                };

                for (uint32_t i = 0; i < PassCount; i++) {
                    if (timestamps[i + 1] >= timestamps[i]) {
                        average(passMs[i], float(double(timestamps[i + 1] - timestamps[i]) / 1000000.0));
                    }
                }

                const uint32_t last = uint32_t(PassTimestamp::Post);
                if (timestamps[last] >= timestamps[0]) {
                    average(totalPassMs, float(double(timestamps[last] - timestamps[0]) / 1000000.0));
                }

                const uint32_t indirectEnd = uint32_t(PassTimestamp::Indirect);
                if (timestamps[indirectEnd] >= timestamps[indirectEnd - 1]) {
                    average(indirectPassMs[(passQueryBounces[slot] >= 2) ? 1 : 0], float(double(timestamps[indirectEnd] - timestamps[indirectEnd - 1]) / 1000000.0));
                }
            }

            passQueryWritten[slot] = false;
        }

        activePassQueryPool = pool.get();
        activePassQuerySlot = slot;
        passQueryBounces[slot] = rtParams.indirectBounces;
        if (activePassQueryPool != nullptr) {
            worker->commandList->resetQueryPool(activePassQueryPool, 0, uint32_t(PassTimestamp::Count));
            worker->commandList->writeTimestamp(activePassQueryPool, uint32_t(PassTimestamp::Start));
        }
    }

    void RaytracingResources::writePassTimestamp(RenderWorker *worker, PassTimestamp timestamp) {
        if (activePassQueryPool != nullptr) {
            worker->commandList->writeTimestamp(activePassQueryPool, uint32_t(timestamp));
        }
    }

    void RaytracingResources::endPassTiming() {
        if (activePassQueryPool != nullptr) {
            passQueryWritten[activePassQuerySlot] = true;
        }

        activePassQueryPool = nullptr;
    }

    void RaytracingResources::logUpscalerDiagnostics() {
        rtDiagLog("  upscaler: %s (requested %s), quality %s, sharpness %.0f%%; ray tracing %u x %u, display %u x %u (%.0f%% of the pixels)",
            upscalerKindName(activeUpscalerKind), upscalerKindName(requestedUpscaler), Upscaler::getQualityName(upscalerQuality), upscalerSharpness * 100.0f,
            textureWidth, textureHeight, displayWidth, displayHeight, 100.0f * float(textureWidth * textureHeight) / float(std::max(displayWidth * displayHeight, 1u)));
        if (activeUpscaler != nullptr) {
            rtDiagLog("  upscaler backend %s; jitter phase %d of %d, offset %.3f %.3f; history resets %u%s",
                activeUpscaler->getName(), jitterIndex, jitterPhaseCount, float(rtParams.pixelJitter.x), float(rtParams.pixelJitter.y), upscalerResets,
                upscalerResetThisFrame ? " (reset this frame)" : "");
        }

        if ((requestedUpscaler == RaytracingUpscaler::FSR3) && (activeUpscalerKind != RaytracingUpscaler::FSR3)) {
            rtDiagLog("  FSR 3 unavailable: %s", fsr3FailureReason.empty() ? "unknown reason" : fsr3FailureReason.c_str());
        }

        char line[512];
        int length = snprintf(line, sizeof(line), "  GPU ms per pass (running average):");
        for (uint32_t i = 0; (i < PassCount) && (length > 0) && (length < int(sizeof(line))); i++) {
            length += snprintf(line + length, sizeof(line) - size_t(length), " %s %.3f,", passNames[i], passMs[i]);
        }

        if ((length > 0) && (length < int(sizeof(line)))) {
            snprintf(line + length, sizeof(line) - size_t(length), " total %.3f", totalPassMs);
        }

        rtDiagLog("%s", line);
    }
};
