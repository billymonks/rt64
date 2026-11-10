//
// RT64
//
// Temporal accumulation (RtTemporalCS) and the edge-aware spatial filter (RtDenoiseCS) of the ray traced lighting, and
// the Graphics menu settings behind them: "Light Samples", "Denoiser" and "Light Accumulation".

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <atomic>
#include <iterator>

#include "rt64_descriptor_sets.h"
#include "rt64_shader_library.h"

#include "shaders/RtTemporalCS.hlsl.spirv.h"
#include "shaders/RtDenoiseCS.hlsl.spirv.h"
#ifdef _WIN32
#   include "shaders/RtTemporalCS.hlsl.dxil.h"
#   include "shaders/RtDenoiseCS.hlsl.dxil.h"
#elif defined(__APPLE__)
#   include "shaders/RtTemporalCS.hlsl.metal.h"
#   include "shaders/RtDenoiseCS.hlsl.metal.h"
#endif

namespace RT64 {
    // Light Samples: sun shadow, light, bounce and occlusion rays per pixel and frame.
    static const uint32_t kLightSamples[] = { 1, 2, 4 };
    static const char *kLightSampleNames[] = { "1x", "2x", "4x" };
    // Denoiser: a-trous passes (steps 1, 2, 4, ...). Off keeps the old blur of the bounce light only.
    static const uint32_t kDenoisePasses[] = { 0, 3, 5 };
    static const char *kDenoiserNames[] = { "Off", "Low", "High" };
    // Light Accumulation: smallest weight of the new frame, longest history and the history clip in standard deviations.
    // Balanced keeps the blend weights of the earlier accumulation (at least 10% new frame, history up to 16 frames).
    static const float kAccumulationMinBlend[] = { 0.2f, 0.1f, 0.05f };
    static const float kAccumulationMaxHistory[] = { 8.0f, 16.0f, 32.0f };
    static const float kAccumulationClipGamma[] = { 1.0f, 1.5f, 2.5f };
    static const char *kAccumulationNames[] = { "Responsive", "Balanced", "Stable" };

    // Edge stopping of RtDenoiseCS: the lighting tolerance in standard deviations of its noise, the depth tolerance
    // against the depth slope, and the shading normal exponent.
    static const float kDenoiseLuminanceSigma = 4.0f;
    static const float kDenoiseDepthSigma = 2.0f;
    static const float kDenoiseNormalPower = 64.0f;

    static std::atomic<uint32_t> lightSamplesSetting{ 0 };
    static std::atomic<uint32_t> denoiserSetting{ 1 };
    static std::atomic<uint32_t> accumulationSetting{ 1 };

    void setRaytracingLightQuality(uint32_t samples, uint32_t denoiser, uint32_t accumulation) {
        lightSamplesSetting.store(std::min(samples, uint32_t(std::size(kLightSamples)) - 1));
        denoiserSetting.store(std::min(denoiser, uint32_t(std::size(kDenoisePasses)) - 1));
        accumulationSetting.store(std::min(accumulation, uint32_t(std::size(kAccumulationMinBlend)) - 1));
    }

    // Must match DenoiseCB in RtDenoiseCS.hlsl.
    struct DenoiseCB {
        uint32_t resolution[2];
        uint32_t stepSize;
        uint32_t padding0;
        float luminanceSigma;
        float depthSigma;
        float normalPower;
        float padding1;
    };

    static std::unique_ptr<RenderShader> createLightFilterShader(RenderDevice *device, RenderShaderFormat format, const void *dxil, uint64_t dxilSize, const void *spirv, uint64_t spirvSize, const void *msl, uint64_t mslSize) {
        const void *blob = nullptr;
        uint64_t blobSize = 0;
        if (format == RenderShaderFormat::DXIL) {
            blob = dxil;
            blobSize = dxilSize;
        }
        else if (format == RenderShaderFormat::SPIRV) {
            blob = spirv;
            blobSize = spirvSize;
        }
        else if (format == RenderShaderFormat::METAL) {
            blob = msl;
            blobSize = mslSize;
        }

        if ((blob == nullptr) || (blobSize == 0)) {
            return nullptr;
        }

        return device->createShader(blob, blobSize, "CSMain", format);
    }

#ifdef _WIN32
#   define RT64_LIGHT_FILTER_DXIL(name) name##BlobDXIL, sizeof(name##BlobDXIL)
#else
#   define RT64_LIGHT_FILTER_DXIL(name) nullptr, 0
#endif
#ifdef __APPLE__
#   define RT64_LIGHT_FILTER_MSL(name) name##BlobMSL, sizeof(name##BlobMSL)
#else
#   define RT64_LIGHT_FILTER_MSL(name) nullptr, 0
#endif

    void RaytracingResources::updateLightQuality() {
        lightSamplesLevel = lightSamplesSetting.load();
        denoiserLevel = denoiserSetting.load();
        accumulationLevel = accumulationSetting.load();
        rtParams.lightSamples = kLightSamples[lightSamplesLevel];
        rtParams.temporalMinBlend = kAccumulationMinBlend[accumulationLevel];
        rtParams.temporalMaxHistory = kAccumulationMaxHistory[accumulationLevel];
        rtParams.temporalClipGamma = kAccumulationClipGamma[accumulationLevel];
        denoisePassCount = kDenoisePasses[denoiserLevel];
    }

    void RaytracingResources::createLightFilterPipelines() {
        if (lightFilterPipelinesCreated) {
            return;
        }

        lightFilterPipelinesCreated = true;
        std::unique_ptr<RenderShader> temporalShader = createLightFilterShader(device, shaderFormat,
            RT64_LIGHT_FILTER_DXIL(RtTemporalCS), RtTemporalCSBlobSPIRV, sizeof(RtTemporalCSBlobSPIRV), RT64_LIGHT_FILTER_MSL(RtTemporalCS));
        if (temporalShader != nullptr) {
            RtTemporalDescriptorSet layoutSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addDescriptorSet(layoutSet);
            layoutBuilder.end();
            lightTemporalPipelineLayout = layoutBuilder.create(device);
            lightTemporalPipeline = device->createComputePipeline(RenderComputePipelineDesc(lightTemporalPipelineLayout.get(), temporalShader.get(), 8, 8, 1));
        }

        std::unique_ptr<RenderShader> denoiseShader = createLightFilterShader(device, shaderFormat,
            RT64_LIGHT_FILTER_DXIL(RtDenoiseCS), RtDenoiseCSBlobSPIRV, sizeof(RtDenoiseCSBlobSPIRV), RT64_LIGHT_FILTER_MSL(RtDenoiseCS));
        if (denoiseShader != nullptr) {
            RtDenoiseDescriptorSet layoutSet;
            RenderPipelineLayoutBuilder layoutBuilder;
            layoutBuilder.begin();
            layoutBuilder.addPushConstant(0, 0, sizeof(DenoiseCB), RenderShaderStageFlag::COMPUTE);
            layoutBuilder.addDescriptorSet(layoutSet);
            layoutBuilder.end();
            lightDenoisePipelineLayout = layoutBuilder.create(device);
            lightDenoisePipeline = device->createComputePipeline(RenderComputePipelineDesc(lightDenoisePipelineLayout.get(), denoiseShader.get(), 8, 8, 1));
        }

        rtDiagLog("Light filters: temporal accumulation %s, denoiser %s", (lightTemporalPipeline != nullptr) ? "ready" : "unavailable",
            (lightDenoisePipeline != nullptr) ? "ready" : "unavailable");
    }

    void RaytracingResources::submitLightTemporal(RenderWorker *worker) {
        createLightFilterPipelines();

        const int current = swapBuffers ? 1 : 0;
        const int previous = swapBuffers ? 0 : 1;
        RenderTexture *rawDirect = filteredDirectLightTexture[1].get();
        RenderTexture *rawIndirect = filteredIndirectLightTexture[1].get();
        RenderTexture *direct = directLightTexture[current].get();
        RenderTexture *indirect = indirectLightTexture[current].get();
        if ((lightTemporalPipeline == nullptr) || (rtParamsBuffer.get() == nullptr)) {
            // Without the pass, this frame's samples are shown as they are.
            worker->commandList->barriers(RenderBarrierStage::COPY, {
                RenderTextureBarrier(rawDirect, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(rawIndirect, RenderTextureLayout::COPY_SOURCE),
                RenderTextureBarrier(direct, RenderTextureLayout::COPY_DEST),
                RenderTextureBarrier(indirect, RenderTextureLayout::COPY_DEST)
            });

            worker->commandList->copyTexture(direct, rawDirect);
            worker->commandList->copyTexture(indirect, rawIndirect);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
                RenderTextureBarrier(rawDirect, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(rawIndirect, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(direct, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(indirect, RenderTextureLayout::GENERAL)
            });

            return;
        }

        if (lightTemporalSet == nullptr) {
            lightTemporalSet = std::make_unique<RtTemporalDescriptorSet>(device);
        }

        RtTemporalDescriptorSet &set = *lightTemporalSet;
        set.setBuffer(set.RtParams, rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        set.setTexture(set.gDepth, depthTexture[current].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gShadingPosition, shadingPositionTexture.get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gShadingNormal, shadingNormalTexture.get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gNormalRoughness, normalRoughnessTexture[current].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gPrevNormalRoughness, normalRoughnessTexture[previous].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gRawDirect, rawDirect, RenderTextureLayout::GENERAL);
        set.setTexture(set.gRawIndirect, rawIndirect, RenderTextureLayout::GENERAL);
        set.setTexture(set.gPrevDirect, directLightTexture[previous].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gPrevIndirect, indirectLightTexture[previous].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gPrevMoments, momentsTexture[previous].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gDirectOut, direct, RenderTextureLayout::GENERAL);
        set.setTexture(set.gIndirectOut, indirect, RenderTextureLayout::GENERAL);
        set.setTexture(set.gMomentsOut, momentsTexture[current].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gVarianceOut, lightVarianceTexture[0].get(), RenderTextureLayout::GENERAL);
        set.setTexture(set.gGuideOut, lightGuideTexture.get(), RenderTextureLayout::GENERAL);

        // The light passes wrote this frame's samples and the primary pass the surfaces; wait for them.
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
            RenderTextureBarrier(depthTexture[current].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(shadingPositionTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(shadingNormalTexture.get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(normalRoughnessTexture[current].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(normalRoughnessTexture[previous].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rawDirect, RenderTextureLayout::GENERAL),
            RenderTextureBarrier(rawIndirect, RenderTextureLayout::GENERAL),
            RenderTextureBarrier(directLightTexture[previous].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(indirectLightTexture[previous].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(momentsTexture[previous].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(direct, RenderTextureLayout::GENERAL),
            RenderTextureBarrier(indirect, RenderTextureLayout::GENERAL),
            RenderTextureBarrier(momentsTexture[current].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(lightVarianceTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(lightGuideTexture.get(), RenderTextureLayout::GENERAL)
        });

        worker->commandList->setPipeline(lightTemporalPipeline.get());
        worker->commandList->setComputePipelineLayout(lightTemporalPipelineLayout.get());
        worker->commandList->setComputeDescriptorSet(set.get(), 0);
        worker->commandList->dispatch((textureWidth + 7) / 8, (textureHeight + 7) / 8, 1);

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
            RenderTextureBarrier(direct, RenderTextureLayout::GENERAL),
            RenderTextureBarrier(indirect, RenderTextureLayout::GENERAL),
            RenderTextureBarrier(momentsTexture[current].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(lightVarianceTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(lightGuideTexture.get(), RenderTextureLayout::GENERAL)
        });
    }

    bool RaytracingResources::submitLightDenoise(RenderWorker *worker) {
        const uint32_t passes = std::min(denoisePassCount, MaxDenoisePasses);
        if ((passes == 0) || (lightDenoisePipeline == nullptr) || (lightTemporalPipeline == nullptr)) {
            return false;
        }

        const int current = swapBuffers ? 1 : 0;
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
            RenderTextureBarrier(filteredDirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(filteredDirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(filteredIndirectLightTexture[0].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL),
            RenderTextureBarrier(lightVarianceTexture[1].get(), RenderTextureLayout::GENERAL)
        });

        worker->commandList->setPipeline(lightDenoisePipeline.get());
        worker->commandList->setComputePipelineLayout(lightDenoisePipelineLayout.get());

        // Pass i reads what pass i - 1 wrote and the last one writes the textures ComposePS reads,
        // filteredDirectLightTexture[1] and filteredIndirectLightTexture[1].
        RenderTexture *directIn = directLightTexture[current].get();
        RenderTexture *indirectIn = indirectLightTexture[current].get();
        RenderTexture *varianceIn = lightVarianceTexture[0].get();
        for (uint32_t i = 0; i < passes; i++) {
            const int output = (((passes - 1 - i) % 2) == 0) ? 1 : 0;
            RenderTexture *directOut = filteredDirectLightTexture[output].get();
            RenderTexture *indirectOut = filteredIndirectLightTexture[output].get();
            RenderTexture *varianceOut = lightVarianceTexture[((i % 2) == 0) ? 1 : 0].get();
            if (lightDenoiseSets[i] == nullptr) {
                lightDenoiseSets[i] = std::make_unique<RtDenoiseDescriptorSet>(device);
            }

            RtDenoiseDescriptorSet &set = *lightDenoiseSets[i];
            set.setTexture(set.gGuide, lightGuideTexture.get(), RenderTextureLayout::GENERAL);
            set.setTexture(set.gDirectIn, directIn, RenderTextureLayout::GENERAL);
            set.setTexture(set.gIndirectIn, indirectIn, RenderTextureLayout::GENERAL);
            set.setTexture(set.gVarianceIn, varianceIn, RenderTextureLayout::GENERAL);
            set.setTexture(set.gDirectOut, directOut, RenderTextureLayout::GENERAL);
            set.setTexture(set.gIndirectOut, indirectOut, RenderTextureLayout::GENERAL);
            set.setTexture(set.gVarianceOut, varianceOut, RenderTextureLayout::GENERAL);

            DenoiseCB denoiseCB = {};
            denoiseCB.resolution[0] = textureWidth;
            denoiseCB.resolution[1] = textureHeight;
            denoiseCB.stepSize = 1u << i;
            denoiseCB.luminanceSigma = kDenoiseLuminanceSigma;
            denoiseCB.depthSigma = kDenoiseDepthSigma;
            denoiseCB.normalPower = kDenoiseNormalPower;
            worker->commandList->setComputePushConstants(0, &denoiseCB);
            worker->commandList->setComputeDescriptorSet(set.get(), 0);
            worker->commandList->dispatch((textureWidth + 7) / 8, (textureHeight + 7) / 8, 1);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, {
                RenderTextureBarrier(directOut, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(indirectOut, RenderTextureLayout::GENERAL),
                RenderTextureBarrier(varianceOut, RenderTextureLayout::GENERAL)
            });

            directIn = directOut;
            indirectIn = indirectOut;
            varianceIn = varianceOut;
        }

        worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, {
            RenderTextureBarrier(filteredDirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ),
            RenderTextureBarrier(filteredIndirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ)
        });

        return true;
    }

    void RaytracingResources::logLightQualityDiagnostics() {
        const auto ms = [this](PassTimestamp pass) {
            return passMs[uint32_t(pass) - 1];
        };

        rtDiagLog("  light quality: samples %s (%u sun shadow, light, bounce and occlusion rays per pixel and frame), denoiser %s (%u edge-aware filter passes%s), accumulation %s (new frame weight at least %.2f, history up to %.0f frames, clipped to %.1f standard deviations of the neighborhood)",
            kLightSampleNames[lightSamplesLevel], rtParams.lightSamples, kDenoiserNames[denoiserLevel], denoisePassCount,
            (denoisePassCount == 0) ? ", bounce light blur only" : "", kAccumulationNames[accumulationLevel], rtParams.temporalMinBlend,
            rtParams.temporalMaxHistory, rtParams.temporalClipGamma);
        rtDiagLog("  light quality GPU ms: direct %.3f and indirect %.3f (both scale with the samples), temporal accumulation %.3f, denoiser %.3f; temporal pass %s, denoiser pass %s",
            ms(PassTimestamp::Direct), ms(PassTimestamp::Indirect), ms(PassTimestamp::Temporal), ms(PassTimestamp::Denoise),
            (lightTemporalPipeline != nullptr) ? "ready" : "unavailable", (lightDenoisePipeline != nullptr) ? "ready" : "unavailable");
    }
};
