//
// RT64
//

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <mutex>

#include "hle/rt64_framebuffer.h"

#include "rt64_descriptor_sets.h"
#include "rt64_framebuffer_renderer_call.h"
#include "rt64_raytracing_shader_cache.h"
#include "rt64_render_target.h"
#include "rt64_render_target_manager.h"
#include "rt64_shader_library.h"
#include "rt64_upscaler.h"

namespace RT64 {
    static const RenderTextureFlags kUav = RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS;
    static const uint64_t kHitBufferBytes = 256;
    // The counters in gHitInstanceId (RtLibrary.hlsl kDiag*): 256 of them, the decal statistics from 64 on.
    static const uint64_t kDiagCounterBytes = 1024;
    // Offsets of acceleration structures and their scratch memory: D3D12 and Vulkan both accept 256 bytes.
    static const uint64_t kAccelerationStructureAlignment = 256;
    static const uint64_t kScratchBatchBudget = 32ULL << 20;

    static uint64_t roundUp(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static std::unique_ptr<RenderTexture> makeTexture(RenderDevice *device, uint32_t width, uint32_t height, RenderFormat format, RenderTextureFlags flags) {
        RenderTextureDesc desc = RenderTextureDesc::Texture2D(std::max(width, 1u), std::max(height, 1u), 1, format, flags);
        desc.committed = true;
        return device->createTexture(desc);
    }

    static void uploadToBuffer(RenderWorker *worker, std::unique_ptr<RenderBuffer> &upload, uint64_t &uploadCapacity, RenderBuffer *destination, const void *data, uint64_t size) {
        if ((destination == nullptr) || (data == nullptr) || (size == 0)) {
            return;
        }

        const uint64_t uploadSize = roundUp(std::max(size, uint64_t(256)), 256);
        if ((upload == nullptr) || (uploadCapacity < uploadSize)) {
            upload = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(uploadSize));
            uploadCapacity = uploadSize;
        }

        void *mapped = upload->map();
        if (mapped == nullptr) {
            return;
        }
        memcpy(mapped, data, size);
        if (uploadSize > size) {
            memset(static_cast<uint8_t *>(mapped) + size, 0, size_t(uploadSize - size));
        }
        upload->unmap();

        worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(destination, RenderBufferAccess::WRITE));
        worker->commandList->copyBufferRegion(destination->at(0), upload->at(0), size);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(destination, RenderBufferAccess::READ));
    }

    static void growBuffer(RenderDevice *device, std::unique_ptr<RenderBuffer> &buffer, uint64_t &capacity, uint64_t size, RenderBufferFlags flags) {
        if ((buffer != nullptr) && (capacity >= size)) {
            return;
        }

        capacity = roundUp(std::max(size, uint64_t(256)), 256);
        buffer = device->createBuffer(RenderBufferDesc::DefaultBuffer(capacity, flags));
    }

    RaytracingResources::RaytracingResources(RenderWorker *worker, UserConfiguration::GraphicsAPI) {
        device = worker->device;
        static_assert(sizeof(interop::PointLight) == 80, "Point light packing must match the ray tracing shader.");
    }

    RaytracingResources::~RaytracingResources() = default;

    void RaytracingResources::resetBottomLevelAS() {
        pendingMeshes.clear();
        pendingMeshLevels.clear();
        bottomLevelASVector.clear();
        geometryBuffers.clear();
        proxyVertexData.clear();
        proxyIndexData.clear();
        proxyMeshRefs.clear();
    }

    void RaytracingResources::addBottomLevelASMesh(const RenderBottomLevelASMesh &mesh) {
        pendingMeshes.push_back(mesh);
    }

    uint32_t RaytracingResources::addBillboardProxyMesh(const std::vector<float> &positions) {
        const uint32_t vertexCount = uint32_t(positions.size() / 4);
        ProxyMeshRef ref;
        ref.meshIndex = uint32_t(pendingMeshes.size());
        ref.vertexOffset = proxyVertexData.size() * sizeof(float);
        ref.indexOffset = proxyIndexData.size() * sizeof(uint32_t);
        proxyVertexData.insert(proxyVertexData.end(), positions.begin(), positions.end());
        for (uint32_t i = 0; i < vertexCount; i++) {
            proxyIndexData.push_back(i);
        }

        proxyMeshRefs.push_back(ref);

        // The buffer references are filled in by updateBottomLevelASResources.
        RenderBottomLevelASMesh mesh;
        mesh.indexFormat = RenderFormat::R32_UINT;
        mesh.vertexFormat = RenderFormat::R32G32B32_FLOAT;
        mesh.indexCount = vertexCount;
        mesh.vertexCount = vertexCount;
        mesh.vertexStride = sizeof(float) * 4;
        mesh.isOpaque = false;
        pendingMeshes.push_back(mesh);
        return ref.meshIndex;
    }

    Upscaler *RaytracingResources::getUpscaler(UpscaleMode mode) const {
        switch (mode) {
        case UpscaleMode::FSR:
            return fsr3Upscaler.get();
        case UpscaleMode::FSR1:
            return fsr1Upscaler.get();
        default:
            return nullptr;
        }
    }

    void RaytracingResources::updateMultisampling() {
    }

    void RaytracingResources::setRaytracingConfig(const RaytracingConfiguration &rtConfig, bool resolutionChanged) {
        rtParams.diSamples = uint32_t(std::max(rtConfig.diSamples, 0));
        rtParams.giSamples = uint32_t(std::max(rtConfig.giSamples, 0));
        rtParams.maxLights = uint32_t(std::max(rtConfig.maxLights, 1));
        rtParams.motionBlurStrength = std::max(rtConfig.motionBlurStrength, 0.0f);
        rtParams.motionBlurSamples = uint32_t(std::max(rtConfig.motionBlurSamples, 0));
        rtParams.visualizationMode = rtConfig.visualizationMode;
        denoiserEnabled = rtConfig.denoiserEnabled;
        upscalerMode = rtConfig.upscalerMode;
        upscalerReactiveMask = rtConfig.upscalerReactiveMask;
        upscalerLockMask = rtConfig.upscalerLockMask;
        maxReflections = std::max(rtConfig.maxReflections, 0);
        upscaleActive = false;
        if (resolutionChanged) {
            updateOutputBuffers = true;
        }
    }

    void RaytracingResources::createOutputBuffers(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t displayWidth, uint32_t displayHeight) {
        textureWidth = std::max(width, 1u);
        textureHeight = std::max(height, 1u);
        this->displayWidth = std::max(displayWidth, 1u);
        this->displayHeight = std::max(displayHeight, 1u);
        upscalerResetPending = true;
        rtParams.resolution = { float(textureWidth), float(textureHeight), 1.0f / float(textureWidth), 1.0f / float(textureHeight) };
        transitionOutputBuffers = true;

        const RenderTextureFlags uav = kUav;
        viewDirectionTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        shadingPositionTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        shadingNormalTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        shadingSpecularTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        diffuseTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        instanceIdTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32_SINT, uav);
        reflectionTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        refractionTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        transparentTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        flowTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32G32_FLOAT, uav);
        reactiveMaskTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32_FLOAT, uav);
        lockMaskTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32_FLOAT, uav);
        for (int i = 0; i < 2; i++) {
            directLightTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            indirectLightTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            normalRoughnessTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            depthTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32_FLOAT, uav);
            filteredDirectLightTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            filteredIndirectLightTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            momentsTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            lightVarianceTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            volumetricTexture[i] = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
            outputTexture[i] = worker->device->createTexture(RenderTextureDesc::ColorTarget(textureWidth, textureHeight, RenderFormat::R32G32B32A32_FLOAT));
            const RenderTexture *color = outputTexture[i].get();
            outputFramebuffer[i] = worker->device->createFramebuffer(RenderFramebufferDesc(&color, 1));
        }

        // Full precision: the view depth in alpha can exceed half float range in large areas.
        lightGuideTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32G32B32A32_FLOAT, uav);
        lumaAverageTexture = makeTexture(device, 1, 1, RenderFormat::R32_FLOAT, uav);
        downscaledOutputTexture = makeTexture(device, std::max(textureWidth / 8, 1u), std::max(textureHeight / 8, 1u), RenderFormat::R32G32B32A32_FLOAT, uav);
        upscaledOutputTexture = makeTexture(device, this->displayWidth, this->displayHeight, RenderFormat::R16G16B16A16_FLOAT, uav);
        motionVectorTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R16G16_FLOAT, uav);
        deviceDepthTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32_FLOAT, uav);

        const RenderBufferFlags formatted = RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::FORMATTED;
        auto growOnce = [&](std::unique_ptr<RenderBuffer> &buffer, uint64_t size, RenderBufferFlags flags) {
            uint64_t capacity = 0;
            growBuffer(device, buffer, capacity, size, flags);
        };
        growOnce(hitVelocityDistanceBuffer, kHitBufferBytes, formatted);
        growOnce(hitColorBuffer, kHitBufferBytes, formatted);
        // gHitNormalFog holds the smoothed color of each emissive light slot, then its smoothed glow and color stats,
        // then the texture stats of the material probes.
        growOnce(hitNormalFogBuffer, HitNormalFogEntries * sizeof(float) * 4, formatted);
        growOnce(hitInstanceIdBuffer, kDiagCounterBytes, formatted);
        hitVelocityDistanceBufferView = hitVelocityDistanceBuffer->createBufferFormattedView(RenderFormat::R32G32B32A32_FLOAT);
        hitColorBufferView = hitColorBuffer->createBufferFormattedView(RenderFormat::R32G32B32A32_FLOAT);
        hitNormalFogBufferView = hitNormalFogBuffer->createBufferFormattedView(RenderFormat::R32G32B32A32_FLOAT);
        hitInstanceIdBufferView = hitInstanceIdBuffer->createBufferFormattedView(RenderFormat::R32_UINT);
        {
            uint64_t histogramCapacity = 0;
            growBuffer(device, histogramBuffer, histogramCapacity, 256, RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS);
        }

        composeSet.reset();
        postProcessSet.reset();
        lumaSetSet.reset();
        lumaSet.reset();
        lumaAvgSet.reset();
        lumaClearSet.reset();
        downscaleSet.reset();
        indirectFilterSets[0].reset();
        indirectFilterSets[1].reset();
    }

    void RaytracingResources::updateShaderSets(RenderWorker *worker, const ShaderLibrary *shaderLibrary) {
        if ((shaderLibrary == nullptr) || (diffuseTexture == nullptr)) {
            return;
        }

        const SamplerLibrary &samplers = shaderLibrary->samplerLibrary;
        if (composeSet == nullptr) {
            composeSet = std::make_unique<RaytracingComposeDescriptorSet>(samplers, device);
        }
        if (postProcessSet == nullptr) {
            postProcessSet = std::make_unique<PostProcessDescriptorSet>(samplers, device);
        }
        if (lumaSetSet == nullptr) {
            lumaSetSet = std::make_unique<HistogramSetDescriptorSet>(device);
        }
        if (indirectFilterSets[0] == nullptr) {
            indirectFilterSets[0] = std::make_unique<GaussianFilterDescriptorSet>(samplers, device);
            indirectFilterSets[1] = std::make_unique<GaussianFilterDescriptorSet>(samplers, device);
        }

        const int current = swapBuffers ? 1 : 0;
        composeSet->setTexture(composeSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gDiffuse, diffuseTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gDirectLight, filteredDirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gIndirectLight, filteredIndirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gReflection, reflectionTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gRefraction, refractionTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gTransparent, transparentTexture.get(), RenderTextureLayout::SHADER_READ);
        composeSet->setTexture(composeSet->gVolumetric, volumetricTexture[current].get(), RenderTextureLayout::SHADER_READ);

        if (rtParamsBuffer.get() != nullptr) {
            postProcessSet->setBuffer(postProcessSet->RtParams, rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        }
        RenderTexture *postProcessInput = upscaleActive ? upscaledOutputTexture.get() : outputTexture[current].get();
        postProcessSet->setTexture(postProcessSet->gInput, postProcessInput, RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gLumaAvg, lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gCoverage, diffuseTexture.get(), RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gVolumetric, volumetricTexture[current].get(), RenderTextureLayout::SHADER_READ);
        lumaSetSet->setTexture(lumaSetSet->LuminanceOutput, lumaAverageTexture.get(), RenderTextureLayout::GENERAL);

        indirectFilterSets[0]->setTexture(indirectFilterSets[0]->gInput, filteredIndirectLightTexture[0].get(), RenderTextureLayout::SHADER_READ);
        indirectFilterSets[0]->setTexture(indirectFilterSets[0]->gOutput, filteredIndirectLightTexture[1].get(), RenderTextureLayout::GENERAL);
        indirectFilterSets[1]->setTexture(indirectFilterSets[1]->gInput, filteredIndirectLightTexture[1].get(), RenderTextureLayout::SHADER_READ);
        indirectFilterSets[1]->setTexture(indirectFilterSets[1]->gOutput, filteredIndirectLightTexture[0].get(), RenderTextureLayout::GENERAL);
        (void)worker;
    }

    void RaytracingResources::updateInterleavedRenderTargets(RenderWorker *worker, uint32_t width, uint32_t height, uint32_t count, const RenderMultisampling &multisampling, bool usesHDR) {
        interleavedColorTargetVector.resize(count);
        interleavedDepthTargetVector.resize(count);
        interleavedFramebufferStorageVector.resize(count);
        for (uint32_t i = 0; i < count; i++) {
            if (interleavedColorTargetVector[i] == nullptr) {
                interleavedColorTargetVector[i] = std::make_unique<RenderTarget>(0, Framebuffer::Type::Color, multisampling, usesHDR);
                interleavedDepthTargetVector[i] = std::make_unique<RenderTarget>(0, Framebuffer::Type::Depth, multisampling, usesHDR);
                interleavedFramebufferStorageVector[i] = std::make_unique<RenderFramebufferStorage>();
            }
            interleavedColorTargetVector[i]->resize(worker, std::max(width, 1u), std::max(height, 1u));
            interleavedDepthTargetVector[i]->resize(worker, std::max(width, 1u), std::max(height, 1u));
            interleavedFramebufferStorageVector[i]->setup(device, {}, interleavedColorTargetVector[i].get(), interleavedDepthTargetVector[i].get());
        }
    }

    void RaytracingResources::updateBottomLevelASResources(RenderWorker *worker) {
        bottomLevelASVector.clear();
        geometryBuffers.clear();
        bottomLevelASVector.reserve(pendingMeshes.size());
        pendingMeshLevels.assign(pendingMeshes.size(), -1);

        if (!proxyMeshRefs.empty()) {
            const RenderBufferFlags proxyFlags = RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT | RenderBufferFlag::STORAGE;
            const uint64_t vertexBytes = proxyVertexData.size() * sizeof(float);
            const uint64_t indexBytes = proxyIndexData.size() * sizeof(uint32_t);
            growBuffer(device, proxyVertexBuffer, proxyVertexCapacity, vertexBytes, proxyFlags);
            growBuffer(device, proxyIndexBuffer, proxyIndexCapacity, indexBytes, proxyFlags);
            uploadToBuffer(worker, proxyVertexUpload, proxyVertexUploadCapacity, proxyVertexBuffer.get(), proxyVertexData.data(), vertexBytes);
            uploadToBuffer(worker, proxyIndexUpload, proxyIndexUploadCapacity, proxyIndexBuffer.get(), proxyIndexData.data(), indexBytes);
            for (const ProxyMeshRef &ref : proxyMeshRefs) {
                RenderBottomLevelASMesh &mesh = pendingMeshes[ref.meshIndex];
                mesh.vertexBuffer = proxyVertexBuffer->at(ref.vertexOffset);
                mesh.indexBuffer = proxyIndexBuffer->at(ref.indexOffset);
            }
        }

        if ((dummyVertexBuffer == nullptr) || (dummyIndexBuffer == nullptr)) {
            const float vertices[12] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            const uint32_t indices[3] = { 0, 1, 2 };
            uint64_t vertexCapacity = 0;
            uint64_t indexCapacity = 0;
            growBuffer(device, dummyVertexBuffer, vertexCapacity, sizeof(vertices), RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT | RenderBufferFlag::STORAGE);
            growBuffer(device, dummyIndexBuffer, indexCapacity, sizeof(indices), RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT | RenderBufferFlag::STORAGE);
            uploadToBuffer(worker, dummyVertexUpload, dummyVertexUploadCapacity, dummyVertexBuffer.get(), vertices, sizeof(vertices));
            uploadToBuffer(worker, dummyIndexUpload, dummyIndexUploadCapacity, dummyIndexBuffer.get(), indices, sizeof(indices));
        }

        uint64_t scratchNeeded = 0;
        uint64_t scratchBatchUsed = 0;
        uint64_t poolNeeded = 0;
        bottomLevelScratchBatches = 0;
        for (size_t pendingIndex = 0; pendingIndex < pendingMeshes.size(); pendingIndex++) {
            RenderBottomLevelASMesh mesh = pendingMeshes[pendingIndex];
            if ((mesh.indexCount == 0) || (mesh.vertexCount == 0) || (mesh.indexBuffer.ref == nullptr) || (mesh.vertexBuffer.ref == nullptr)) {
                mesh = RenderBottomLevelASMesh(dummyIndexBuffer->at(0), dummyVertexBuffer->at(0), RenderFormat::R32_UINT, RenderFormat::R32G32B32_FLOAT, 3, 3, sizeof(float) * 4, true);
            }

            BottomLevel level;
            level.mesh = mesh;
            worker->device->setBottomLevelASBuildInfo(level.buildInfo, &level.mesh, 1, true, false);
            if (level.buildInfo.accelerationStructureSize == 0) {
                continue;
            }

            level.offset = poolNeeded;
            poolNeeded += roundUp(level.buildInfo.accelerationStructureSize, kAccelerationStructureAlignment);

            const uint64_t scratchSize = roundUp(level.buildInfo.scratchSize, kAccelerationStructureAlignment);
            if ((bottomLevelScratchBatches == 0) || ((scratchBatchUsed > 0) && (scratchBatchUsed + scratchSize > kScratchBatchBudget))) {
                level.newScratchBatch = true;
                bottomLevelScratchBatches++;
                scratchBatchUsed = 0;
            }

            level.scratchOffset = scratchBatchUsed;
            scratchBatchUsed += scratchSize;
            scratchNeeded = std::max(scratchNeeded, scratchBatchUsed);

            if (mesh.indexBuffer.ref != nullptr) {
                geometryBuffers.push_back(const_cast<RenderBuffer *>(mesh.indexBuffer.ref));
            }
            if (mesh.vertexBuffer.ref != nullptr) {
                geometryBuffers.push_back(const_cast<RenderBuffer *>(mesh.vertexBuffer.ref));
            }
            pendingMeshLevels[pendingIndex] = int32_t(bottomLevelASVector.size());
            bottomLevelASVector.push_back(std::move(level));
        }

        std::sort(geometryBuffers.begin(), geometryBuffers.end());
        geometryBuffers.erase(std::unique(geometryBuffers.begin(), geometryBuffers.end()), geometryBuffers.end());

        // The previous frame finished on the GPU before this one is recorded, so the pool is reused in place. Growing it
        // with headroom keeps the allocation rare as the scene changes.
        bottomLevelPoolUsed = poolNeeded;
        if ((poolNeeded > 0) && ((bottomLevelPool == nullptr) || (poolNeeded > bottomLevelPoolCapacity))) {
            bottomLevelPoolCapacity = roundUp(poolNeeded + poolNeeded / 2, 1ULL << 20);
            bottomLevelPool = worker->device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(bottomLevelPoolCapacity));
        }

        if (bottomLevelPool != nullptr) {
            for (BottomLevel &level : bottomLevelASVector) {
                level.accelerationStructure = worker->device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL,
                    bottomLevelPool->at(level.offset), level.buildInfo.accelerationStructureSize));
            }
        }

        if (scratchNeeded > scratchCapacity) {
            scratchCapacity = roundUp(scratchNeeded, 256);
            scratchBuffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(scratchCapacity, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH | RenderBufferFlag::UNORDERED_ACCESS));
        }
    }

    void RaytracingResources::submitBottomLevelASCreation(RenderWorker *worker) {
        if ((scratchBuffer == nullptr) || (bottomLevelPool == nullptr) || bottomLevelASVector.empty()) {
            return;
        }

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(scratchBuffer.get(), RenderBufferAccess::WRITE));
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(bottomLevelPool.get(), RenderBufferAccess::WRITE));

        // After the first barrier, which ends any active render pass: query pools can't be reset inside one.
        beginBuildTiming(worker);
        if (!geometryBuffers.empty()) {
            std::vector<RenderBufferBarrier> barriers;
            barriers.reserve(geometryBuffers.size());
            for (RenderBuffer *buffer : geometryBuffers) {
                barriers.emplace_back(buffer, RenderBufferAccess::READ);
            }
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, barriers);
        }

        for (size_t i = 0; i < bottomLevelASVector.size(); i++) {
            BottomLevel &level = bottomLevelASVector[i];
            if (level.newScratchBatch && (i > 0)) {
                worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(scratchBuffer.get(), RenderBufferAccess::WRITE));
            }

            worker->commandList->buildBottomLevelAS(level.accelerationStructure.get(), scratchBuffer->at(level.scratchOffset), level.buildInfo);
        }

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(bottomLevelPool.get(), RenderBufferAccess::WRITE));

        // Hand the geometry back to the raster passes. Trace rays transitions it again.
        if (!geometryBuffers.empty()) {
            std::vector<RenderBufferBarrier> barriers;
            barriers.reserve(geometryBuffers.size());
            for (RenderBuffer *buffer : geometryBuffers) {
                barriers.emplace_back(buffer, RenderBufferAccess::READ);
            }
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS, barriers);
        }
    }

    void RaytracingResources::updateTopLevelASResources(RenderWorker *worker, const std::vector<RaytracingInstance> &sceneInstances) {
        std::vector<RenderTopLevelASInstance> instances;
        instances.reserve(sceneInstances.size());
        for (const RaytracingInstance &sceneInstance : sceneInstances) {
            // Every ray traced draw in every framebuffer adds a mesh, so the scene's instances are not
            // the first N meshes. Use the mesh recorded for this draw.
            const uint32_t meshIndex = sceneInstance.bottomLevelIndex;
            if ((meshIndex >= pendingMeshLevels.size()) || (pendingMeshLevels[meshIndex] < 0)) {
                continue;
            }
            const BottomLevel &level = bottomLevelASVector[pendingMeshLevels[meshIndex]];
            const uint32_t mask = (sceneInstance.queryMask == 0) ? 0xFF : sceneInstance.queryMask;
            const bool cullDisable = sceneInstance.cullDisable;

            // Instance ID: render index in the low 21 bits, then the billboard proxy, culling and alpha test flags for
            // the any-hit shaders.
            const uint32_t instanceID = (sceneInstance.renderIndex & 0x1FFFFFu) | (sceneInstance.proxy ? (1u << 21) : 0u) |
                (sceneInstance.culling ? (1u << 22) : 0u) | (sceneInstance.alphaTested ? (1u << 23) : 0u);
            instances.emplace_back(bottomLevelPool->at(level.offset), instanceID, mask, 0, cullDisable, RenderAffineTransform());
        }

        if (instances.empty()) {
            topLevelAS.reset();
            topLevelBuffer.reset();
            return;
        }

        worker->device->setTopLevelASBuildInfo(topLevelBuildInfo, instances.data(), uint32_t(instances.size()), true, false);
        topLevelBuffer = worker->device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(topLevelBuildInfo.accelerationStructureSize));
        topLevelAS = worker->device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::TOP_LEVEL, topLevelBuffer->at(0), topLevelBuildInfo.accelerationStructureSize));
        if (topLevelBuildInfo.scratchSize > scratchCapacity) {
            scratchCapacity = roundUp(topLevelBuildInfo.scratchSize, 256);
            scratchBuffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(scratchCapacity, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH | RenderBufferFlag::UNORDERED_ACCESS));
        }

        growBuffer(device, instancesBuffer, instancesCapacity, std::max(topLevelBuildInfo.instancesBufferData.size(), size_t(256)), RenderBufferFlag::ACCELERATION_STRUCTURE_INPUT | RenderBufferFlag::STORAGE);
    }

    void RaytracingResources::submitTopLevelASCreation(RenderWorker *worker) {
        if ((topLevelAS == nullptr) || (scratchBuffer == nullptr) || (instancesBuffer == nullptr) || topLevelBuildInfo.instancesBufferData.empty()) {
            return;
        }

        uploadToBuffer(worker, instancesUpload, instancesUploadCapacity, instancesBuffer.get(), topLevelBuildInfo.instancesBufferData.data(), topLevelBuildInfo.instancesBufferData.size());
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(scratchBuffer.get(), RenderBufferAccess::WRITE));
        worker->commandList->buildTopLevelAS(topLevelAS.get(), scratchBuffer->at(0), instancesBuffer->at(0), topLevelBuildInfo);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(topLevelBuffer.get(), RenderBufferAccess::WRITE));
    }

    void RaytracingResources::setShaderReadBuffers(RenderBuffer *normalBuffer, RenderBuffer *shadedColorBuffer) {
        shaderReadBuffers.clear();
        if (normalBuffer != nullptr) {
            shaderReadBuffers.push_back(normalBuffer);
        }
        if (shadedColorBuffer != nullptr) {
            shaderReadBuffers.push_back(shadedColorBuffer);
        }
    }

    void RaytracingResources::prepareForDispatch(RenderWorker *worker) {
        std::vector<RenderBufferBarrier> barriers;
        barriers.reserve(geometryBuffers.size() + shaderReadBuffers.size());
        for (RenderBuffer *buffer : geometryBuffers) {
            barriers.emplace_back(buffer, RenderBufferAccess::READ);
        }
        for (RenderBuffer *buffer : shaderReadBuffers) {
            barriers.emplace_back(buffer, RenderBufferAccess::READ);
        }
        if (!barriers.empty()) {
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, barriers);
        }
    }

    void RaytracingResources::createShaderBindingTable(RenderWorker *worker, const RaytracingState *rtState, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount, const std::vector<RenderPipelineProgram> &hitGroups) {
        if ((rtState == nullptr) || !rtState->valid || (rtState->pipeline == nullptr) || hitGroups.empty()) {
            return;
        }

        RenderShaderBindingGroups groups(
            RenderShaderBindingGroup(rtState->rayGenPrograms.data(), uint32_t(rtState->rayGenPrograms.size())),
            RenderShaderBindingGroup(rtState->missPrograms.data(), uint32_t(rtState->missPrograms.size())),
            RenderShaderBindingGroup(hitGroups.data(), uint32_t(hitGroups.size())));
        worker->device->setShaderBindingTableInfo(shaderBindingTableInfo, groups, rtState->pipeline.get(), descriptorSets, descriptorSetCount);
        if (shaderBindingTableInfo.tableBufferData.empty()) {
            return;
        }

        growBuffer(device, shaderBindingTableBuffer, shaderBindingTableCapacity, shaderBindingTableInfo.tableBufferData.size(), RenderBufferFlag::SHADER_BINDING_TABLE | RenderBufferFlag::STORAGE);
        uploadToBuffer(worker, shaderBindingTableUpload, shaderBindingTableUploadCapacity, shaderBindingTableBuffer.get(), shaderBindingTableInfo.tableBufferData.data(), shaderBindingTableInfo.tableBufferData.size());
    }

    // Which emitter kind (rt_emitters.json "only kinds") a candidate piece is.
    static uint32_t emitterKindOf(const EmissiveCandidate &candidate) {
        if (candidate.untextured) {
            return RtEmitterKindUntextured;
        }

        switch (candidate.category) {
        case EmissiveCategory::Additive:
            return RtEmitterKindAdditive;
        case EmissiveCategory::Cutout:
            return RtEmitterKindCutout;
        default:
            return RtEmitterKindBlended;
        }
    }

    static const char *emitterKindName(uint32_t kind) {
        switch (kind) {
        case RtEmitterKindAdditive: return "additive";
        case RtEmitterKindCutout: return "cutout";
        case RtEmitterKindUntextured: return "untextured";
        default: return "blended";
        }
    }

    // Emissive light shape. The range is where the light fades out; the size sets how close the light can get before
    // it stops brightening (see emissiveAttenuation in RtLibrary.hlsl).
    static const float kEmissiveRangePerExtent = 12.0f;
    static const float kEmissiveMinRange = 80.0f;
    static const float kEmissiveMaxRange = 500.0f;
    static const float kEmissiveMaxClusterExtent = 200.0f;
    static const float kEmissivePositionSmoothing = 0.35f;
    // Default fade per frame while the source is seen and once it's gone (rt_emitters.json can set its own): about 6
    // frames in and 10 frames out.
    static const float kEmissiveFadeIn = 1.0f / 6.0f;
    // Hit sparks (untextured effects): one flash gathered from pieces within kEmissiveSparkMergeRadius, two frames in
    // and six out by default, lighting no further than kEmissiveSparkMaxRange.
    static const float kEmissiveSparkMergeRadius = 40.0f;
    // Legacy glow of unlisted untextured effects: intensity grows with the effect's size, from a small flash for hit
    // sparks to a strong light for the item pillar, at most kEmissiveLegacyMaxLights lights per draw signature.
    static const float kEmissiveLegacyExtentPerIntensity = 40.0f;
    static const float kEmissiveLegacyMinIntensity = 0.45f;
    static const float kEmissiveLegacyMaxIntensity = 1.5f;
    static const uint32_t kEmissiveLegacyMaxLights = 2;
    static const float kEmissiveSparkFadeIn = 0.5f;
    static const float kEmissiveSparkFadeOut = 1.0f / 6.0f;
    static const float kEmissiveSparkMaxRange = 120.0f;
    // Blended spell effects can be brief, so they reach full strength in about 3 frames.
    static const float kEmissiveEffectFadeIn = 1.0f / 3.0f;
    // Ground rings (the battle circle, radius about 100 with pieces 25 to 40 across) against spark bursts, which are
    // small and in the air.
    static const float kEmissiveMinRingRadius = 40.0f;
    static const float kEmissiveMinRingPieceExtent = 8.0f;
    static const float kEmissiveFadeOut = 1.0f / 10.0f;
    static const uint32_t kEmissiveResetFlag = 0x100;
    static const uint32_t kEmissiveLightFlag = 0x200;
    static const uint32_t kEmissiveNoSampleFlag = 0x400;
    static const uint32_t kEmissiveCategoryShift = 12;
    // The light takes its color from the sampled sprite (rt_emitters.json "color": "sample").
    static const uint32_t kEmissiveSampleColorFlag = 0x4000;
    // The light only shines as much as the GPU finds its sprite glows (legacy glow).
    static const uint32_t kEmissiveGlowGateFlag = 0x8000;
    // A manual light (rt_lights.json) with "casts shadows" false: it lights through walls, as a fill light.
    static const uint32_t kManualNoShadowFlag = 0x10000;

    static const char *emissiveCategoryName(EmissiveCategory category) {
        switch (category) {
        case EmissiveCategory::Additive:
            return "additive";
        case EmissiveCategory::Blended:
            return "blended";
        case EmissiveCategory::Cutout:
            return "cutout";
        default:
            return "unknown";
        }
    }

    // A source whose sampled glow stays below the low mark after a few samples counts as non-emissive, until a later
    // sample passes the high mark. Non-emissive sources are sampled again once in a while, without emitting.
    static const float kEmissiveVerdictMinGlow = 0.05f;
    static const float kEmissiveVerdictAcceptGlow = 0.15f;
    static const uint32_t kEmissiveVerdictMinSamples = 3;
    // A source whose glow once reached kEmissiveConfirmedGlow is only demoted after this many dark samples in a row
    // (about 4 seconds of being sampled dark). Its peak decays by kEmissivePeakGlowDecay per sample.
    // A cool colored effect seen in every frame (with gaps of at most kEmissiveVisibilityGapFrames) for this long is
    // scenery, like Larapool's fountain: about 12 seconds, longer than any spell.
    static const uint32_t kEmissivePersistentFrames = 720;
    static const uint32_t kEmissiveVisibilityGapFrames = 30;
    static const float kEmissiveConfirmedGlow = 0.4f;
    static const uint32_t kEmissiveConfirmedDarkSamples = 240;
    static const float kEmissivePeakGlowDecay = 0.999f;
    static const uint32_t kEmissiveVerdictResampleFrames = 120;
    static const uint32_t kEmissiveVerdictForgetFrames = 3600;

    bool RaytracingResources::emissiveKeyRejected(uint64_t key) const {
        auto it = emissiveVerdicts.find(key);
        if (it == emissiveVerdicts.end()) {
            return false;
        }

        const EmissiveVerdict &verdict = it->second;
        return verdict.rejected && ((emissiveFrame - verdict.lastSampledFrame) < kEmissiveVerdictResampleFrames);
    }

    bool RaytracingResources::emissiveKeySuppressed(uint64_t key) const {
        auto it = emissiveVerdicts.find(key);
        return (it != emissiveVerdicts.end()) && it->second.rejected;
    }

    float RaytracingResources::emissiveKeyGlow(uint64_t key) const {
        auto it = emissiveVerdicts.find(key);
        return (it != emissiveVerdicts.end()) ? it->second.glow : -1.0f;
    }

    void RaytracingResources::updateMaterialProbes(const RaytracingScene &rtScene) {
        // The probes follow the emissive lights in the lights buffer, as fake point lights that name the draw.
        static const uint32_t kMaterialProbeFlag = 0x800;
        frameMaterialProbes.clear();
        for (const RtMaterialProbe &probe : rtScene.materialProbes) {
            if (frameMaterialProbes.size() >= MaterialProbeSlots) {
                break;
            }

            interop::PointLight light = {};
            light.flickerIntensity = float(probe.renderIndex);
            light.shadowOffset = float(probe.triangleCount);
            light.groupBits = uint32_t(frameMaterialProbes.size()) | kMaterialProbeFlag;
            frameLights.push_back(light);
            frameMaterialProbes.push_back(probe);
        }

        rtParams.materialProbeCount = uint32_t(frameMaterialProbes.size());
        frameMaterialKeys = rtScene.materialKeys;
        frameDrawSignatures = rtScene.drawSignatures;
    }

    void RaytracingResources::copyMaterialPreviews(RenderWorker *worker, const RaytracingScene &rtScene) {
        for (const RaytracingScene::MaterialPreview &preview : rtScene.materialPreviews) {
            // Rows of a texture copy into a buffer are aligned to 256 bytes.
            const uint32_t rowWidth = ((preview.width * 4 + 255) / 256) * 64;
            PendingMaterialPreview pending;
            pending.key = preview.key;
            pending.width = preview.width;
            pending.height = preview.height;
            pending.rowWidth = rowWidth;
            pending.frame = emissiveFrame;
            pending.readback = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(rowWidth) * 4 * preview.height));
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(preview.texture, RenderTextureLayout::COPY_SOURCE));
            worker->commandList->copyTextureRegion(
                RenderTextureCopyLocation::PlacedFootprint(pending.readback.get(), RenderFormat::R8G8B8A8_UNORM, preview.width, preview.height, 1, rowWidth),
                RenderTextureCopyLocation::Subresource(preview.texture));
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(preview.texture, RenderTextureLayout::SHADER_READ));
            pendingMaterialPreviews.push_back(std::move(pending));
        }
    }

    void RaytracingResources::readMaterialPreviews() {
        // Copies recorded a few frames ago have finished on the GPU.
        const uint32_t ReadbackDelayFrames = 4;
        for (auto it = pendingMaterialPreviews.begin(); it != pendingMaterialPreviews.end();) {
            if ((emissiveFrame - it->frame) < ReadbackDelayFrames) {
                it++;
                continue;
            }

            const uint64_t bytes = uint64_t(it->rowWidth) * 4 * it->height;
            const RenderRange readRange(0, bytes);
            const uint8_t *data = static_cast<const uint8_t *>(it->readback->map(0, &readRange));
            if (data != nullptr) {
                std::vector<uint8_t> pixels(size_t(it->width) * it->height * 4);
                for (uint32_t y = 0; y < it->height; y++) {
                    memcpy(pixels.data() + size_t(y) * it->width * 4, data + size_t(y) * it->rowWidth * 4, size_t(it->width) * 4);
                }

                it->readback->unmap(0, nullptr);
                rtMaterialDatabase().submitPreview(it->key, std::move(pixels), it->width, it->height);
            }

            it = pendingMaterialPreviews.erase(it);
        }
    }

    void RaytracingResources::copyEmissiveGlow(RenderWorker *worker) {
        if (((rtParams.emissiveLightCount == 0) && (rtParams.materialProbeCount == 0)) || (hitNormalFogBuffer == nullptr)) {
            return;
        }

        // The glow half of the emissive slots, then the material probes.
        const uint32_t ringIndex = emissiveCopyRingIndex;
        const uint64_t glowOffset = EmissiveSlots * sizeof(float) * 4;
        const uint64_t copyBytes = (HitNormalFogEntries - EmissiveSlots) * sizeof(float) * 4;
        if (emissiveReadback[ringIndex] == nullptr) {
            emissiveReadback[ringIndex] = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(copyBytes));
        }

        worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(hitNormalFogBuffer.get(), RenderBufferAccess::READ));
        worker->commandList->copyBufferRegion(emissiveReadback[ringIndex]->at(0), hitNormalFogBuffer->at(glowOffset), copyBytes);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(hitNormalFogBuffer.get(), RenderBufferAccess::WRITE));
        memcpy(emissiveReadbackKeys[ringIndex], emissiveSlotKeys, sizeof(emissiveSlotKeys));
        memcpy(emissiveReadbackSampled[ringIndex], emissiveSlotSampled, sizeof(emissiveSlotSampled));
        materialProbeRing[ringIndex] = frameMaterialProbes;
        emissiveReadbackPending[ringIndex] = true;
    }

    void RaytracingResources::readEmissiveGlow() {
        // The oldest copy in the ring, written two frames ago; its command list has finished by now.
        const uint32_t ringIndex = (emissiveFrame + 1) % EmissiveReadbackDepth;
        if (emissiveReadbackPending[ringIndex] && (emissiveReadback[ringIndex] != nullptr)) {
            const uint64_t glowBytes = (HitNormalFogEntries - EmissiveSlots) * sizeof(float) * 4;
            const RenderRange readRange(0, glowBytes);
            const float *glows = static_cast<const float *>(emissiveReadback[ringIndex]->map(0, &readRange));
            if (glows != nullptr) {
                for (uint32_t s = 0; s < EmissiveSlots; s++) {
                    if (!emissiveReadbackSampled[ringIndex][s]) {
                        continue;
                    }

                    const uint64_t key = emissiveReadbackKeys[ringIndex][s];
                    EmissiveVerdict &verdict = emissiveVerdicts[key];
                    const float glow = glows[s * 4];
                    verdict.coolTint = (glows[s * 4 + 3] >= 3.5f);
                    verdict.glow = (verdict.samples == 0) ? glow : (verdict.glow + (glow - verdict.glow) * 0.5f);
                    verdict.peakGlow = std::max(glow, verdict.peakGlow * kEmissivePeakGlowDecay);
                    verdict.darkStreak = (glow < kEmissiveVerdictMinGlow) ? (verdict.darkStreak + 1) : 0;
                    verdict.samples++;
                    verdict.lastSampledFrame = emissiveFrame;
                    if (verdict.rejected) {
                        verdict.rejected = (verdict.glow < kEmissiveVerdictAcceptGlow);
                    }
                    else {
                        // Sources that clearly emitted before (animated spells fade and flicker) need a long run of
                        // dark samples; others are judged after a few.
                        const bool confirmed = (verdict.peakGlow >= kEmissiveConfirmedGlow);
                        const uint32_t darkSamplesNeeded = confirmed ? kEmissiveConfirmedDarkSamples : kEmissiveVerdictMinSamples;
                        verdict.rejected = (verdict.samples >= kEmissiveVerdictMinSamples) && (verdict.glow < kEmissiveVerdictMinGlow) && (verdict.darkStreak >= darkSamplesNeeded);
                    }

                    verdict.lastSeenFrame = emissiveFrame;
                }

                // Material probes: average color and coverage, then luminance deviation and peak brightness.
                const std::vector<RtMaterialProbe> &probes = materialProbeRing[ringIndex];
                for (size_t p = 0; p < probes.size(); p++) {
                    const float *colorStats = glows + (EmissiveSlots + p * 2) * 4;
                    const float *spreadStats = colorStats + 4;
                    RtMaterialStats stats;
                    stats.color[0] = colorStats[0];
                    stats.color[1] = colorStats[1];
                    stats.color[2] = colorStats[2];
                    stats.coverage = colorStats[3];
                    stats.luminanceDeviation = spreadStats[0];
                    stats.peakBrightness = spreadStats[1];
                    if (probes[p].effect) {
                        // Effect textures only get the old bright-and-tinted sprite test, to tag their candidates.
                        const float averageBrightness = std::max(stats.color[0], std::max(stats.color[1], stats.color[2]));
                        const float darkest = std::min(stats.color[0], std::min(stats.color[1], stats.color[2]));
                        const float saturation = (averageBrightness > 1e-3f) ? ((averageBrightness - darkest) / averageBrightness) : 0.0f;
                        const float brightness = 0.5f * (averageBrightness + stats.peakBrightness);
                        if ((stats.coverage > 0.0f) && (brightness >= 0.82f) && (saturation >= 0.15f)) {
                            rtMaterialDatabase().markTextureWouldGlow(probes[p].key);
                        }

                        rtMaterialDatabase().finishProbe(probes[p].key);
                    }
                    else {
                        rtMaterialDatabase().addGuess(probes[p].key, stats, probes[p].placement);
                    }
                    materialGuessesThisSession++;
                }

                materialProbeRing[ringIndex].clear();
                emissiveReadback[ringIndex]->unmap(0, nullptr);
            }

            emissiveReadbackPending[ringIndex] = false;
        }

        if (emissiveVerdicts.size() > 2048) {
            for (auto it = emissiveVerdicts.begin(); it != emissiveVerdicts.end();) {
                if (!it->second.ring && ((emissiveFrame - it->second.lastSeenFrame) > kEmissiveVerdictForgetFrames)) {
                    it = emissiveVerdicts.erase(it);
                }
                else {
                    it++;
                }
            }
        }
    }

    void RaytracingResources::updateEmissiveLights(const RaytracingScene &rtScene, const float cameraPosition[3]) {
        readEmissiveGlow();
        emissiveCopyRingIndex = emissiveFrame % EmissiveReadbackDepth;
        frameLights.clear();
        if ((rtScene.pointLights != nullptr) && (rtScene.lightCount > 0)) {
            frameLights.insert(frameLights.end(), rtScene.pointLights, rtScene.pointLights + rtScene.lightCount);
        }

        rtParams.emissiveLightStart = uint32_t(frameLights.size());

        // Cluster the candidate pieces: a flame is often several quads. A cluster takes the most light-like category
        // of its pieces (additive, then blended, then cutout).
        struct Cluster {
            float center[3];
            float extent;
            float weight;
            uint64_t key;
            uint32_t renderIndex;
            uint32_t triangleCount;
            EmissiveCategory category;
            bool thin;
            bool untextured;
            // The rt_emitters.json entry the pieces matched.
            RtEmitterParams params;
        };

        // Thin pieces arranged around a common center form a ring of line segments (the battle circle) and are dropped,
        // even when each segment comes from its own draw.
        const uint32_t MinRingPieces = 6;
        bool rejectThinRing = false;
        emissiveRingPieces = 0;
        emissiveRings = 0;
        {
            float centroid[3] = {};
            uint32_t thinCount = 0;
            float extentSum = 0.0f;
            for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
                if (candidate.thin) {
                    for (int c = 0; c < 3; c++) {
                        centroid[c] += candidate.center[c];
                    }

                    extentSum += candidate.extent;
                    thinCount++;
                }
            }

            if (thinCount >= MinRingPieces) {
                for (int c = 0; c < 3; c++) {
                    centroid[c] /= float(thinCount);
                }

                float radiusSum = 0.0f;
                float radiusSquareSum = 0.0f;
                for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
                    if (candidate.thin) {
                        const float dx = candidate.center[0] - centroid[0];
                        const float dy = candidate.center[1] - centroid[1];
                        const float dz = candidate.center[2] - centroid[2];
                        const float radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                        radiusSum += radius;
                        radiusSquareSum += radius * radius;
                    }
                }

                const float meanRadius = radiusSum / float(thinCount);
                const float variance = std::max(radiusSquareSum / float(thinCount) - meanRadius * meanRadius, 0.0f);
                const float meanExtent = extentSum / float(thinCount);
                rejectThinRing = (meanRadius > kEmissiveMinRingRadius) && (meanRadius > (meanExtent * 2.0f)) && (std::sqrt(variance) < (meanRadius * 0.3f));
                if (rejectThinRing) {
                    emissiveRingPieces = thinCount;
                    emissiveRings = 1;
                }
            }
        }

        // Many pieces with the same texture arranged around a common center at a similar distance also form a ring
        // (the battle circle's quads), whether or not each piece is thin.
        const uint32_t MinRingKeyPieces = 8;
        thread_local std::vector<uint64_t> ringKeys;
        ringKeys.clear();
        emissiveRingKeys = 0;
        emissiveRingKeyPieces = 0;
        {
            struct KeyGroup {
                uint64_t key;
                uint32_t count;
                float centroid[3];
                float extentSum;
                float minY;
                float maxY;
                bool untextured;
            };

            thread_local std::vector<KeyGroup> groups;
            groups.clear();
            for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
                auto it = std::find_if(groups.begin(), groups.end(), [&](const KeyGroup &group) { return group.key == candidate.key; });
                if (it == groups.end()) {
                    groups.push_back({ candidate.key, 0, { 0.0f, 0.0f, 0.0f }, 0.0f, 1e30f, -1e30f, false });
                    it = groups.end() - 1;
                }

                it->count++;
                it->extentSum += candidate.extent;
                it->minY = std::min(it->minY, candidate.center[1]);
                it->maxY = std::max(it->maxY, candidate.center[1]);
                it->untextured = it->untextured || candidate.untextured;
                for (int c = 0; c < 3; c++) {
                    it->centroid[c] += candidate.center[c];
                }
            }

            // Spark bursts fly out radially from a hit, so they look like small rings in the air. A ground ring is
            // wide and flat.
            for (KeyGroup &group : groups) {
                for (int c = 0; c < 3; c++) {
                    group.centroid[c] /= float(group.count);
                }

                if (group.count < MinRingKeyPieces) {
                    continue;
                }

                float radiusSum = 0.0f;
                float radiusSquareSum = 0.0f;
                for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
                    if (candidate.key == group.key) {
                        const float dx = candidate.center[0] - group.centroid[0];
                        const float dy = candidate.center[1] - group.centroid[1];
                        const float dz = candidate.center[2] - group.centroid[2];
                        const float radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                        radiusSum += radius;
                        radiusSquareSum += radius * radius;
                    }
                }

                const float meanRadius = radiusSum / float(group.count);
                const float spread = std::sqrt(std::max(radiusSquareSum / float(group.count) - meanRadius * meanRadius, 0.0f));
                const float meanExtent = group.extentSum / float(group.count);
                const bool flat = (group.maxY - group.minY) < (meanRadius * 0.3f);
                if ((meanRadius > kEmissiveMinRingRadius) && flat && (meanRadius > (meanExtent * 1.5f)) && (spread < (meanRadius * 0.3f))) {
                    ringKeys.push_back(group.key);
                    emissiveRingKeys++;
                    emissiveRingKeyPieces += group.count;
                }
            }

            if (rejectThinRing) {
                for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
                    if (candidate.thin && (std::find(ringKeys.begin(), ringKeys.end(), candidate.key) == ringKeys.end())) {
                        ringKeys.push_back(candidate.key);
                    }
                }
            }

            // Only full rings are remembered: the partial ground pattern below also catches spells that sweep along the
            // ground (wind attacks), so it only darkens them for the frame.
            const size_t fullRingKeyCount = ringKeys.size();

            // A partly drawn ring (while it appears or animates) fails the radius test, but its pieces still lie
            // flat and spread wide around the ground, unlike a spell's pieces.
            const uint32_t MinGroundPatternPieces = 10;
            const float MinGroundPatternSpread = 60.0f;
            for (const KeyGroup &group : groups) {
                if ((group.count < MinGroundPatternPieces) || (std::find(ringKeys.begin(), ringKeys.end(), group.key) != ringKeys.end())) {
                    continue;
                }

                float horizontal = 0.0f;
                for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
                    if (candidate.key == group.key) {
                        const float dx = candidate.center[0] - group.centroid[0];
                        const float dz = candidate.center[2] - group.centroid[2];
                        horizontal = std::max(horizontal, std::sqrt(dx * dx + dz * dz));
                    }
                }

                const float meanExtent = group.extentSum / float(group.count);
                if ((horizontal > std::max(MinGroundPatternSpread, meanExtent * 3.0f)) && ((group.maxY - group.minY) < (horizontal * 0.25f))) {
                    ringKeys.push_back(group.key);
                    emissiveRingKeys++;
                    emissiveRingKeyPieces += group.count;
                }
            }

            // Once a texture formed a full ring it stays dark, even on frames where only part of the ring is drawn.
            // Only textures with ring-sized pieces are remembered; untextured effects and small pieces are judged each
            // frame.
            for (size_t k = 0; k < fullRingKeyCount; k++) {
                const uint64_t key = ringKeys[k];
                auto group = std::find_if(groups.begin(), groups.end(), [&](const KeyGroup &entry) { return entry.key == key; });
                if ((key == 0) || (group == groups.end()) || group->untextured || ((group->extentSum / float(group->count)) < kEmissiveMinRingPieceExtent)) {
                    continue;
                }

                EmissiveVerdict &verdict = emissiveVerdicts[key];
                verdict.ring = true;
                verdict.lastSeenFrame = emissiveFrame;
            }
        }

        thread_local std::vector<Cluster> clusters;
        clusters.clear();
        emissiveVerdictRejections = 0;
        emissivePersistentRejections = 0;
        emissiveUnlistedPieces = 0;
        emissiveDisabledPieces = 0;
        emissiveLegacyPieces = 0;
        RtMaterialDatabase &emitterList = rtMaterialDatabase();
        for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
            emissiveVerdicts[candidate.key].lastSeenFrame = emissiveFrame;

            // Only draws matching an entry listed and enabled in rt_emitters.json give light. Others are recorded by
            // signature as candidates for the user to enable.
            RtEmitterSignature signature = candidate.signature;
            if (signature.signature == 0) {
                signature.kind = emitterKindOf(candidate);
                signature.texture = candidate.untextured ? 0 : candidate.key;
                signature.legacyKey = candidate.key;
                signature.triangles = candidate.triangleCount;
                signature.computeSignature();
            }

            RtEmitterParams params;
            bool applies = false;
            const bool listed = emitterList.matchEmitter(signature, rtScene.areaMap, rtScene.areaRoom, params, applies);
            if (!listed) {
                // The heuristic used before the list lit additive and untextured effects when bright, and sprites the
                // GPU found glowing.
                auto verdict = emissiveVerdicts.find(candidate.key);
                const bool sampledGlow = (verdict != emissiveVerdicts.end()) && (verdict->second.samples > 0) && (verdict->second.peakGlow >= kEmissiveVerdictMinGlow);
                const bool wouldGlow = candidate.untextured || (candidate.category == EmissiveCategory::Additive) || sampledGlow;
                emitterList.recordEmitterCandidate(signature, candidate.extent, rtScene.areaMap, rtScene.areaRoom, rtScene.areaName, wouldGlow);
                emissiveUnlistedPieces++;

                // Legacy glow: unlisted untextured polygon effects (the item pillar, hit sparks) still glow as they did
                // before, as far as the GPU finds them bright, scaled by their size so sparks stay a small flash.
                if (!(candidate.untextured && emitterList.legacyGlowUntextured())) {
                    continue;
                }

                params = RtEmitterParams();
                params.id = signature.signature;
                params.enabled = true;
                params.glowGate = true;
                params.sampleColor = true;
                params.intensity = std::clamp(candidate.extent / kEmissiveLegacyExtentPerIntensity, kEmissiveLegacyMinIntensity, kEmissiveLegacyMaxIntensity);
                params.maxLights = kEmissiveLegacyMaxLights;
                emissiveLegacyPieces++;
                applies = true;
            }

            if (!applies) {
                emissiveDisabledPieces++;
                continue;
            }

            // Thin pieces stay single lights so a row of segments never becomes one wide light.
            bool merged = false;
            for (Cluster &cluster : clusters) {
                if (candidate.thin || cluster.thin) {
                    continue;
                }

                // Pieces only gather into one light under the same rt_emitters.json entry, so each light has one
                // set of settings.
                if (cluster.params.id != params.id) {
                    continue;
                }

                const float dx = cluster.center[0] - candidate.center[0];
                const float dy = cluster.center[1] - candidate.center[1];
                const float dz = cluster.center[2] - candidate.center[2];
                const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                // Hit sparks fly apart from the hit, so their pieces gather over a wider distance into one flash.
                const bool sparks = candidate.untextured && cluster.untextured;
                const float mergeDistance = sparks ? std::max(cluster.extent + candidate.extent, kEmissiveSparkMergeRadius) : (cluster.extent + candidate.extent);
                if (distance <= mergeDistance) {
                    const float weight = std::max(candidate.extent, 1e-3f);
                    const float total = cluster.weight + weight;
                    for (int c = 0; c < 3; c++) {
                        cluster.center[c] = (cluster.center[c] * cluster.weight + candidate.center[c] * weight) / total;
                    }

                    cluster.extent = std::max(cluster.extent, distance * 0.5f + candidate.extent);
                    cluster.weight = total;
                    if (uint32_t(candidate.category) < uint32_t(cluster.category)) {
                        cluster.category = candidate.category;
                        cluster.renderIndex = candidate.renderIndex;
                        cluster.triangleCount = candidate.triangleCount;
                        cluster.key = candidate.key;
                    }
                    else if ((candidate.category == cluster.category) && (candidate.key != cluster.key) &&
                        ((emissiveKeyGlow(candidate.key) > emissiveKeyGlow(cluster.key)) ||
                        ((emissiveKeyGlow(candidate.key) == emissiveKeyGlow(cluster.key)) && ((candidate.triangleCount > cluster.triangleCount) ||
                        ((candidate.triangleCount == cluster.triangleCount) && (candidate.key < cluster.key))))))
                    {
                        cluster.renderIndex = candidate.renderIndex;
                        cluster.triangleCount = candidate.triangleCount;
                        cluster.key = candidate.key;
                    }
                    else if ((candidate.category == cluster.category) && (candidate.key == cluster.key) && (candidate.triangleCount > cluster.triangleCount)) {
                        cluster.renderIndex = candidate.renderIndex;
                        cluster.triangleCount = candidate.triangleCount;
                        cluster.key = candidate.key;
                    }

                    merged = true;
                    break;
                }
            }

            if (!merged) {
                Cluster cluster;
                for (int c = 0; c < 3; c++) {
                    cluster.center[c] = candidate.center[c];
                }

                cluster.extent = candidate.extent;
                cluster.weight = std::max(candidate.extent, 1e-3f);
                cluster.key = candidate.key;
                cluster.renderIndex = candidate.renderIndex;
                cluster.triangleCount = candidate.triangleCount;
                cluster.category = candidate.category;
                cluster.thin = candidate.thin;
                cluster.untextured = candidate.untextured;
                cluster.params = params;
                clusters.push_back(cluster);
            }
        }

        // Clusters that grew too wide are area effects such as ground rings, not point light sources.
        const size_t clustersBefore = clusters.size();
        clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [](const Cluster &cluster) { return cluster.extent > kEmissiveMaxClusterExtent; }), clusters.end());
        emissiveClustersTooLarge = uint32_t(clustersBefore - clusters.size());

        // An entry can limit how many lights it makes at once; its largest clusters are kept.
        {
            std::stable_sort(clusters.begin(), clusters.end(), [](const Cluster &a, const Cluster &b) { return a.extent > b.extent; });
            thread_local std::unordered_map<uint64_t, uint32_t> lightsPerEntry;
            lightsPerEntry.clear();
            clusters.erase(std::remove_if(clusters.begin(), clusters.end(), [&](const Cluster &cluster) {
                return (cluster.params.maxLights > 0) && (++lightsPerEntry[cluster.params.id] > cluster.params.maxLights);
            }), clusters.end());
        }

        emissiveClusterCount = uint32_t(clusters.size());

        // Fade speeds of an entry, or the defaults by kind.
        auto fadeRates = [&](const RtEmitterParams &params, EmissiveCategory category, bool untextured, float &fadeIn, float &fadeOut) {
            fadeIn = kEmissiveFadeIn;
            fadeOut = kEmissiveFadeOut;
            if (untextured) {
                fadeIn = kEmissiveSparkFadeIn;
                fadeOut = kEmissiveSparkFadeOut;
            }
            else if (category == EmissiveCategory::Additive) {
                fadeIn = 1.0f;
            }
            else if (category == EmissiveCategory::Blended) {
                fadeIn = kEmissiveEffectFadeIn;
            }

            fadeIn = (params.fadeInFrames > 0) ? (1.0f / float(params.fadeInFrames)) : fadeIn;
            fadeOut = (params.fadeOutFrames > 0) ? (1.0f / float(params.fadeOutFrames)) : fadeOut;
        };

        // Match against the tracked lights by position so each light keeps its slot and fade.
        for (TrackedEmissive &tracked : trackedEmissives) {
            tracked.seen = false;
        }

        thread_local std::vector<bool> clusterMatched;
        clusterMatched.assign(clusters.size(), false);
        for (TrackedEmissive &tracked : trackedEmissives) {
            int best = -1;
            float bestDistance = std::max(tracked.extent * 2.0f, 8.0f);
            for (size_t c = 0; c < clusters.size(); c++) {
                if (clusterMatched[c] || (clusters[c].params.id != tracked.params.id)) {
                    continue;
                }

                const float dx = tracked.position[0] - clusters[c].center[0];
                const float dy = tracked.position[1] - clusters[c].center[1];
                const float dz = tracked.position[2] - clusters[c].center[2];
                const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = int(c);
                }
            }

            if (best < 0) {
                continue;
            }

            const Cluster &cluster = clusters[best];
            clusterMatched[best] = true;
            for (int c = 0; c < 3; c++) {
                tracked.position[c] += (cluster.center[c] - tracked.position[c]) * kEmissivePositionSmoothing;
            }

            tracked.extent += (cluster.extent - tracked.extent) * kEmissivePositionSmoothing;
            // A different source took over this light: sample it from scratch instead of inheriting the old glow.
            tracked.keyChanged = (tracked.key != cluster.key);
            tracked.key = cluster.key;
            tracked.renderIndex = cluster.renderIndex;
            tracked.triangleCount = cluster.triangleCount;
            tracked.category = cluster.category;
            tracked.untextured = cluster.untextured;
            tracked.params = cluster.params;
            tracked.seen = true;
        }

        // Fade the tracked lights and drop the ones that faded out.
        // Additive bursts can last only a few frames, so they light up at once; hit sparks take two frames.
        for (TrackedEmissive &tracked : trackedEmissives) {
            tracked.fresh = tracked.seen && tracked.keyChanged;
            tracked.keyChanged = false;
            float fadeIn, fadeOut;
            fadeRates(tracked.params, tracked.category, tracked.untextured, fadeIn, fadeOut);
            tracked.fade = tracked.seen ? std::min(tracked.fade + fadeIn, 1.0f) : (tracked.fade - fadeOut);
        }

        trackedEmissives.erase(std::remove_if(trackedEmissives.begin(), trackedEmissives.end(), [](const TrackedEmissive &tracked) { return tracked.fade <= 0.0f; }), trackedEmissives.end());

        // New sources start a light in a free slot, nearest to the camera first.
        bool slotTaken[EmissiveSlots] = {};
        for (const TrackedEmissive &tracked : trackedEmissives) {
            slotTaken[tracked.slot] = true;
        }

        auto cameraDistance2 = [&](const float position[3]) {
            const float dx = position[0] - cameraPosition[0];
            const float dy = position[1] - cameraPosition[1];
            const float dz = position[2] - cameraPosition[2];
            return dx * dx + dy * dy + dz * dz;
        };

        thread_local std::vector<uint32_t> newClusters;
        newClusters.clear();
        for (uint32_t c = 0; c < clusters.size(); c++) {
            if (!clusterMatched[c]) {
                newClusters.push_back(c);
            }
        }

        std::sort(newClusters.begin(), newClusters.end(), [&](uint32_t a, uint32_t b) {
            return cameraDistance2(clusters[a].center) < cameraDistance2(clusters[b].center);
        });

        for (uint32_t c : newClusters) {
            if (trackedEmissives.size() >= MaxEmissiveLights) {
                break;
            }

            uint32_t slot = EmissiveSlots;
            for (uint32_t s = 0; s < EmissiveSlots; s++) {
                if (!slotTaken[s]) {
                    slot = s;
                    break;
                }
            }

            if (slot >= EmissiveSlots) {
                break;
            }

            const Cluster &cluster = clusters[c];
            TrackedEmissive tracked;
            for (int k = 0; k < 3; k++) {
                tracked.position[k] = cluster.center[k];
            }

            tracked.extent = cluster.extent;
            float fadeIn, fadeOut;
            fadeRates(cluster.params, cluster.category, cluster.untextured, fadeIn, fadeOut);
            tracked.fade = fadeIn;

            tracked.untextured = cluster.untextured;
            tracked.params = cluster.params;
            tracked.key = cluster.key;
            tracked.slot = slot;
            tracked.renderIndex = cluster.renderIndex;
            tracked.triangleCount = cluster.triangleCount;
            tracked.category = cluster.category;
            tracked.fresh = true;
            tracked.seen = true;
            slotTaken[slot] = true;
            trackedEmissives.push_back(tracked);
        }

        for (const TrackedEmissive &tracked : trackedEmissives) {
            // Intensity, color and reach come from the matching rt_emitters.json entry.
            const RtEmitterParams &params = tracked.params;
            const float intensity = params.intensity * tracked.fade;
            const float rangeScale = params.rangeScale;
            const float tint[3] = {
                params.sampleColor ? 1.0f : params.color[0],
                params.sampleColor ? 1.0f : params.color[1],
                params.sampleColor ? 1.0f : params.color[2]
            };

            interop::PointLight light = {};
            light.position = { tracked.position[0], tracked.position[1], tracked.position[2] };
            light.direction = { 0.0f, 0.0f, 0.0f };
            light.diffuseColor = { tint[0] * intensity, tint[1] * intensity, tint[2] * intensity };
            light.attenuationRadius = std::clamp(tracked.extent * kEmissiveRangePerExtent, kEmissiveMinRange, tracked.untextured ? kEmissiveSparkMaxRange : kEmissiveMaxRange) * rangeScale;
            light.pointRadius = std::max(tracked.extent, 1.0f);
            // The GPU samples the source sprite's material for the light's color: its draw and triangle count. A light
            // whose source wasn't drawn this frame keeps its last color while it fades out.
            light.flickerIntensity = float(tracked.renderIndex);
            light.shadowOffset = float(tracked.triangleCount);
            light.groupBits = tracked.slot | kEmissiveLightFlag | (tracked.fresh ? kEmissiveResetFlag : 0u) | (tracked.seen ? 0u : kEmissiveNoSampleFlag) |
                (uint32_t(tracked.category) << kEmissiveCategoryShift) | (params.sampleColor ? kEmissiveSampleColorFlag : 0u) | (params.glowGate ? kEmissiveGlowGateFlag : 0u);
            frameLights.push_back(light);
        }

        // Manual point lights of this area (rt_lights.json), after the tracked emissive lights (whose diagnostics pair
        // frameLights with trackedEmissives by index) and before the material probes.
        frameManualLightIds.clear();
        emitterList.pollEditedFiles();
        const std::shared_ptr<const std::vector<RtManualLight>> manual = emitterList.manualLightList();
        frameManualLights.clear();
        frameAreaManualLights = 0;
        static const auto start = std::chrono::steady_clock::now();
        const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
        const float fadeStep = std::clamp((seconds - manualLightsLastSeconds) / ManualLightFadeSeconds, 0.0f, 1.0f);
        manualLightsLastSeconds = seconds;
        if (manual && (rtScene.areaMap >= 0)) {
            // Every light of the area is ranked each frame by how much it can light near the camera: brighter and
            // wider lights win, closer ones more so. Lights whose reach ends too far from the camera to light anything
            // on screen are left out. The MaxManualLights best are shown; a light already shown ranks a little higher,
            // so two similar lights don't trade places from frame to frame. Lights fade in and out over
            // ManualLightFadeSeconds as they join and leave the set, and keep the file's order in the light list.
            struct Ranked {
                const RtManualLight *light;
                float score;
                float distance;
            };

            std::vector<Ranked> ranked;
            for (const RtManualLight &manualLight : *manual) {
                if (!manualLight.enabled || (manualLight.map != rtScene.areaMap) || ((manualLight.room >= 0) && (manualLight.room != rtScene.areaRoom))) {
                    continue;
                }

                frameAreaManualLights++;
                if (ranked.size() >= MaxAreaManualLights) {
                    continue;
                }

                float distanceSquared = 0.0f;
                for (int c = 0; c < 3; c++) {
                    const float delta = manualLight.position[c] - cameraPosition[c];
                    distanceSquared += delta * delta;
                }

                const float distance = std::sqrt(distanceSquared);
                if ((distance - manualLight.range) > ManualLightViewReach) {
                    continue;
                }

                const float brightness = manualLight.intensity * std::max(manualLight.color[0], std::max(manualLight.color[1], manualLight.color[2]));
                const float nearest = std::max(manualLight.range * 0.25f, 1.0f);
                float score = brightness * manualLight.range * manualLight.range / std::max(distanceSquared, nearest * nearest);
                auto state = manualLightFades.find(manualLight.id);
                if ((state != manualLightFades.end()) && state->second.selected) {
                    score *= ManualLightHysteresis;
                }

                ranked.push_back({ &manualLight, score, distance });
            }

            if ((frameAreaManualLights > MaxAreaManualLights) && !manualLightCapLogged) {
                rtDiagLog("manual lights: map %d room %d has %u lights; only the first %u are considered", rtScene.areaMap, rtScene.areaRoom, frameAreaManualLights, MaxAreaManualLights);
                manualLightCapLogged = true;
            }

            const size_t selectedCount = std::min(ranked.size(), size_t(MaxManualLights));
            std::partial_sort(ranked.begin(), ranked.begin() + selectedCount, ranked.end(), [](const Ranked &a, const Ranked &b) { return a.score > b.score; });
            for (auto &pair : manualLightFades) {
                pair.second.selected = false;
                pair.second.seen = false;
            }

            for (size_t i = 0; i < ranked.size(); i++) {
                ManualLightState &state = manualLightFades[ranked[i].light->id];
                state.selected = (i < selectedCount);
                state.seen = true;
                state.distance = ranked[i].distance;
                state.score = ranked[i].score;
            }

            // Lights no longer in the area (another room, removed from the file) are dropped at once.
            for (auto it = manualLightFades.begin(); it != manualLightFades.end();) {
                ManualLightState &state = it->second;
                state.fade = state.selected ? std::min(state.fade + fadeStep, 1.0f) : (state.fade - fadeStep);
                if (!state.seen || (state.fade <= 0.0f)) {
                    it = manualLightFades.erase(it);
                }
                else {
                    ++it;
                }
            }

            for (const RtManualLight &manualLight : *manual) {
                auto stateIt = manualLightFades.find(manualLight.id);
                if ((stateIt == manualLightFades.end()) || !manualLight.enabled || (manualLight.map != rtScene.areaMap) ||
                    ((manualLight.room >= 0) && (manualLight.room != rtScene.areaRoom)))
                {
                    continue;
                }

                const ManualLightState &state = stateIt->second;
                // Flicker on the intensity: three waves at unrelated speeds, offset per light.
                float flicker = 1.0f;
                if (manualLight.flicker > 0.0f) {
                    const float t = seconds * manualLight.flickerSpeed;
                    const float p = manualLight.flickerPhase;
                    const float TwoPi = 6.2831853f;
                    const float wave = 0.5f * std::sin(TwoPi * 1.7f * t + p) + 0.3f * std::sin(TwoPi * 4.3f * t + 2.1f * p) + 0.2f * std::sin(TwoPi * 9.1f * t + 3.7f * p);
                    flicker = 1.0f - manualLight.flicker * 0.5f * (1.0f - wave);
                }

                const float scale = manualLight.intensity * flicker * state.fade;
                interop::PointLight light = {};
                light.position = { manualLight.position[0], manualLight.position[1], manualLight.position[2] };
                light.direction = { 0.0f, 0.0f, 0.0f };
                light.diffuseColor = { manualLight.color[0] * scale, manualLight.color[1] * scale, manualLight.color[2] * scale };
                light.specularColor = light.diffuseColor;
                light.attenuationRadius = manualLight.range;
                light.pointRadius = std::max(manualLight.sourceRadius, 0.5f);
                light.flickerIntensity = 0.0f;
                light.shadowOffset = 0.0f;
                light.groupBits = 0xFFu | kEmissiveLightFlag | kEmissiveNoSampleFlag | (manualLight.castsShadows ? 0u : kManualNoShadowFlag);
                frameLights.push_back(light);
                frameManualLightIds.push_back(manualLight.id);
                frameManualLights.push_back({ manualLight.id, state.distance, state.fade, state.score, state.selected });
            }
        }
        else {
            manualLightFades.clear();
        }

        // Lava lights: the emissive cells nearest the camera, each a light a little above the surface, colored by the
        // material's average texture color times its emission and reaching further from larger cells.
        frameLavaLightCount = 0;
        if (!rtScene.emissiveCells.empty() || !lavaLights.empty()) {
            const float FadeIn = 0.15f;
            const float FadeOut = 0.1f;
            const float LightHeight = 12.0f;
            struct Candidate {
                int64_t cell;
                float position[3];
                float radiance[3];
                float area;
                float distance;
            };

            std::vector<Candidate> candidates;
            candidates.reserve(rtScene.emissiveCells.size());
            for (const auto &pair : rtScene.emissiveCells) {
                const RaytracingScene::EmissiveCell &cell = pair.second;
                if (cell.area <= 1e-3f) {
                    continue;
                }

                Candidate candidate;
                candidate.cell = pair.first;
                candidate.area = cell.area;
                const float normalLength = std::sqrt(cell.normal[0] * cell.normal[0] + cell.normal[1] * cell.normal[1] + cell.normal[2] * cell.normal[2]);
                float distanceSquared = 0.0f;
                for (int c = 0; c < 3; c++) {
                    const float normal = (normalLength > 1e-6f) ? (cell.normal[c] / normalLength) : ((c == 1) ? 1.0f : 0.0f);
                    candidate.position[c] = cell.position[c] / cell.area + normal * LightHeight;
                    candidate.radiance[c] = cell.radiance[c] / cell.area;
                    const float delta = candidate.position[c] - cameraPosition[c];
                    distanceSquared += delta * delta;
                }

                candidate.distance = distanceSquared;
                candidates.push_back(candidate);
            }

            const size_t selectedCount = std::min(candidates.size(), size_t(MaxLavaLights));
            std::partial_sort(candidates.begin(), candidates.begin() + selectedCount, candidates.end(), [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
            for (LavaLight &light : lavaLights) {
                light.selected = false;
            }

            for (size_t i = 0; i < selectedCount; i++) {
                const Candidate &candidate = candidates[i];
                auto it = std::find_if(lavaLights.begin(), lavaLights.end(), [&](const LavaLight &light) { return light.cell == candidate.cell; });
                if (it == lavaLights.end()) {
                    lavaLights.emplace_back();
                    it = lavaLights.end() - 1;
                    it->cell = candidate.cell;
                }

                it->selected = true;
                const float size = std::sqrt(candidate.area);
                for (int c = 0; c < 3; c++) {
                    it->position[c] = candidate.position[c];
                    it->radiance[c] = candidate.radiance[c];
                }

                it->range = std::clamp(60.0f + size * 1.5f, 80.0f, 320.0f);
                it->radius = std::clamp(size * 0.4f, 4.0f, 32.0f);
            }

            for (LavaLight &light : lavaLights) {
                light.fade = light.selected ? std::min(light.fade + FadeIn, 1.0f) : (light.fade - FadeOut);
            }

            lavaLights.erase(std::remove_if(lavaLights.begin(), lavaLights.end(), [](const LavaLight &light) { return light.fade <= 0.0f; }), lavaLights.end());
            for (const LavaLight &lava : lavaLights) {
                interop::PointLight light = {};
                light.position = { lava.position[0], lava.position[1], lava.position[2] };
                light.direction = { 0.0f, 0.0f, 0.0f };
                light.diffuseColor = { lava.radiance[0] * lava.fade, lava.radiance[1] * lava.fade, lava.radiance[2] * lava.fade };
                light.specularColor = light.diffuseColor;
                light.attenuationRadius = lava.range;
                light.pointRadius = lava.radius;
                light.flickerIntensity = 0.0f;
                light.shadowOffset = 0.0f;
                light.groupBits = 0xFFu | kEmissiveLightFlag | kEmissiveNoSampleFlag;
                frameLights.push_back(light);
                frameLavaLightCount++;
            }
        }

        rtParams.emissiveLightCount = uint32_t(frameLights.size()) - rtParams.emissiveLightStart;

        // Which key each slot is sampled for this frame, for the glow readback.
        memset(emissiveSlotSampled, 0, sizeof(emissiveSlotSampled));
        for (const TrackedEmissive &tracked : trackedEmissives) {
            if (tracked.slot < EmissiveSlots) {
                emissiveSlotKeys[tracked.slot] = tracked.key;
                emissiveSlotSampled[tracked.slot] = tracked.seen;
            }
        }

        emissiveFrame++;
    }

    void RaytracingResources::updateLightsBuffer(RenderWorker *worker) {
        const uint32_t count = uint32_t(frameLights.size());
        rtParams.lightsCount = count;
        interop::PointLight fallback = {};
        const void *data = (count > 0) ? static_cast<const void *>(frameLights.data()) : static_cast<const void *>(&fallback);
        const uint64_t bytes = sizeof(interop::PointLight) * std::max(count, 1u);
        growBuffer(device, lightsBuffer, lightsCapacity, bytes, RenderBufferFlag::STORAGE);
        uploadToBuffer(worker, lightsUpload, lightsUploadCapacity, lightsBuffer.get(), data, bytes);
    }

    // Diagnostics

    // Counters, hit color and hit velocity buffers, then the colors and glow of the first 16 emissive light slots.
    static const uint64_t kDiagEmissiveBytes = RaytracingResources::EmissiveSlots * sizeof(float) * 4;
    static const uint64_t kDiagReadbackBytes = kDiagCounterBytes + kHitBufferBytes * 2 + kDiagEmissiveBytes * 2;

    static std::atomic<bool> diagnosticsCaptureRequested(false);

    static std::atomic<float> diagnosticsPixelX(-1.0f);
    static std::atomic<float> diagnosticsPixelY(-1.0f);

    void requestRaytracingDiagnosticsCapture() {
        diagnosticsCaptureRequested.store(true);
    }

    void setRaytracingDiagnosticsPixel(float x, float y) {
        diagnosticsPixelX.store(x);
        diagnosticsPixelY.store(y);
    }

    void raytracingDiagnosticsPixel(float &x, float &y) {
        x = diagnosticsPixelX.load();
        y = diagnosticsPixelY.load();
    }

    bool raytracingDiagnosticsCaptureRequested() {
        return diagnosticsCaptureRequested.load();
    }

    bool consumeRaytracingDiagnosticsCapture() {
        return diagnosticsCaptureRequested.exchange(false);
    }

    // Debug views of the ray traced lighting, cycled with F10 in game.
    static const char *kDebugViewNames[] = {
        "off (normal image)",
        "face normal (geometric, as RGB)",
        "shading normal (smoothed, as RGB)",
        "normal disagreement (red where face and shading normals differ by more than about 72 degrees)",
        "sun visibility (white lit, black shadowed)",
        "direct light factor",
        "draw id (each draw its own color)",
        "lit from a solid face pointing away from the sun (red) / two-sided (blue)",
        "decal resolve (green: decal shown; decal found but not taken: yellow too far from the surface, red drawn before a character or small object, blue not parallel, magenta hidden by the surface in front; grey: no decal)"
    };

    static std::atomic<uint32_t> debugViewMode(0);

    void cycleRaytracingDebugView() {
        const uint32_t count = uint32_t(std::size(kDebugViewNames));
        const uint32_t mode = (debugViewMode.load() + 1) % count;
        debugViewMode.store(mode);
        rtDiagLog("debug view %u: %s (F10 cycles)", mode, kDebugViewNames[mode]);
    }

    uint32_t raytracingDebugView() {
        return debugViewMode.load();
    }

    // Diffuse bounces of the indirect light. The game always sets 1; the second bounce stays in the shaders, unused.
    static std::atomic<uint32_t> indirectBounceCount(1);

    void setRaytracingIndirectBounces(uint32_t bounces) {
        indirectBounceCount.store(std::clamp(bounces, 1u, 2u));
    }

    void toggleRaytracingSecondBounce() {
        const uint32_t bounces = (indirectBounceCount.load() >= 2) ? 1u : 2u;
        indirectBounceCount.store(bounces);
        rtDiagLog("indirect light: %u bounce%s", bounces, (bounces == 1) ? "" : "s");
    }

    uint32_t raytracingIndirectBounces() {
        return indirectBounceCount.load();
    }

    // Graphics menu "Advanced Water Rendering": blended water surfaces are ray traced (reflections, sun glint, shadows)
    // instead of rasterized after the path traced pass.
    static std::atomic<bool> advancedWater(true);

    void setRaytracingAdvancedWater(bool enabled) {
        advancedWater.store(enabled);
    }

    bool raytracingAdvancedWater() {
        return advancedWater.load();
    }

    static std::atomic<bool> volumetricFogEnabled(true);

    void setRaytracingVolumetricFog(bool enabled) {
        volumetricFogEnabled.store(enabled);
    }

    bool raytracingVolumetricFog() {
        return volumetricFogEnabled.load();
    }

    static std::atomic<bool> godRaysEnabled(true);

    void setRaytracingGodRays(bool enabled) {
        godRaysEnabled.store(enabled);
    }

    bool raytracingGodRays() {
        return godRaysEnabled.load();
    }

    void rtDiagLog(const char *fmt, ...) {
#   if defined(_WIN32)
        const char *localAppData = std::getenv("LOCALAPPDATA");
        if (localAppData == nullptr) {
            return;
        }

        // rt.log lives with the game's own settings in %LOCALAPPDATA%\EltaleReborn-PC. The log from the old
        // %LOCALAPPDATA%\Quest64Recompiled folder is copied over once (and kept there).
        static std::once_flag migrated;
        std::call_once(migrated, [localAppData]() {
            std::error_code error;
            const std::filesystem::path directory = std::filesystem::path(localAppData) / "EltaleReborn-PC";
            std::filesystem::create_directories(directory, error);
            const std::filesystem::path oldLog = std::filesystem::path(localAppData) / "Quest64Recompiled" / "rt.log";
            if (!std::filesystem::exists(directory / "rt.log", error) && std::filesystem::exists(oldLog, error)) {
                std::filesystem::copy_file(oldLog, directory / "rt.log", std::filesystem::copy_options::skip_existing, error);
            }
        });

        char path[512];
        std::snprintf(path, sizeof(path), "%s\\EltaleReborn-PC\\rt.log", localAppData);
        FILE *file = nullptr;
        if ((fopen_s(&file, path, "a") != 0) || (file == nullptr)) {
            return;
        }

        std::fputs("[rt] ", file);
        va_list args;
        va_start(args, fmt);
        std::vfprintf(file, fmt, args);
        va_end(args);
        std::fputc('\n', file);
        std::fclose(file);
#   else
        (void)fmt;
#   endif
    }

    // Row-vector convention: out = v * m. This is the convention of every matrix RT64 keeps on the CPU,
    // and it is what mul(M, v) computes in HLSL once the matrix is uploaded to a column major cbuffer.
    static void mulRowVector(const interop::float4x4 &m, const float v[4], float out[4]) {
        for (int j = 0; j < 4; j++) {
            out[j] = v[0] * float(m.m[0][j]) + v[1] * float(m.m[1][j]) + v[2] * float(m.m[2][j]) + v[3] * float(m.m[3][j]);
        }
    }

    static void logMatrix(const char *name, const interop::float4x4 &m) {
        float e[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                e[i][j] = float(m.m[i][j]);
            }
        }
        rtDiagLog("  %s rows: [%g %g %g %g] [%g %g %g %g] [%g %g %g %g] [%g %g %g %g]", name,
            e[0][0], e[0][1], e[0][2], e[0][3],
            e[1][0], e[1][1], e[1][2], e[1][3],
            e[2][0], e[2][1], e[2][2], e[2][3],
            e[3][0], e[3][1], e[3][2], e[3][3]);
    }

    void RaytracingResources::logSceneDiagnostics(const RaytracingScene &rtScene) {
        uint32_t nonOpaque = 0;
        for (const RaytracingInstance &instance : rtScene.instances) {
            nonOpaque += instance.alphaTested ? 1 : 0;
        }

        rtDiagLog("=== Ray tracing frame %u (path traced materials) ===", diagFrame);
        rtDiagLog("  path traced draws %zu (alpha tested %u), not path traced (no depth test+write) %u, rasterized after the path traced pass %u (of which opaque %u)",
            rtScene.instances.size(), nonOpaque, rtScene.skippedDraws, rtScene.rasterAfterDraws, rtScene.rasterOnlyOpaqueDraws);
        rtDiagLog("  path traced draw kinds: backface culled %u, 2-cycle %u, texture LOD %u, uses texture1 %u, untextured %u, flat shaded %u, no texture perspective %u",
            rtScene.drawsCulling, rtScene.drawsTwoCycle, rtScene.drawsTextureLOD, rtScene.drawsTexture1, rtScene.drawsUntextured, rtScene.drawsFlatShade, rtScene.drawsTexturePerspectiveOff);
        rtDiagLog("  draws kept rasterized because their world positions don't match where the raster drew them: %u", rtScene.drawsWorldMismatch);
        for (const RaytracingScene::WorldMismatch &mismatch : rtScene.worldMismatchSamples) {
            rtDiagLog("    world mismatch: draw %u, texture %016llX, vertex address %08X, %u triangles, largest error %.3f (relative to depth)", mismatch.renderIndex,
                (unsigned long long)mismatch.texture, mismatch.vertexAddress, mismatch.triangles, mismatch.error);
        }
        rtDiagLog("  projections rasterized because their camera differs from the scene's: %u; translucent depth mode draws rasterized after: %u",
            rtScene.incompatibleProjections, rtScene.skippedLayeredDraws);
        const uint32_t decalsExplained = rtScene.decalsWorldMismatch + rtScene.decalsAdditive + rtScene.decalsFallback;
        const uint32_t decalsOtherwise = (rtScene.decalsRasterAfter > decalsExplained) ? (rtScene.decalsRasterAfter - decalsExplained) : 0;
        rtDiagLog("  decals (decal depth mode: paths, windows, doors, wall details): %u path traced, lit like the surface under them; %u still rasterized after (world positions don't match %u, additive %u, fallback after three decal checks saw them mostly missed %u, other camera or outside the scene %u)",
            rtScene.decalsTraced, rtScene.decalsRasterAfter, rtScene.decalsWorldMismatch, rtScene.decalsAdditive, rtScene.decalsFallback, decalsOtherwise);
        // The largest path traced decals get their camera hits counted by this capture (see readDiagnosticBuffers).
        assignDecalSlots(rtScene, false);
        for (const RaytracingScene::DecalInfo &info : rtScene.decalInfos) {
            rtDiagLog("    decal draw %u: texture %016llX, %u triangles, span %.0f, %s%s, combiner %08X %08X, prim alpha %.2f, env alpha %.2f; in the acceleration structure with mask 0x%X (%s), %s",
                info.renderIndex, (unsigned long long)info.key, info.triangles, info.extent, info.blended ? "alpha blended (blended over the surface below by its alpha)" : "opaque blend",
                info.alphaTested ? ", texels tested by alpha" : "", info.combiner[0], info.combiner[1], info.primAlpha, info.envAlpha, info.queryMask,
                (info.queryMask == RtDecalRayMask) ? "the camera ray's decal query only" : "camera rays", info.culling ? "back faces culled like the raster" : "no culling");
        }

        rtDiagLog("  foliage: %u path traced draws shaded as foliage (camera facing billboards with a shadow proxy, upright cutouts without N64 lighting, or class foliage in rt_materials.json); %u billboard proxies",
            rtScene.foliageDraws, rtScene.billboardProxies);

        // Emissive materials (rt_materials.json "emission", class lava) and the lights they cast.
        uint32_t emissiveTriangles = 0;
        for (const RaytracingScene::EmissiveDraw &draw : rtScene.emissiveDraws) {
            emissiveTriangles += draw.triangles;
        }

        rtDiagLog("  emissive materials: %zu draws (%u triangles) in %zu cells of %g units; %u lava lights (max %u)%s", rtScene.emissiveDraws.size(), emissiveTriangles,
            rtScene.emissiveCells.size(), 96.0, frameLavaLightCount, MaxLavaLights, (rtParams.emissionOffset != 0) ? "" : "; no emissive draws, so the shaders skip emission");
        for (size_t d = 0; d < std::min(rtScene.emissiveDraws.size(), size_t(16)); d++) {
            const RaytracingScene::EmissiveDraw &draw = rtScene.emissiveDraws[d];
            rtDiagLog("    emissive draw %u: texture %016llX, %u triangles, light color %.2f %.2f %.2f", draw.renderIndex, (unsigned long long)draw.key, draw.triangles,
                draw.radiance[0], draw.radiance[1], draw.radiance[2]);
        }

        for (const LavaLight &lava : lavaLights) {
            rtDiagLog("    lava light at %.0f %.0f %.0f: color %.2f %.2f %.2f, range %.0f, radius %.0f, fade %.2f%s", lava.position[0], lava.position[1], lava.position[2],
                lava.radiance[0], lava.radiance[1], lava.radiance[2], lava.range, lava.radius, lava.fade, lava.selected ? "" : " (fading out)");
        }

        if (!raytracingAdvancedWater()) {
            rtDiagLog("  water: Advanced Water Rendering is off (Graphics menu); water stays rasterized after the path traced pass");
        }
        else {
            rtDiagLog("  water: %zu draws shaded by the water pass (blended ones are also counted in the path traced draws above); %u water-like draws left alone",
                rtScene.waterDraws.size(), rtScene.waterRejected);
            for (const RaytracingScene::WaterDraw &water : rtScene.waterDraws) {
                rtDiagLog("    water draw %u: texture %016llX, %s, %u triangles, span %.0f, %.0f%% facing up", water.renderIndex, (unsigned long long)water.key,
                    water.source, water.triangles, water.extent, water.upFacing * 100.0f);
            }
            for (const RaytracingScene::WaterReject &reject : rtScene.waterRejects) {
                rtDiagLog("    not water, draw %u: texture %016llX, span %.0f: %s (to make it water, set its class to water in rt_materials.html)", reject.renderIndex,
                    (unsigned long long)reject.key, reject.extent, reject.reason);
            }
        }
        rtDiagLog("  billboard sprites detected: cylindrical %u, spherical %u; light facing shadow proxies built for %u draws",
            rtScene.billboardsCylindrical, rtScene.billboardsSpherical, rtScene.billboardProxies);
        rtDiagLog("  pixel spread angle %g, cull winding sign %g", rtParams.pixelSpreadAngle, rtParams.cullWindingSign);

        uint32_t candidatesByCategory[3] = {};
        for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
            candidatesByCategory[uint32_t(candidate.category)]++;
        }

        const EmissiveRejections &rejected = rtScene.emissiveRejections;
        rtDiagLog("  emissive: candidate draws %u, candidate pieces %zu (additive %u, blended %u, cutout %u), clusters %u, lights %u (max %u), scene lights %u",
            rtScene.emissiveCandidateDraws, rtScene.emissiveCandidates.size(), candidatesByCategory[0], candidatesByCategory[1], candidatesByCategory[2],
            emissiveClusterCount, rtParams.emissiveLightCount, MaxEmissiveLights, rtParams.emissiveLightStart);
        rtDiagLog("  emissive sources: untextured effect draws %u (hit sparks), additive draws that write depth %u (fireballs); shadow-only casters outside the path traced scene %u (spell rocks)",
            rtScene.emissiveUntexturedAdditiveDraws, rtScene.emissiveDepthWritingAdditiveDraws, rtScene.shadowOnlyCasters);
        rtDiagLog("  emissive rejections: (no longer used) untextured %u, thin line pieces %u, ring of thin pieces %u (%u pieces), ground-flat pieces %u, pieces too large %u, pieces too small %u, clusters too wide %u (dark, dust-like and non-flame colors are rejected on the GPU, see the slot lines)",
            rejected.untexturedDraws, rejected.thinLinePieces, emissiveRings, emissiveRingPieces, rejected.groundFlat, rejected.tooLarge, rejected.tooSmall, emissiveClustersTooLarge);
        for (size_t i = 0; i < rtScene.emissiveDrawInfos.size(); i++) {
            const EmissiveDrawInfo &info = rtScene.emissiveDrawInfos[i];
            rtDiagLog("    candidate draw %zu: key %016llX (0 when untextured), %s, %u triangles, %u pieces (%u thin), largest aspect %.1f, largest extent %.1f, %u pieces kept",
                i, (unsigned long long)info.key, emissiveCategoryName(info.category), info.triangleCount, info.pieces, info.thinPieces, info.maxAspect, info.maxExtent, info.accepted);
        }
        uint32_t rejectedKeys = 0;
        uint32_t ringMarkedKeys = 0;
        uint32_t sceneryKeys = 0;
        for (const auto &entry : emissiveVerdicts) {
            rejectedKeys += emissiveKeyRejected(entry.first) ? 1 : 0;
            ringMarkedKeys += entry.second.ring ? 1 : 0;
            sceneryKeys += entry.second.persistent ? 1 : 0;
        }

        rtDiagLog("  emissive scenery: %u keys judged persistent cool colored scenery (water, mist), %u candidate pieces skipped for it this frame; %u lit cutout draws skipped as model parts",
            sceneryKeys, emissivePersistentRejections, rtScene.emissiveRejections.litCutouts);

        rtDiagLog("  emissive verdicts: %zu keys known, %u currently judged non-emissive, %u marked as ring textures; candidate pieces skipped as non-emissive %u; ring textures this frame %u (%u pieces)",
            emissiveVerdicts.size(), rejectedKeys, ringMarkedKeys, emissiveVerdictRejections, emissiveRingKeys, emissiveRingKeyPieces);
        {
            uint32_t classCounts[size_t(RtMaterialClass::Count)];
            uint32_t userEntries = 0;
            rtMaterialDatabase().classCounts(classCounts, userEntries);
            std::string countsText;
            for (size_t c = 0; c < size_t(RtMaterialClass::Count); c++) {
                char part[48];
                std::snprintf(part, sizeof(part), "%s%s %u", countsText.empty() ? "" : ", ", rtMaterialClassName(RtMaterialClass(c)), classCounts[c]);
                countsText += part;
            }

            uint32_t frameTextured = 0;
            for (uint64_t key : rtScene.materialKeys) {
                frameTextured += (key != 0) ? 1 : 0;
            }

            rtDiagLog("  materials: %zu entries (%u from the user) in %s; by class: %s; %u guesses this session, %zu probes this frame, %u textured path traced draws",
                rtMaterialDatabase().size(), userEntries, rtMaterialDatabase().path().c_str(), countsText.c_str(), materialGuessesThisSession, rtScene.materialProbes.size(), frameTextured);
        }

        rtDiagLog("  lights (emits if: additive = bright; blended = bright and tinted; cutout = bright, saturated, orange to yellow):");
        for (uint32_t i = 0; (i < rtParams.emissiveLightCount) && (i < trackedEmissives.size()); i++) {
            const TrackedEmissive &tracked = trackedEmissives[i];
            const interop::PointLight &light = frameLights[rtParams.emissiveLightStart + i];
            auto verdict = emissiveVerdicts.find(tracked.key);
            const float verdictGlow = (verdict != emissiveVerdicts.end()) ? verdict->second.glow : -1.0f;
            rtDiagLog("    light %u: slot %u, key %016llX, %s, fade %.2f%s, at %.0f %.0f %.0f, range %.0f, size %.1f, intensity %.2f, draw %u (%u tris), known glow %.2f",
                i, tracked.slot, (unsigned long long)tracked.key, emissiveCategoryName(tracked.category), tracked.fade, tracked.seen ? "" : " (fading)",
                light.position.x, light.position.y, light.position.z, light.attenuationRadius, light.pointRadius,
                std::max(light.diffuseColor.x, std::max(light.diffuseColor.y, light.diffuseColor.z)), tracked.renderIndex, tracked.triangleCount, verdictGlow);
        }

        // Every candidate draw of the frame by signature, and whether rt_emitters.json lets it give light.
        struct KeyStats {
            RtEmitterSignature signature;
            uint32_t pieces;
            float maxExtent;
        };

        std::vector<KeyStats> keyStats;
        for (const EmissiveCandidate &candidate : rtScene.emissiveCandidates) {
            RtEmitterSignature signature = candidate.signature;
            if (signature.signature == 0) {
                signature.kind = emitterKindOf(candidate);
                signature.texture = candidate.untextured ? 0 : candidate.key;
                signature.legacyKey = candidate.key;
                signature.triangles = candidate.triangleCount;
                signature.computeSignature();
            }

            auto it = std::find_if(keyStats.begin(), keyStats.end(), [&](const KeyStats &stats) { return stats.signature.signature == signature.signature; });
            if (it == keyStats.end()) {
                keyStats.push_back({ signature, 1, candidate.extent });
            }
            else {
                it->pieces++;
                it->maxExtent = std::max(it->maxExtent, candidate.extent);
            }
        }

        uint32_t listedEmitters = 0, enabledEmitters = 0, candidateEmitters = 0;
        rtMaterialDatabase().emitterCounts(listedEmitters, enabledEmitters, candidateEmitters);
        rtDiagLog("  emitters (%s): %u listed, %u of them enabled, %u unlisted candidates recorded; this frame %u candidate pieces unlisted (%u of them lit by the legacy glow for untextured effects, which is %s), %u listed but disabled or filtered out; area %s map %d room %d",
            rtMaterialDatabase().path().empty() ? "rt_emitters.json not loaded yet" : "rt_emitters.json", listedEmitters, enabledEmitters, candidateEmitters,
            emissiveUnlistedPieces, emissiveLegacyPieces, rtMaterialDatabase().legacyGlowUntextured() ? "on" : "off", emissiveDisabledPieces, rtScene.areaName, rtScene.areaMap, rtScene.areaRoom);
        {
            const std::shared_ptr<const std::vector<RtManualLight>> manual = rtMaterialDatabase().manualLightList();
            const size_t loadedLights = manual ? manual->size() : 0;
            const size_t enabledLights = manual ? size_t(std::count_if(manual->begin(), manual->end(), [](const RtManualLight &light) { return light.enabled; })) : 0;
            uint32_t shown = 0;
            for (const ManualLightInfo &info : frameManualLights) {
                shown += info.selected ? 1 : 0;
            }

            rtDiagLog("  manual lights: %zu loaded (%zu enabled) from %s; map %d room %d has %u, the nearest %u (most important for the camera) are shown at once: %u shown, %zu in the light list with the ones fading out",
                loadedLights, enabledLights, rtMaterialDatabase().manualLightsPath().c_str(), rtScene.areaMap, rtScene.areaRoom, frameAreaManualLights, MaxManualLights, shown,
                frameManualLights.size());
            for (const ManualLightInfo &info : frameManualLights) {
                rtDiagLog("    manual light %s: %.0f units from the camera, fade %.2f, rank score %.1f%s", info.id.c_str(), info.distance, info.fade, info.score,
                    info.selected ? "" : " (fading out)");
            }
            rtDiagLog("  room settings: sun and moon %s for map %d room %d (from %s, %s)", (rtParams.sunDisabled != 0) ? "OFF" : "ON", rtScene.areaMap, rtScene.areaRoom,
                sunSettingSource, sunSettingScope);
            rtDiagLog("  volumetric fog (%s, from %s, %s, clock %.2f h): density %.5f, color (%.2f, %.2f, %.2f), ambient %.2f, sun %.2f, lights %.2f, shadows %.2f, anisotropy %.2f, height falloff %.4f above %.1f, distance %.0f, game fog x%.2f, %u steps",
                raytracingVolumetricFog() ? "on in the Graphics menu" : "OFF in the Graphics menu", fogSettingSource, fogSettingScope, lastFogHour, rtParams.fogDensity,
                rtParams.fogColorR, rtParams.fogColorG, rtParams.fogColorB, rtParams.fogAmbient, rtParams.fogSunScatter, rtParams.fogLightScatter, rtParams.fogShadows, rtParams.fogAnisotropy,
                rtParams.fogHeightFalloff, rtParams.fogBaseHeight, rtParams.fogDistance, rtParams.fogGameScale, rtParams.fogSteps);
            rtDiagLog("  god rays (%s): strength %.2f, length %.2f, color (%.2f, %.2f, %.2f), from %s, windows %.2f (%u window draws path traced), bias %.2f, %u facing groups",
                raytracingGodRays() ? "on in the Graphics menu" : "OFF in the Graphics menu", rtParams.raysStrength, rtParams.raysLength,
                rtParams.raysColorR, rtParams.raysColorG, rtParams.raysColorB, (rtParams.raysFromSun != 0) ? "the sun or moon and the sky" : "the room's ray direction",
                rtParams.raysWindows, lastWindowDraws, rtParams.raysWindowBias, rtParams.raysWindowGroups);
            const float groupNormals[4][3] = {
                { rtParams.raysWindowGroup0X, rtParams.raysWindowGroup0Y, rtParams.raysWindowGroup0Z },
                { rtParams.raysWindowGroup1X, rtParams.raysWindowGroup1Y, rtParams.raysWindowGroup1Z },
                { rtParams.raysWindowGroup2X, rtParams.raysWindowGroup2Y, rtParams.raysWindowGroup2Z },
                { rtParams.raysWindowGroup3X, rtParams.raysWindowGroup3Y, rtParams.raysWindowGroup3Z },
            };
            for (uint32_t g = 0; (g < rtParams.raysWindowGroups) && (g < 4); g++) {
                rtDiagLog("    windows facing out (%.2f, %.2f, %.2f)", groupNormals[g][0], groupNormals[g][1], groupNormals[g][2]);
            }
            if (gameFogEnd > 0.0f) {
                rtDiagLog("  game fog: from %.0f to %.0f units, color (%.2f, %.2f, %.2f), mul %.1f offset %.1f", gameFogStart, gameFogEnd,
                    rtScene.gameFogColor[0], rtScene.gameFogColor[1], rtScene.gameFogColor[2], rtScene.gameFogMul, rtScene.gameFogOffset);
            }
            else {
                rtDiagLog("  game fog: none on the path traced draws%s", rtScene.gameFogValid ? " (found, but its range couldn't be worked out)" : "");
            }
        }

        std::sort(keyStats.begin(), keyStats.end(), [](const KeyStats &a, const KeyStats &b) { return a.pieces > b.pieces; });
        for (const KeyStats &stats : keyStats) {
            const RtEmitterSignature &s = stats.signature;
            RtEmitterParams params;
            bool applies = false;
            const char *status = ((s.kind == RtEmitterKindUntextured) && rtMaterialDatabase().legacyGlowUntextured()) ?
                "NOT LISTED, lit by the legacy glow for untextured effects (list it in rt_emitters.html to tune it or keep it dark)" :
                "NOT LISTED (no light; enable it in rt_emitters.html)";
            std::string label;
            if (rtMaterialDatabase().matchEmitter(s, rtScene.areaMap, rtScene.areaRoom, params, applies)) {
                RtEmitterEntry entry;
                if (rtMaterialDatabase().findEmitter(params.id, entry)) {
                    label = entry.label;
                }

                status = applies ? "listed and enabled: gives light" : (params.enabled ? "listed, but its only kinds / only areas filter excludes it here" : "listed, DISABLED");
            }

            rtDiagLog("    candidate draw signature %016llX (%s): texture %016llX, vertex address %08X, combiner %08X %08X, other mode %08X %08X, prim %.2f %.2f %.2f %.2f, env %.2f %.2f %.2f %.2f, vertex color %.2f %.2f %.2f, %u triangles, %u pieces, largest extent %.1f: %s%s%s",
                (unsigned long long)s.signature, emitterKindName(s.kind), (unsigned long long)s.texture, s.vertexAddress, s.combiner[0], s.combiner[1], s.otherMode[0], s.otherMode[1],
                s.prim[0], s.prim[1], s.prim[2], s.prim[3], s.env[0], s.env[1], s.env[2], s.env[3], s.vertexColor[0], s.vertexColor[1], s.vertexColor[2], s.triangles,
                stats.pieces, stats.maxExtent, status, label.empty() ? "" : " - ", label.c_str());
        }
        rtDiagLog("  pending meshes %zu, BLAS built %zu, TLAS instances %u, TLAS %s",
            pendingMeshes.size(), bottomLevelASVector.size(),
            (topLevelAS != nullptr) ? topLevelBuildInfo.instanceCount : 0u, (topLevelAS != nullptr) ? "present" : "MISSING");
        rtDiagLog("  shadow tint %g %g %g, facing floor %g", float(rtParams.cameraW.x), float(rtParams.cameraW.y), float(rtParams.cameraW.z), float(rtParams.cameraW.w));

        uint64_t totalTriangles = 0;
        uint32_t logged = 0;
        for (size_t i = 0; i < rtScene.instances.size(); i++) {
            const RaytracingInstance &rt = rtScene.instances[i];
            const uint32_t drawIndex = (i < rtScene.instanceIndices.size()) ? rtScene.instanceIndices[i] : 0;
            if (rt.bottomLevelIndex >= pendingMeshes.size()) {
                rtDiagLog("  draw %u has mesh index %u out of range", drawIndex, rt.bottomLevelIndex);
                continue;
            }
            const RenderBottomLevelASMesh &mesh = pendingMeshes[rt.bottomLevelIndex];
            totalTriangles += mesh.indexCount / 3;
            if (logged < 4) {
                rtDiagLog("  instance draw %u mesh %u: indexStart %llu indexCount %u vertexCount %u stride %u mask 0x%X level %d",
                    drawIndex, rt.bottomLevelIndex, (unsigned long long)(mesh.indexBuffer.offset / sizeof(uint32_t)), mesh.indexCount, mesh.vertexCount,
                    mesh.vertexStride, rt.queryMask, (rt.bottomLevelIndex < pendingMeshLevels.size()) ? pendingMeshLevels[rt.bottomLevelIndex] : -2);
                logged++;
            }
        }
        rtDiagLog("  total triangles %llu", (unsigned long long)totalTriangles);

        rtDiagLog("  resolution %g x %g, viewport x %g y %g w %g h %g, viewport scale y %g, screen %d x %d",
            float(rtParams.resolution.x), float(rtParams.resolution.y), float(rtParams.viewport.x), float(rtParams.viewport.y), float(rtParams.viewport.z), float(rtParams.viewport.w),
            rtScene.viewportScaleY, rtScene.screenWidth, rtScene.screenHeight);
        rtDiagLog("  lights %u", rtParams.lightsCount);
        const PresetScene &preset = rtScene.presetScene;
        rtDiagLog("  tonemap exposure %g white %g black %g, min luminance %g, luminance range %g, ambient %g %g %g, eye light %g %g %g, GI strength %g",
            preset.tonemapExposure, preset.tonemapWhite, preset.tonemapBlack, preset.minLuminance, preset.luminanceRange,
            float(preset.ambientBaseColor.x), float(preset.ambientBaseColor.y), float(preset.ambientBaseColor.z),
            float(preset.eyeLightDiffuseColor.x), float(preset.eyeLightDiffuseColor.y), float(preset.eyeLightDiffuseColor.z), preset.giDiffuseStrength);
        if ((rtScene.pointLights != nullptr) && (rtScene.lightCount > 0)) {
            const interop::PointLight &light = rtScene.pointLights[0];
            rtDiagLog("  light 0 position %g %g %g, diffuse %g %g %g, radius %g, groups 0x%X",
                float(light.position.x), float(light.position.y), float(light.position.z), float(light.diffuseColor.x), float(light.diffuseColor.y), float(light.diffuseColor.z),
                float(light.attenuationRadius), uint32_t(light.groupBits));
        }

        logMatrix("view", rtParams.view);
        logMatrix("projection", rtParams.projection);
        logMatrix("viewProj", rtParams.viewProj);
        logMatrix("viewProj inverse", rtParams.viewI);

        // Same camera math as PrimaryRayGen.
        const float originClip[4] = { 0.0f, 0.0f, 1.0f, 0.0f };
        float originH[4];
        mulRowVector(rtParams.viewI, originClip, originH);
        float origin[3] = { 0.0f, 0.0f, 0.0f };
        if (std::fabs(originH[3]) > 1e-12f) {
            origin[0] = originH[0] / originH[3];
            origin[1] = originH[1] / originH[3];
            origin[2] = originH[2] / originH[3];
        }
        const float centerClip[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        float centerH[4];
        mulRowVector(rtParams.viewI, centerClip, centerH);
        float centerDir[3] = { 0.0f, 0.0f, 0.0f };
        if (std::fabs(centerH[3]) > 1e-12f) {
            for (int i = 0; i < 3; i++) {
                centerDir[i] = centerH[i] / centerH[3] - origin[i];
            }
        }
        const float centerLen = std::sqrt(centerDir[0] * centerDir[0] + centerDir[1] * centerDir[1] + centerDir[2] * centerDir[2]);
        if (centerLen > 0.0f) {
            for (int i = 0; i < 3; i++) {
                centerDir[i] /= centerLen;
            }
        }
        rtDiagLog("  CPU camera origin %g %g %g (homogeneous w %g), center ray direction %g %g %g",
            origin[0], origin[1], origin[2], originH[3], centerDir[0], centerDir[1], centerDir[2]);

        if (!rtScene.probeValid) {
            rtDiagLog("  probe vertex unavailable");
            return;
        }

        const float world[4] = { rtScene.probeWorld[0], rtScene.probeWorld[1], rtScene.probeWorld[2], 1.0f };
        float clip[4];
        mulRowVector(rtParams.viewProj, world, clip);
        rtDiagLog("  probe vertex %u world %g %g %g", rtScene.probeVertex, world[0], world[1], world[2]);
        rtDiagLog("  probe clip via viewProj %g %g %g %g, RSP clip %g %g %g %g",
            clip[0], clip[1], clip[2], clip[3], rtScene.probeClipRSP[0], rtScene.probeClipRSP[1], rtScene.probeClipRSP[2], rtScene.probeClipRSP[3]);
        if (std::fabs(clip[3]) > 1e-12f) {
            const float ndcX = clip[0] / clip[3];
            const float ndcY = clip[1] / clip[3];
            const float s = (rtScene.viewportScaleY >= 0.0f) ? 1.0f : -1.0f;
            const float px = rtParams.viewport.x + (ndcX * 0.5f + 0.5f) * rtParams.viewport.z;
            const float py = rtParams.viewport.y + (0.5f - ndcY / (2.0f * s)) * rtParams.viewport.w;
            rtDiagLog("  probe ndc %g %g %g, expected pixel %g %g", ndcX, ndcY, clip[2] / clip[3], px, py);
        }
        const float toProbe[3] = { world[0] - origin[0], world[1] - origin[1], world[2] - origin[2] };
        const float probeDist = std::sqrt(toProbe[0] * toProbe[0] + toProbe[1] * toProbe[1] + toProbe[2] * toProbe[2]);
        const float cosAngle = (probeDist > 0.0f) ? ((toProbe[0] * centerDir[0] + toProbe[1] * centerDir[1] + toProbe[2] * centerDir[2]) / probeDist) : 0.0f;
        rtDiagLog("  probe distance from camera %g, cos angle to center ray %g", probeDist, cosAngle);
    }

    void RaytracingResources::clearDiagnosticBuffers(RenderWorker *worker) {
        if ((hitInstanceIdBuffer == nullptr) || (hitColorBuffer == nullptr) || (hitVelocityDistanceBuffer == nullptr)) {
            return;
        }

        const uint64_t zeroBytes = std::max(kHitBufferBytes, kDiagCounterBytes);
        if (diagZeroUpload == nullptr) {
            diagZeroUpload = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(zeroBytes));
            void *mapped = diagZeroUpload->map();
            if (mapped != nullptr) {
                memset(mapped, 0, size_t(zeroBytes));
                diagZeroUpload->unmap();
            }
        }

        RenderBuffer *targets[] = { hitInstanceIdBuffer.get(), hitColorBuffer.get(), hitVelocityDistanceBuffer.get() };
        for (RenderBuffer *target : targets) {
            const uint64_t targetBytes = (target == hitInstanceIdBuffer.get()) ? kDiagCounterBytes : kHitBufferBytes;
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(target, RenderBufferAccess::WRITE));
            worker->commandList->copyBufferRegion(target->at(0), diagZeroUpload->at(0), targetBytes);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(target, RenderBufferAccess::WRITE));
        }
    }

    void RaytracingResources::copyDiagnosticBuffers(RenderWorker *worker) {
        if ((hitInstanceIdBuffer == nullptr) || (hitColorBuffer == nullptr) || (hitVelocityDistanceBuffer == nullptr) || (hitNormalFogBuffer == nullptr)) {
            return;
        }

        if (diagReadback == nullptr) {
            diagReadback = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(kDiagReadbackBytes));
        }

        diagMaterialKeys = frameMaterialKeys;
        diagDrawSignatures = frameDrawSignatures;
        RenderBuffer *sources[] = { hitInstanceIdBuffer.get(), hitColorBuffer.get(), hitVelocityDistanceBuffer.get(), hitNormalFogBuffer.get() };
        const uint64_t offsets[] = { 0, kDiagCounterBytes, kDiagCounterBytes + kHitBufferBytes, kDiagCounterBytes + kHitBufferBytes * 2 };
        for (uint32_t i = 0; i < 4; i++) {
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(sources[i], RenderBufferAccess::READ));
            if (sources[i] == hitNormalFogBuffer.get()) {
                // Colors, then glow, of every emissive slot.
                worker->commandList->copyBufferRegion(diagReadback->at(offsets[i]), sources[i]->at(0), kDiagEmissiveBytes * 2);
            }
            else {
                worker->commandList->copyBufferRegion(diagReadback->at(offsets[i]), sources[i]->at(0), (i == 0) ? kDiagCounterBytes : kHitBufferBytes);
            }

            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(sources[i], RenderBufferAccess::WRITE));
        }

        diagReadbackDelay = 8;
    }

    static const uint64_t kDiagTextureSlotBytes = 512;
    static const uint32_t kDiagTextureSlots = 8;
    static const char *kDiagTextureNames[kDiagTextureSlots] = {
        "albedo from material (alpha = ray hit)", "direct light raw", "indirect light raw", "direct light filtered",
        "indirect light filtered", "composed output", "luma average (expect 0.5)", "final color target"
    };

    void RaytracingResources::copyDiagnosticTextures(RenderWorker *worker, RenderTexture *finalColor, RenderFormat finalColorFormat, RenderTextureLayout finalColorLayout, uint32_t finalX, uint32_t finalY) {
        if (diffuseTexture == nullptr) {
            return;
        }

        if (diagTextureReadback == nullptr) {
            diagTextureReadback = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(kDiagTextureSlotBytes * kDiagTextureSlots));
        }

        const int current = swapBuffers ? 1 : 0;
        struct Source {
            RenderTexture *texture;
            RenderFormat format;
            RenderTextureLayout layout;
            uint32_t x, y;
        };

        const uint32_t cx = textureWidth / 2;
        const uint32_t cy = textureHeight / 2;
        const Source sources[kDiagTextureSlots] = {
            { diffuseTexture.get(), RenderFormat::R16G16B16A16_FLOAT, RenderTextureLayout::SHADER_READ, cx, cy },
            { directLightTexture[current].get(), RenderFormat::R16G16B16A16_FLOAT, RenderTextureLayout::GENERAL, cx, cy },
            { indirectLightTexture[current].get(), RenderFormat::R16G16B16A16_FLOAT, RenderTextureLayout::GENERAL, cx, cy },
            { filteredDirectLightTexture[1].get(), RenderFormat::R16G16B16A16_FLOAT, RenderTextureLayout::GENERAL, cx, cy },
            { filteredIndirectLightTexture[1].get(), RenderFormat::R16G16B16A16_FLOAT, RenderTextureLayout::GENERAL, cx, cy },
            { outputTexture[current].get(), RenderFormat::R32G32B32A32_FLOAT, RenderTextureLayout::SHADER_READ, cx, cy },
            { lumaAverageTexture.get(), RenderFormat::R32_FLOAT, RenderTextureLayout::SHADER_READ, 0, 0 },
            { finalColor, finalColorFormat, finalColorLayout, finalX, finalY },
        };

        for (uint32_t i = 0; i < kDiagTextureSlots; i++) {
            const Source &source = sources[i];
            diagTextureFormats[i] = RenderFormat::UNKNOWN;
            if ((source.texture == nullptr) || (source.format == RenderFormat::UNKNOWN)) {
                continue;
            }

            const uint32_t formatSize = RenderFormatSize(source.format);
            const uint32_t rowWidth = uint32_t(256 / formatSize);
            const RenderBox box(int32_t(source.x), int32_t(source.y), int32_t(source.x + 1), int32_t(source.y + 1));
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(source.texture, RenderTextureLayout::COPY_SOURCE));
            worker->commandList->copyTextureRegion(
                RenderTextureCopyLocation::PlacedFootprint(diagTextureReadback.get(), source.format, 1, 1, 1, rowWidth, kDiagTextureSlotBytes * i),
                RenderTextureCopyLocation::Subresource(source.texture), 0, 0, 0, &box);
            worker->commandList->barriers(RenderBarrierStage::GRAPHICS_AND_COMPUTE, RenderTextureBarrier(source.texture, source.layout));
            diagTextureFormats[i] = source.format;
        }

        diagTexturesCopied = true;
    }

    static float halfToFloat(uint16_t h) {
        const uint32_t sign = uint32_t(h >> 15) << 31;
        const uint32_t exponent = (h >> 10) & 0x1F;
        const uint32_t mantissa = h & 0x3FF;
        uint32_t bits;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            }
            else {
                float value = std::ldexp(float(mantissa), -24);
                return (h & 0x8000) ? -value : value;
            }
        }
        else if (exponent == 31) {
            bits = sign | 0x7F800000u | (mantissa << 13);
        }
        else {
            bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }

        float result;
        memcpy(&result, &bits, sizeof(result));
        return result;
    }

    static void decodeTexel(RenderFormat format, const uint8_t *data, float out[4]) {
        out[0] = out[1] = out[2] = out[3] = 0.0f;
        switch (format) {
        case RenderFormat::R16G16B16A16_FLOAT: {
            uint16_t h[4];
            memcpy(h, data, sizeof(h));
            for (int i = 0; i < 4; i++) {
                out[i] = halfToFloat(h[i]);
            }
            break;
        }
        case RenderFormat::R32G32B32A32_FLOAT:
            memcpy(out, data, sizeof(float) * 4);
            break;
        case RenderFormat::R32_FLOAT:
            memcpy(out, data, sizeof(float));
            break;
        case RenderFormat::R8G8B8A8_UNORM:
            for (int i = 0; i < 4; i++) {
                out[i] = data[i] / 255.0f;
            }
            break;
        case RenderFormat::R16G16B16A16_UNORM: {
            uint16_t u[4];
            memcpy(u, data, sizeof(u));
            for (int i = 0; i < 4; i++) {
                out[i] = u[i] / 65535.0f;
            }
            break;
        }
        default:
            out[0] = -12345.0f;
            break;
        }
    }

    void RaytracingResources::assignDecalSlots(const RaytracingScene &rtScene, bool rotate) {
        std::vector<const RaytracingScene::DecalInfo *> decals;
        for (const RaytracingScene::DecalInfo &info : rtScene.decalInfos) {
            decals.push_back(&info);
        }

        // Captures count the largest decals (and take the slots over from a pending check); the periodic checks go
        // through all of them in turn.
        if (!rotate) {
            decalCheckDelay = -1;
            std::sort(decals.begin(), decals.end(), [](const RaytracingScene::DecalInfo *a, const RaytracingScene::DecalInfo *b) { return a->extent > b->extent; });
        }

        const size_t start = (rotate && !decals.empty()) ? (decalCheckCursor % decals.size()) : 0;
        diagDecalCount = std::min(uint32_t(decals.size()), DiagDecalSlots);
        uint32_t *slotParams[DiagDecalSlots] = { &rtParams.diagDecal0, &rtParams.diagDecal1, &rtParams.diagDecal2, &rtParams.diagDecal3, &rtParams.diagDecal4, &rtParams.diagDecal5 };
        for (uint32_t s = 0; s < DiagDecalSlots; s++) {
            const RaytracingScene::DecalInfo *decal = (s < diagDecalCount) ? decals[(start + s) % decals.size()] : nullptr;
            *slotParams[s] = (decal != nullptr) ? (decal->renderIndex + 1) : 0u;
            if (decal != nullptr) {
                diagDecals[s] = *decal;
            }
        }

        if (rotate) {
            decalCheckCursor += diagDecalCount;
        }
    }

    void RaytracingResources::evaluateDecalSlots(const uint32_t *counters, bool log) {
        // Counters per slot, see kDiagDecalStats in RtLibrary.hlsl.
        const uint32_t StatsStart = 64;
        const uint32_t StatsStride = 16;
        const uint32_t MinSamples = 16;
        const uint32_t BadChecksToFallBack = 3;
        static const float kBinUpper[8] = { 2.0f, 4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 32.0f, 1e9f };
        static const char *kBinNames[8] = { "under 2", "2-4", "4-8", "8-12", "12-16", "16-24", "24-32", "over 32" };
        for (uint32_t s = 0; s < diagDecalCount; s++) {
            const RaytracingScene::DecalInfo &decal = diagDecals[s];
            const uint32_t *stats = &counters[StatsStart + s * StatsStride];
            const uint32_t hits = stats[0];
            const uint32_t tooFar = stats[2];
            const uint32_t drawOrder = stats[3];
            const uint32_t notParallel = stats[4];
            const uint32_t behind = stats[5];
            const uint32_t inFront = stats[6];
            const float minDistance = (stats[7] != 0) ? [](uint32_t bits) { float value; memcpy(&value, &bits, sizeof(value)); return value; }(~stats[7]) : -1.0f;
            // Histogram bins holding the median and the 75th percentile of the too far distances.
            uint32_t median = 7;
            uint32_t upper = 7;
            bool medianFound = false;
            bool upperFound = false;
            uint32_t seen = 0;
            for (uint32_t b = 0; (b < 8) && (tooFar > 0); b++) {
                seen += stats[8 + b];
                if (!medianFound && ((seen * 2) >= tooFar)) {
                    median = b;
                    medianFound = true;
                }
                if (!upperFound && ((seen * 4) >= (tooFar * 3))) {
                    upper = b;
                    upperFound = true;
                }
            }

            // Draw order rejections are characters and small objects in front of the decal, which is right. The rest
            // decide: a decal mostly too far from its surface first learns a larger offset (enough for three quarters
            // of them, up to 16 units); one that still loses most of its pixels goes back to the raster after three
            // checks in a row.
            const uint32_t samples = hits + tooFar + notParallel;
            const bool mostlyMissed = (samples >= MinSamples) && ((hits * 2) < samples);
            const char *verdict = "";
            if (mostlyMissed && (tooFar >= notParallel) && (kBinUpper[upper] <= 16.0f)) {
                float &learned = decalOffsets[decal.id];
                const float wanted = std::min(kBinUpper[upper] + 1.0f, 16.0f);
                if (wanted > std::max(learned, 8.0f)) {
                    learned = wanted;
                    verdict = ": offset raised to fit the decal";
                }
            }

            uint8_t &badChecks = decalBadChecks[decal.id];
            if (mostlyMissed && (verdict[0] == '\0')) {
                badChecks = uint8_t(std::min(badChecks + 1, 255));
                verdict = ": mostly missed";
                if (badChecks >= BadChecksToFallBack) {
                    decalRasterFallback[decal.id] = diagFrame;
                    verdict = ": mostly missed three checks in a row, rasterized after the path traced pass for now";
                }
            }
            else if (samples >= MinSamples) {
                badChecks = 0;
            }

            const auto learnedIt = decalOffsets.find(decal.id);
            const float offset = (learnedIt != decalOffsets.end()) ? learnedIt->second : 8.0f;
            if (log || (verdict[0] != '\0')) {
                rtDiagLog("%sdecal draw %u (texture %016llX, %u triangles, span %.0f, offset %.1f): %u camera hits; not taken: %u too far (%u behind, %u in front of the surface, nearest %.1f, median %s, 3/4 within %s units), %u draw order, %u not parallel%s%s",
                    log ? "    " : "decal check: ", decal.renderIndex, (unsigned long long)decal.key, decal.triangles, decal.extent, offset, hits, tooFar, behind, inFront, minDistance,
                    kBinNames[median], kBinNames[upper], drawOrder, notParallel, verdict, ((hits + tooFar + drawOrder + notParallel) == 0) ? " (not on screen, or hidden)" : "");
            }
        }

        diagDecalCount = 0;
    }

    void RaytracingResources::copyDecalCheckCounters(RenderWorker *worker) {
        if (hitInstanceIdBuffer == nullptr) {
            return;
        }

        if (decalCheckReadback == nullptr) {
            decalCheckReadback = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(kDiagCounterBytes));
        }

        worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(hitInstanceIdBuffer.get(), RenderBufferAccess::READ));
        worker->commandList->copyBufferRegion(decalCheckReadback->at(0), hitInstanceIdBuffer->at(0), kDiagCounterBytes);
        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(hitInstanceIdBuffer.get(), RenderBufferAccess::WRITE));
        decalCheckDelay = 8;
    }

    void RaytracingResources::readDecalCheckCounters() {
        if (decalCheckDelay < 0) {
            return;
        }

        if (decalCheckDelay > 0) {
            decalCheckDelay--;
            return;
        }

        decalCheckDelay = -1;
        if (decalCheckReadback == nullptr) {
            return;
        }

        const RenderRange readRange(0, kDiagCounterBytes);
        const uint8_t *bytes = static_cast<const uint8_t *>(decalCheckReadback->map(0, &readRange));
        if (bytes == nullptr) {
            return;
        }

        uint32_t counters[256];
        memcpy(counters, bytes, sizeof(counters));
        decalCheckReadback->unmap(0, nullptr);
        evaluateDecalSlots(counters, false);
    }

    void RaytracingResources::readDiagnosticBuffers() {
        if (diagReadbackDelay < 0) {
            return;
        }

        if (diagReadbackDelay > 0) {
            diagReadbackDelay--;
            return;
        }

        diagReadbackDelay = -1;
        if (diagReadback == nullptr) {
            return;
        }

        const RenderRange readRange(0, kDiagReadbackBytes);
        const uint8_t *bytes = static_cast<const uint8_t *>(diagReadback->map(0, &readRange));
        if (bytes == nullptr) {
            rtDiagLog("  GPU readback: map failed");
            return;
        }

        uint32_t counters[256];
        float color[16][4];
        float extra[16][4];
        memcpy(counters, bytes, sizeof(counters));
        memcpy(color, bytes + kDiagCounterBytes, sizeof(color));
        memcpy(extra, bytes + kDiagCounterBytes + kHitBufferBytes, sizeof(extra));
        float emissiveColors[EmissiveSlots][4];
        float emissiveGlows[EmissiveSlots][4];
        memcpy(emissiveColors, bytes + kDiagCounterBytes + kHitBufferBytes * 2, sizeof(emissiveColors));
        memcpy(emissiveGlows, bytes + kDiagCounterBytes + kHitBufferBytes * 2 + kDiagEmissiveBytes, sizeof(emissiveGlows));
        diagReadback->unmap(0, nullptr);

        rtDiagLog("--- GPU readback ---");
        rtDiagLog("  shader marker 0x%X (expect 0xC0FFEE)", counters[8]);
        rtDiagLog("  pixels in bounds %u, outside viewport %u, primary rays traced %u, primary hits %u, bad direction %u, non-finite origin %u",
            counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]);
        rtDiagLog("  center ray origin %g %g %g, direction %g %g %g", color[0][0], color[0][1], color[0][2], color[1][0], color[1][1], color[1][2]);
        rtDiagLog("  any-hit rejected texels: camera/bounce rays %u, shadow rays %u; camera ray hits skipped as raster-culled faces %u", counters[6], counters[7], counters[14]);
        rtDiagLog("  culled hits on triangles crossing the camera plane %u; pixels where a surface closer than the near plane was skipped %u", counters[16], counters[17]);
        rtDiagLog("  shadow rays from camera hits %u, blocked %u", counters[12], counters[13]);
        rtDiagLog("  coplanar tie-breaks that picked a later draw: %u pixels", counters[15]);
        rtDiagLog("  center ray hit %g, hitT %g, hit point velocity %g %g %g, render index %g", color[2][3], color[1][3], color[2][0], color[2][1], color[2][2], color[5][3]);
        {
            // The material of the surface in the middle of the screen, to find its entry in rt_materials.json.
            const int32_t centerIndex = int32_t(color[5][3]);
            const uint64_t centerKey = ((color[2][3] > 0.5f) && (centerIndex >= 0) && (size_t(centerIndex) < diagMaterialKeys.size())) ? diagMaterialKeys[centerIndex] : 0;
            RtMaterialEntry centerMaterial;
            if (centerKey == 0) {
                rtDiagLog("  center material: no textured path traced surface (render index %d)", centerIndex);
            }
            else if (rtMaterialDatabase().find(centerKey, centerMaterial)) {
                rtDiagLog("  center material: key %016llX, class %s, roughness %.2f, metalness %.2f, specular %.2f, source %s (%s)", (unsigned long long)centerKey,
                    rtMaterialClassName(centerMaterial.materialClass), centerMaterial.roughness, centerMaterial.metalness, centerMaterial.specular,
                    centerMaterial.user ? "user" : "guess", centerMaterial.label.c_str());
            }
            else {
                rtDiagLog("  center material: key %016llX, not guessed yet (matte until then)", (unsigned long long)centerKey);
            }

            // The signature of the draw at the screen center, to copy into rt_emitters.json.
            if ((color[2][3] > 0.5f) && (centerIndex >= 0) && (size_t(centerIndex) < diagDrawSignatures.size()) && (diagDrawSignatures[centerIndex].signature != 0)) {
                const RtEmitterSignature &s = diagDrawSignatures[centerIndex];
                RtEmitterParams params;
                bool applies = false;
                const bool listed = rtMaterialDatabase().matchEmitter(s, -1, -1, params, applies);
                rtDiagLog("  center draw: signature %016llX (%s), texture %016llX, vertex address %08X, combiner %08X %08X, other mode %08X %08X, prim %.2f %.2f %.2f %.2f, env %.2f %.2f %.2f %.2f, %u triangles; %s",
                    (unsigned long long)s.signature, emitterKindName(s.kind), (unsigned long long)s.texture, s.vertexAddress, s.combiner[0], s.combiner[1], s.otherMode[0], s.otherMode[1],
                    s.prim[0], s.prim[1], s.prim[2], s.prim[3], s.env[0], s.env[1], s.env[2], s.env[3], s.triangles,
                    listed ? (params.enabled ? "matches an enabled emitter entry" : "matches a disabled emitter entry") : "not an emitter (path traced surfaces only glow when listed)");
            }
            else {
                rtDiagLog("  center draw: no path traced draw at the screen center (effects drawn over the scene are listed as candidate draws above)");
            }
        }
        const float historyRate = (counters[18] > 0) ? (100.0f * float(counters[19]) / float(counters[18])) : 0.0f;
        const float averageOcclusion = (counters[21] > 0) ? (float(counters[20]) / (255.0f * float(counters[21]))) : 1.0f;
        rtDiagLog("  temporal history reused for %.1f%% of lit pixels (%u of %u, the rest rejected by depth or normal); average ambient occlusion %.2f (1 = open); smooth shaded camera hits %u",
            historyRate, counters[19], counters[18], averageOcclusion, counters[22]);
        rtDiagLog("  GPU viewport %g %g %g %g, resolution %g %g, y sign %g, lights %g",
            color[3][0], color[3][1], color[3][2], color[3][3], color[4][0], color[4][1], color[5][0], color[5][1]);
        rtDiagLog("  center material: albedo %g %g %g (hit %g), fog color %g %g %g amount %g",
            color[9][0], color[9][1], color[9][2], color[9][3], color[10][0], color[10][1], color[10][2], color[10][3]);
        rtDiagLog("  center material: draw %g, texture0 %g, texture1 %g, alpha compare %g, tile index %g, tile count %g, cvgXAlpha %g, alpha blend %g",
            color[11][0], color[11][1], color[11][2], color[11][3], color[12][0], color[12][1], color[12][2], color[12][3]);
        rtDiagLog("  center light: NdotL %g, visibility %g, light distance %g, direct factor %g", color[13][0], color[13][1], color[13][2], color[13][3]);
        rtDiagLog("  center light: surface normal %g %g %g, light direction %g %g %g, blocker render index %g, blocker hitT %g (-1 = none)",
            color[14][0], color[14][1], color[14][2], color[15][0], color[15][1], color[15][2], color[14][3], color[15][3]);
        {
            // What lights the surface at the screen center: aim at a wrongly lit polygon and press F9.
            const float agreement = color[6][0] * color[14][0] + color[6][1] * color[14][1] + color[6][2] * color[14][2];
            rtDiagLog("  center surface: geometric normal (toward the camera) %.3f %.3f %.3f, shading normal %.3f %.3f %.3f (agreement %.2f), %s, face toward the sun %.2f (a solid face below -0.25 gets no sun)",
                color[6][0], color[6][1], color[6][2], color[14][0], color[14][1], color[14][2], agreement,
                (color[6][3] > 0.5f) ? "two-sided (alpha tested: lit from either side)" : "one-sided", color[7][3]);
            float triangle[9];
            memcpy(triangle, &counters[24], sizeof(triangle));
            const float e1[3] = { triangle[3] - triangle[0], triangle[4] - triangle[1], triangle[5] - triangle[2] };
            const float e2[3] = { triangle[6] - triangle[0], triangle[7] - triangle[1], triangle[8] - triangle[2] };
            const float faceCross[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
            const float area = 0.5f * std::sqrt(faceCross[0] * faceCross[0] + faceCross[1] * faceCross[1] + faceCross[2] * faceCross[2]);
            rtDiagLog("  center triangle: %s face normal (%s); vertices %.3f %.3f %.3f / %.3f %.3f %.3f / %.3f %.3f %.3f, area %.4f; %u camera hits this frame were on degenerate triangles",
                (counters[33] != 0) ? "DEGENERATE," : "valid", (counters[33] != 0) ? "shading normal used as the face normal" : "from its edges",
                triangle[0], triangle[1], triangle[2], triangle[3], triangle[4], triangle[5], triangle[6], triangle[7], triangle[8], area, counters[23]);
            rtDiagLog("  camera hits whose shading normal disagrees with the face by more than about 72 degrees: %u; debug view %u (F10 cycles: 1 face normal, 2 shading normal, 3 disagreement in red, 4 sun visibility, 5 direct factor, 6 draw ids, 7 lit while facing away in red, 8 decals: green shown, yellow too far, red draw order, blue not parallel)",
                counters[34], raytracingDebugView());
            float probeX, probeY;
            raytracingDiagnosticsPixel(probeX, probeY);
            if (probeX >= 0.0f) {
                rtDiagLog("  note: the center lines describe the pixel under the mouse cursor when F9 was pressed, at %.3f %.3f of the image", probeX, probeY);
            }
            else {
                rtDiagLog("  note: the center lines describe whatever is exactly at the screen center, often the scenery behind the character in third person view");
            }
            float bounceFill[7];
            memcpy(bounceFill, &counters[35], sizeof(bounceFill));
            rtDiagLog("  indirect light: %u bounce%s; center pixel fill this frame from bounce 1 %.3f %.3f %.3f, from bounce 2 %.3f %.3f %.3f, fill strength %.2f",
                counters[42], (counters[42] == 1) ? "" : "s", bounceFill[0], bounceFill[1], bounceFill[2], bounceFill[3], bounceFill[4], bounceFill[5], bounceFill[6]);
            if (counters[44] == 4) {
                rtDiagLog("  water: center pixel is on a scrolling surface the water pass skipped because its color isn't blue or blue-green (draw %u); %u water pixels this frame",
                    counters[45], counters[43]);
            }
            else if (counters[44] != 0) {
                static const char *kWaterKinds[] = { "", "blended water over the scene", "opaque (path traced) water", "blended water against the sky" };
                float waterValues[3];
                memcpy(waterValues, &counters[46], sizeof(waterValues));
                rtDiagLog("  water: center pixel IS water, %s (draw %u): opacity %.2f, Fresnel reflectance %.3f, depth to the ground below %.1f (-1 = none found); %u water pixels this frame",
                    kWaterKinds[std::min(counters[44], 3u)], counters[45], waterValues[0], waterValues[1], waterValues[2], counters[43]);
            }
            else {
                rtDiagLog("  water: center pixel is not water; %u water pixels this frame", counters[43]);
            }
            rtDiagLog("  decals: %u camera hits on path traced decals this frame, %u of them blended over the surface below by the decal's alpha", counters[49], counters[50]);
            float centerEmission[3];
            memcpy(centerEmission, &counters[201], sizeof(centerEmission));
            rtDiagLog("  emission: center pixel %s (emitted color %.2f %.2f %.2f); %u emissive pixels this frame", (counters[200] != 0) ? "IS emissive" : "is not emissive",
                centerEmission[0], centerEmission[1], centerEmission[2], counters[204]);
            float foliageNormal[3];
            memcpy(foliageNormal, &counters[206], sizeof(foliageNormal));
            if (counters[205] != 0) {
                rtDiagLog("  foliage: center pixel IS foliage, soft normal %.3f %.3f %.3f (wrapped light, no self shadowing); %u foliage pixels this frame",
                    foliageNormal[0], foliageNormal[1], foliageNormal[2], counters[209]);
            }
            else {
                rtDiagLog("  foliage: center pixel is not foliage; %u foliage pixels this frame", counters[209]);
            }

            // The terms of the center pixel's color, one per line item, to tell a shadow from the game's own vertex
            // shading: albedo (the texture and vertex shade combined) and the shade alone, the sun's visibility and
            // facing (see "center surface" for the direct factor and the bounce fill) and the proxy skips.
            float centerShade[3];
            memcpy(centerShade, &counters[210], sizeof(centerShade));
            rtDiagLog("  center terms: vertex shade %.3f %.3f %.3f (draw %u; N64 lighting of lit vertices, baked into the albedo; if it differs between the two halves of a tree, the game's own shading has the seam), billboard shadow ray hits on nearby proxies skipped this frame %u",
                centerShade[0], centerShade[1], centerShade[2], counters[213], counters[214]);
            rtDiagLog("  indirect pass GPU time: %.3f ms average with 1 bounce, %.3f ms with 2 bounces (0 = not measured yet in that mode)",
                indirectPassMs[0], indirectPassMs[1]);
            rtDiagLog("  center surface: sun visibility %.2f, N.L %.2f, direct factor %.3f; emissive light added %.3f %.3f %.3f; specular added %.3f %.3f %.3f with diffuse kept %.2f; bounce fill %.3f %.3f %.3f, ambient occlusion %.2f (1 = open)",
                color[13][1], color[13][0], color[13][3], color[7][0], color[7][1], color[7][2], color[8][0], color[8][1], color[8][2], color[8][3],
                extra[15][0], extra[15][1], extra[15][2], extra[15][3]);
        }

        evaluateDecalSlots(counters, true);

        for (uint32_t s = 0; s < EmissiveSlots; s++) {
            const float *c = emissiveColors[s];
            const float *g = emissiveGlows[s];
            if ((c[0] + c[1] + c[2] + c[3]) <= 0.0f) {
                continue;
            }

            // g = smoothed glow, saturation, brightness, category of the last sample.
            const uint32_t categoryTag = uint32_t(g[3] + 0.5f);
            const EmissiveCategory category = EmissiveCategory(std::min(categoryTag % 4u, 2u));
            const char *verdict = "emits";
            if (g[0] < 0.05f) {
                if (g[2] < ((category == EmissiveCategory::Additive) ? 0.15f : 0.7f)) {
                    verdict = "dark (dust or smoke)";
                }
                else if ((category == EmissiveCategory::Blended) && (g[1] < 0.15f)) {
                    verdict = "untinted white or grey (dust or smoke)";
                }
                else if ((category == EmissiveCategory::Cutout) && (g[1] < 0.35f)) {
                    verdict = "low saturation";
                }
                else if ((category == EmissiveCategory::Cutout) && (c[1] < (c[0] * 0.25f))) {
                    verdict = "pure red, not a flame";
                }
                else if (category == EmissiveCategory::Cutout) {
                    verdict = "not flame colored";
                }
                else {
                    verdict = "weak";
                }
            }

            rtDiagLog("  slot %u (%s): color %.2f %.2f %.2f, coverage %.2f, glow %.2f, saturation %.2f, brightness %.2f: %s",
                s, emissiveCategoryName(category), c[0], c[1], c[2], c[3], g[0], g[1], g[2], verdict);
        }

        // Path traced material against the raster color of the same pixel (unfogged hits only).
        rtDiagLog("  material vs raster: compared pixels %u, mismatched (any channel off by > 0.2) %u", counters[9], counters[10]);
        auto bitsOf = [](float value) {
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            return bits;
        };
        auto unpackColor = [](uint32_t packed, float out[3]) {
            out[0] = float(packed & 0xFF) / 255.0f;
            out[1] = float((packed >> 8) & 0xFF) / 255.0f;
            out[2] = float((packed >> 16) & 0xFF) / 255.0f;
        };
        const uint32_t recordCount = std::min(counters[11], 5u);
        for (uint32_t r = 0; r < recordCount; r++) {
            const float *a = extra[r * 3 + 0];
            const float *b = extra[r * 3 + 1];
            const float *c = extra[r * 3 + 2];
            float albedo[3], raster[3];
            unpackColor(bitsOf(b[2]), albedo);
            unpackColor(bitsOf(b[3]), raster);
            const uint32_t tileBits = bitsOf(c[0]);
            const uint32_t pixel = bitsOf(c[3]);
            rtDiagLog("  mismatch %u at pixel %u,%u: draw %g, albedo %.2f %.2f %.2f vs raster %.2f %.2f %.2f",
                r, pixel & 0xFFFF, pixel >> 16, a[0], albedo[0], albedo[1], albedo[2], raster[0], raster[1], raster[2]);
            rtDiagLog("    combiner %08X %08X, other mode %08X %08X, render flags %08X, tile0 fmt %u siz %u palette %u cms %u cmt %u, tiles %u, gpu tile flags %X, texture %u",
                bitsOf(a[1]), bitsOf(a[2]), bitsOf(a[3]), bitsOf(b[0]), bitsOf(b[1]),
                tileBits & 0xF, (tileBits >> 4) & 0xF, (tileBits >> 8) & 0xF, (tileBits >> 12) & 0xF, (tileBits >> 16) & 0xF, (tileBits >> 20) & 0xF,
                bitsOf(c[1]), bitsOf(c[2]));
        }

        if (diagTexturesCopied && (diagTextureReadback != nullptr)) {
            diagTexturesCopied = false;
            const RenderRange textureRange(0, kDiagTextureSlotBytes * kDiagTextureSlots);
            const uint8_t *texels = static_cast<const uint8_t *>(diagTextureReadback->map(0, &textureRange));
            if (texels == nullptr) {
                rtDiagLog("  texture readback: map failed");
                return;
            }

            rtDiagLog("  center pixel per stage (rgba):");
            for (uint32_t i = 0; i < kDiagTextureSlots; i++) {
                if (diagTextureFormats[i] == RenderFormat::UNKNOWN) {
                    rtDiagLog("    %s: not captured", kDiagTextureNames[i]);
                    continue;
                }
                float value[4];
                decodeTexel(diagTextureFormats[i], texels + kDiagTextureSlotBytes * i, value);
                rtDiagLog("    %s: %g %g %g %g", kDiagTextureNames[i], value[0], value[1], value[2], value[3]);
            }
            diagTextureReadback->unmap(0, nullptr);
        }
        else {
            rtDiagLog("  texture readback: not captured this frame");
        }
    }
};
