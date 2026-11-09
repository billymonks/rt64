//
// RT64
//

#include "rt64_raytracing_resources.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cstdio>

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

    Upscaler *RaytracingResources::getUpscaler(UpscaleMode) const {
        return nullptr;
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

    void RaytracingResources::createOutputBuffers(RenderWorker *worker, uint32_t width, uint32_t height) {
        textureWidth = std::max(width, 1u);
        textureHeight = std::max(height, 1u);
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
            outputTexture[i] = worker->device->createTexture(RenderTextureDesc::ColorTarget(textureWidth, textureHeight, RenderFormat::R32G32B32A32_FLOAT));
            const RenderTexture *color = outputTexture[i].get();
            outputFramebuffer[i] = worker->device->createFramebuffer(RenderFramebufferDesc(&color, 1));
        }

        lumaAverageTexture = makeTexture(device, 1, 1, RenderFormat::R32_FLOAT, uav);
        downscaledOutputTexture = makeTexture(device, std::max(textureWidth / 8, 1u), std::max(textureHeight / 8, 1u), RenderFormat::R32G32B32A32_FLOAT, uav);
        upscaledOutputTexture = makeTexture(device, textureWidth, textureHeight, RenderFormat::R32G32B32A32_FLOAT, uav);

        const RenderBufferFlags formatted = RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::FORMATTED;
        auto growOnce = [&](std::unique_ptr<RenderBuffer> &buffer, uint64_t size, RenderBufferFlags flags) {
            uint64_t capacity = 0;
            growBuffer(device, buffer, capacity, size, flags);
        };
        growOnce(hitVelocityDistanceBuffer, kHitBufferBytes, formatted);
        growOnce(hitColorBuffer, kHitBufferBytes, formatted);
        growOnce(hitNormalFogBuffer, kHitBufferBytes, formatted);
        growOnce(hitInstanceIdBuffer, kHitBufferBytes, formatted);
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

        if (rtParamsBuffer.get() != nullptr) {
            postProcessSet->setBuffer(postProcessSet->RtParams, rtParamsBuffer.get(), sizeof(interop::RaytracingParams));
        }
        postProcessSet->setTexture(postProcessSet->gInput, outputTexture[current].get(), RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gFlow, flowTexture.get(), RenderTextureLayout::SHADER_READ);
        postProcessSet->setTexture(postProcessSet->gLumaAvg, lumaAverageTexture.get(), RenderTextureLayout::SHADER_READ);
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

            level.buffer = worker->device->createBuffer(RenderBufferDesc::AccelerationStructureBuffer(level.buildInfo.accelerationStructureSize));
            level.accelerationStructure = worker->device->createAccelerationStructure(RenderAccelerationStructureDesc(RenderAccelerationStructureType::BOTTOM_LEVEL, level.buffer->at(0), level.buildInfo.accelerationStructureSize));
            scratchNeeded = std::max(scratchNeeded, level.buildInfo.scratchSize);

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

        if (scratchNeeded > scratchCapacity) {
            scratchCapacity = roundUp(scratchNeeded, 256);
            scratchBuffer = worker->device->createBuffer(RenderBufferDesc::DefaultBuffer(scratchCapacity, RenderBufferFlag::ACCELERATION_STRUCTURE_SCRATCH | RenderBufferFlag::UNORDERED_ACCESS));
        }
    }

    void RaytracingResources::submitBottomLevelASCreation(RenderWorker *worker) {
        if ((scratchBuffer == nullptr) || bottomLevelASVector.empty()) {
            return;
        }

        worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(scratchBuffer.get(), RenderBufferAccess::WRITE));
        if (!geometryBuffers.empty()) {
            std::vector<RenderBufferBarrier> barriers;
            barriers.reserve(geometryBuffers.size());
            for (RenderBuffer *buffer : geometryBuffers) {
                barriers.emplace_back(buffer, RenderBufferAccess::READ);
            }
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, barriers);
        }

        for (BottomLevel &level : bottomLevelASVector) {
            // The scratch buffer is reused. Each build must finish before the next one writes it.
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(scratchBuffer.get(), RenderBufferAccess::WRITE));
            worker->commandList->buildBottomLevelAS(level.accelerationStructure.get(), scratchBuffer->at(0), level.buildInfo);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(level.buffer.get(), RenderBufferAccess::WRITE));
        }

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
            instances.emplace_back(level.buffer->at(0), instanceID, mask, 0, cullDisable, RenderAffineTransform());
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

    void RaytracingResources::updateLightsBuffer(RenderWorker *worker, const RaytracingScene &rtScene) {
        uint32_t count = rtScene.lightCount;
        if ((count > 0) && (rtScene.pointLights == nullptr)) {
            count = 0;
        }
        rtParams.lightsCount = count;
        interop::PointLight fallback = {};
        const void *data = (count > 0) ? static_cast<const void *>(rtScene.pointLights) : static_cast<const void *>(&fallback);
        const uint64_t bytes = sizeof(interop::PointLight) * std::max(count, 1u);
        growBuffer(device, lightsBuffer, lightsCapacity, bytes, RenderBufferFlag::STORAGE);
        uploadToBuffer(worker, lightsUpload, lightsUploadCapacity, lightsBuffer.get(), data, bytes);
    }

    // Diagnostics

    static const uint64_t kDiagCounterBytes = 256;
    static const uint64_t kDiagReadbackBytes = kDiagCounterBytes + kHitBufferBytes * 2;

    void rtDiagLog(const char *fmt, ...) {
#   if defined(_WIN32)
        const char *localAppData = std::getenv("LOCALAPPDATA");
        if (localAppData == nullptr) {
            return;
        }

        char path[512];
        std::snprintf(path, sizeof(path), "%s\\Quest64Recompiled\\rt.log", localAppData);
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
        rtDiagLog("  projections rasterized because their camera differs from the scene's: %u; decal/translucent depth mode draws rasterized after: %u",
            rtScene.incompatibleProjections, rtScene.skippedLayeredDraws);
        rtDiagLog("  billboard sprites detected: cylindrical %u, spherical %u; light facing shadow proxies built for %u draws",
            rtScene.billboardsCylindrical, rtScene.billboardsSpherical, rtScene.billboardProxies);
        rtDiagLog("  pixel spread angle %g", rtParams.pixelSpreadAngle);
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

        if (diagZeroUpload == nullptr) {
            diagZeroUpload = worker->device->createBuffer(RenderBufferDesc::UploadBuffer(kHitBufferBytes));
            void *mapped = diagZeroUpload->map();
            if (mapped != nullptr) {
                memset(mapped, 0, size_t(kHitBufferBytes));
                diagZeroUpload->unmap();
            }
        }

        RenderBuffer *targets[] = { hitInstanceIdBuffer.get(), hitColorBuffer.get(), hitVelocityDistanceBuffer.get() };
        for (RenderBuffer *target : targets) {
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(target, RenderBufferAccess::WRITE));
            worker->commandList->copyBufferRegion(target->at(0), diagZeroUpload->at(0), kHitBufferBytes);
            worker->commandList->barriers(RenderBarrierStage::COMPUTE, RenderBufferBarrier(target, RenderBufferAccess::WRITE));
        }
    }

    void RaytracingResources::copyDiagnosticBuffers(RenderWorker *worker) {
        if ((hitInstanceIdBuffer == nullptr) || (hitColorBuffer == nullptr) || (hitVelocityDistanceBuffer == nullptr)) {
            return;
        }

        if (diagReadback == nullptr) {
            diagReadback = worker->device->createBuffer(RenderBufferDesc::ReadbackBuffer(kDiagReadbackBytes));
        }

        RenderBuffer *sources[] = { hitInstanceIdBuffer.get(), hitColorBuffer.get(), hitVelocityDistanceBuffer.get() };
        const uint64_t offsets[] = { 0, kDiagCounterBytes, kDiagCounterBytes + kHitBufferBytes };
        for (uint32_t i = 0; i < 3; i++) {
            worker->commandList->barriers(RenderBarrierStage::COPY, RenderBufferBarrier(sources[i], RenderBufferAccess::READ));
            worker->commandList->copyBufferRegion(diagReadback->at(offsets[i]), sources[i]->at(0), kHitBufferBytes);
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

        uint32_t counters[64];
        float color[16][4];
        float extra[16][4];
        memcpy(counters, bytes, sizeof(counters));
        memcpy(color, bytes + kDiagCounterBytes, sizeof(color));
        memcpy(extra, bytes + kDiagCounterBytes + kHitBufferBytes, sizeof(extra));
        diagReadback->unmap(0, nullptr);

        rtDiagLog("--- GPU readback ---");
        rtDiagLog("  shader marker 0x%X (expect 0xC0FFEE)", counters[8]);
        rtDiagLog("  pixels in bounds %u, outside viewport %u, primary rays traced %u, primary hits %u, bad direction %u, non-finite origin %u",
            counters[0], counters[1], counters[2], counters[3], counters[4], counters[5]);
        rtDiagLog("  center ray origin %g %g %g, direction %g %g %g", color[0][0], color[0][1], color[0][2], color[1][0], color[1][1], color[1][2]);
        rtDiagLog("  any-hit rejected texels: camera/bounce rays %u, shadow rays %u; camera ray hits skipped as raster-culled faces %u", counters[6], counters[7], counters[14]);
        rtDiagLog("  shadow rays from camera hits %u, blocked %u", counters[12], counters[13]);
        rtDiagLog("  coplanar tie-breaks that picked a later draw: %u pixels", counters[15]);
        rtDiagLog("  center ray hit %g, hitT %g, hit position %g %g %g, render index %g", color[2][3], color[1][3], color[2][0], color[2][1], color[2][2], color[5][3]);
        rtDiagLog("  GPU viewport %g %g %g %g, resolution %g %g, y sign %g, lights %g",
            color[3][0], color[3][1], color[3][2], color[3][3], color[4][0], color[4][1], color[5][0], color[5][1]);
        rtDiagLog("  center material: albedo %g %g %g (hit %g), fog color %g %g %g amount %g",
            color[9][0], color[9][1], color[9][2], color[9][3], color[10][0], color[10][1], color[10][2], color[10][3]);
        rtDiagLog("  center material: draw %g, texture0 %g, texture1 %g, alpha compare %g, tile index %g, tile count %g, cvgXAlpha %g, alpha blend %g",
            color[11][0], color[11][1], color[11][2], color[11][3], color[12][0], color[12][1], color[12][2], color[12][3]);
        rtDiagLog("  center light: NdotL %g, visibility %g, light distance %g, direct factor %g", color[13][0], color[13][1], color[13][2], color[13][3]);
        rtDiagLog("  center light: surface normal %g %g %g, light direction %g %g %g, blocker render index %g, blocker hitT %g (-1 = none)",
            color[14][0], color[14][1], color[14][2], color[15][0], color[15][1], color[15][2], color[14][3], color[15][3]);
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
