//
// RT64
//

#include "rt64_upscaler_fsr3.h"

#if RT64_FSR3_ENABLED && defined(_WIN32)

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>

#include <Windows.h>

#include "plume_d3d12.h"

#include "ffx_api.h"
#include "dx12/ffx_api_dx12.h"
#include "ffx_upscale.h"

namespace RT64 {
    void rtDiagLog(const char *fmt, ...);

    static void fsr3Message(uint32_t type, const wchar_t *message) {
        std::string text;
        for (const wchar_t *c = message; (c != nullptr) && (*c != 0); c++) {
            text.push_back((*c < 128) ? char(*c) : '?');
        }

        rtDiagLog("FSR 3 %s: %s", (type == FFX_API_MESSAGE_TYPE_ERROR) ? "error" : "warning", text.c_str());
    }

    static void localFunction() { }

    // The resource state the SDK must assume, from the state Plume last transitioned the texture to. The SDK leaves
    // the resources in these states when it's done.
    static uint32_t ffxStateFromD3D12(D3D12_RESOURCE_STATES states) {
        if (states & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            return FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
        }

        uint32_t ffxState = 0;
        if (states & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) {
            ffxState |= FFX_API_RESOURCE_STATE_COMPUTE_READ;
        }

        if (states & D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE) {
            ffxState |= FFX_API_RESOURCE_STATE_PIXEL_READ;
        }

        if (ffxState != 0) {
            return ffxState;
        }

        if (states & D3D12_RESOURCE_STATE_COPY_SOURCE) {
            return FFX_API_RESOURCE_STATE_COPY_SRC;
        }

        if (states & D3D12_RESOURCE_STATE_COPY_DEST) {
            return FFX_API_RESOURCE_STATE_COPY_DEST;
        }

        if (states & D3D12_RESOURCE_STATE_RENDER_TARGET) {
            return FFX_API_RESOURCE_STATE_RENDER_TARGET;
        }

        return FFX_API_RESOURCE_STATE_COMMON;
    }

    static FfxApiResource ffxResource(RenderTexture *texture) {
        if (texture == nullptr) {
            return FfxApiResource{};
        }

        plume::D3D12Texture *d3dTexture = static_cast<plume::D3D12Texture *>(texture);
        return ffxApiGetResourceDX12(d3dTexture->d3d, ffxStateFromD3D12(d3dTexture->resourceStates));
    }

    struct FSR3Upscaler : Upscaler {
        RenderDevice *device = nullptr;
        HMODULE loaderModule = nullptr;
        PfnFfxCreateContext createContext = nullptr;
        PfnFfxDestroyContext destroyContext = nullptr;
        PfnFfxQuery query = nullptr;
        PfnFfxDispatch dispatch = nullptr;
        ffxContext context = nullptr;
        uint32_t contextWidth = 0;
        uint32_t contextHeight = 0;
        std::string providerName;
        bool failed = false;

        ~FSR3Upscaler() override {
            destroy();
            if (loaderModule != nullptr) {
                FreeLibrary(loaderModule);
            }
        }

        bool load(std::string &failureReason) {
            HMODULE moduleHandle = nullptr;
            if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, LPCWSTR(localFunction), &moduleHandle)) {
                failureReason = "the module path is unknown";
                return false;
            }

            WCHAR modulePath[MAX_PATH];
            const DWORD modulePathSize = GetModuleFileNameW(moduleHandle, modulePath, MAX_PATH);
            if ((modulePathSize == 0) || (modulePathSize >= MAX_PATH)) {
                failureReason = "the module path is unknown";
                return false;
            }

            std::filesystem::path loaderPath(modulePath);
            loaderPath.replace_filename(L"amd_fidelityfx_loader_dx12.dll");
            loaderModule = LoadLibraryW(loaderPath.c_str());
            if (loaderModule == nullptr) {
                failureReason = "amd_fidelityfx_loader_dx12.dll is missing next to the executable";
                return false;
            }

            createContext = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(loaderModule, "ffxCreateContext"));
            destroyContext = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(loaderModule, "ffxDestroyContext"));
            query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(loaderModule, "ffxQuery"));
            dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(loaderModule, "ffxDispatch"));
            if ((createContext == nullptr) || (destroyContext == nullptr) || (query == nullptr) || (dispatch == nullptr)) {
                failureReason = "amd_fidelityfx_loader_dx12.dll doesn't export the FSR API";
                return false;
            }

            return true;
        }

        void destroy() {
            if ((context != nullptr) && (destroyContext != nullptr)) {
                destroyContext(&context, nullptr);
            }

            context = nullptr;
            contextWidth = 0;
            contextHeight = 0;
        }

        bool create(uint32_t displayWidth, uint32_t displayHeight, std::string &failureReason) {
            destroy();

            plume::D3D12Device *d3dDevice = static_cast<plume::D3D12Device *>(device);
            ffxCreateBackendDX12Desc backendDesc = {};
            backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
            backendDesc.device = d3dDevice->d3d;

            ffxCreateContextDescUpscaleVersion versionDesc = {};
            versionDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
            versionDesc.version = FFX_UPSCALER_VERSION;
            backendDesc.header.pNext = &versionDesc.header;

            // The ray traced image is in display space from 0 to 1, with OpenGL style depth mapped to 0 (near) to 1 (far).
            // The maximum render size is the display size, so every quality preset uses the same context.
            ffxCreateContextDescUpscale createDesc = {};
            createDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
            createDesc.header.pNext = &backendDesc.header;
            createDesc.flags = 0;
            createDesc.maxRenderSize = { displayWidth, displayHeight };
            createDesc.maxUpscaleSize = { displayWidth, displayHeight };
            createDesc.fpMessage = fsr3Message;

            const ffxReturnCode_t result = createContext(&context, &createDesc.header, nullptr);
            if (result != FFX_API_RETURN_OK) {
                context = nullptr;
                failureReason = "ffxCreateContext failed with code " + std::to_string(result);
                return false;
            }

            contextWidth = displayWidth;
            contextHeight = displayHeight;

            ffxQueryGetProviderVersion versionQuery = {};
            versionQuery.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
            if ((query(&context, &versionQuery.header) == FFX_API_RETURN_OK) && (versionQuery.versionName != nullptr)) {
                providerName = std::string("AMD FSR ") + versionQuery.versionName;
            }
            else {
                providerName = "AMD FSR (unknown version)";
            }

            rtDiagLog("FSR 3: upscaler context created for %u x %u, provider %s", displayWidth, displayHeight, providerName.c_str());
            return true;
        }

        void set(RenderWorker *worker, QualityMode inQuality, int renderWidth, int renderHeight, int displayWidth, int displayHeight) override {
            if (failed || ((context != nullptr) && (contextWidth == uint32_t(displayWidth)) && (contextHeight == uint32_t(displayHeight)))) {
                return;
            }

            std::string failureReason;
            if (!create(uint32_t(displayWidth), uint32_t(displayHeight), failureReason)) {
                rtDiagLog("FSR 3: %s", failureReason.c_str());
                failed = true;
            }
        }

        bool getQualityInformation(QualityMode quality, int displayWidth, int displayHeight, int &renderWidth, int &renderHeight) override {
            getRenderSize(quality, displayWidth, displayHeight, renderWidth, renderHeight);
            return true;
        }

        int getJitterPhaseCount(int renderWidth, int displayWidth) override {
            int32_t phaseCount = 0;
            ffxQueryDescUpscaleGetJitterPhaseCount phaseQuery = {};
            phaseQuery.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
            phaseQuery.renderWidth = uint32_t(std::max(renderWidth, 1));
            phaseQuery.displayWidth = uint32_t(std::max(displayWidth, 1));
            phaseQuery.pOutPhaseCount = &phaseCount;
            ffxContext *queryContext = (context != nullptr) ? &context : nullptr;
            if ((query(queryContext, &phaseQuery.header) != FFX_API_RETURN_OK) || (phaseCount <= 0)) {
                const float ratio = float(displayWidth) / float(std::max(renderWidth, 1));
                phaseCount = int32_t(std::ceil(8.0f * ratio * ratio));
            }

            return phaseCount;
        }

        void getJitterOffset(int index, int phaseCount, float &x, float &y) override {
            ffxQueryDescUpscaleGetJitterOffset offsetQuery = {};
            offsetQuery.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
            offsetQuery.index = index;
            offsetQuery.phaseCount = std::max(phaseCount, 1);
            offsetQuery.pOutX = &x;
            offsetQuery.pOutY = &y;
            ffxContext *queryContext = (context != nullptr) ? &context : nullptr;
            if (query(queryContext, &offsetQuery.header) != FFX_API_RETURN_OK) {
                Upscaler::getJitterOffset(index, phaseCount, x, y);
            }
        }

        void upscale(RenderWorker *worker, const UpscaleParameters &p) override {
            if (failed || (context == nullptr)) {
                return;
            }

            plume::D3D12CommandList *commandList = static_cast<plume::D3D12CommandList *>(worker->commandList.get());
            ffxDispatchDescUpscale dispatchDesc = {};
            dispatchDesc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
            dispatchDesc.commandList = commandList->d3d;
            dispatchDesc.color = ffxResource(p.inColor);
            dispatchDesc.depth = ffxResource(p.inDepth);
            dispatchDesc.motionVectors = ffxResource(p.inMotionVectors);
            dispatchDesc.reactive = ffxResource(p.inReactiveMask);
            dispatchDesc.output = ffxResource(p.outColor);
            dispatchDesc.jitterOffset = { p.jitterX, p.jitterY };
            dispatchDesc.motionVectorScale = { float(p.inRect.w), float(p.inRect.h) };
            dispatchDesc.renderSize = { uint32_t(p.inRect.w), uint32_t(p.inRect.h) };
            dispatchDesc.upscaleSize = { p.outWidth, p.outHeight };
            dispatchDesc.enableSharpening = (p.sharpness > 0.0f);
            dispatchDesc.sharpness = std::clamp(p.sharpness, 0.0f, 1.0f);
            dispatchDesc.frameTimeDelta = p.deltaTime;
            dispatchDesc.preExposure = 1.0f;
            dispatchDesc.reset = p.resetAccumulation;
            dispatchDesc.cameraNear = p.nearPlane;
            dispatchDesc.cameraFar = p.farPlane;
            dispatchDesc.cameraFovAngleVertical = p.fovY;
            // The game's world units are roughly centimeters.
            dispatchDesc.viewSpaceToMetersFactor = 0.01f;
            dispatchDesc.flags = 0;

            const ffxReturnCode_t result = dispatch(&context, &dispatchDesc.header);

            // The SDK binds its own descriptor heaps, root signature and pipeline on the command list.
            commandList->notifyDescriptorHeapWasChangedExternally();
            commandList->activeComputePipelineLayout = nullptr;
            commandList->activeGraphicsPipelineLayout = nullptr;

            if (result != FFX_API_RETURN_OK) {
                rtDiagLog("FSR 3: ffxDispatch failed with code %u, falling back to FSR 1", result);
                failed = true;
            }
        }

        bool isInitialized() const override {
            return !failed;
        }

        bool requiresNonShaderResourceInputs() const override {
            return false;
        }

        bool isTemporal() const override {
            return true;
        }

        const char *getName() const override {
            return providerName.empty() ? "AMD FSR 3" : providerName.c_str();
        }
    };

    std::unique_ptr<Upscaler> createFSR3Upscaler(RenderDevice *device, RenderShaderFormat shaderFormat, std::string &failureReason) {
        if (shaderFormat != RenderShaderFormat::DXIL) {
            failureReason = "FSR 3 needs the D3D12 graphics API for now";
            return nullptr;
        }

        std::unique_ptr<FSR3Upscaler> upscaler = std::make_unique<FSR3Upscaler>();
        upscaler->device = device;
        if (!upscaler->load(failureReason)) {
            return nullptr;
        }

        return upscaler;
    }
};

#else

namespace RT64 {
    std::unique_ptr<Upscaler> createFSR3Upscaler(RenderDevice *device, RenderShaderFormat shaderFormat, std::string &failureReason) {
        failureReason = "this build doesn't include FSR 3 (RT64_FSR3 is off)";
        return nullptr;
    }
};

#endif
