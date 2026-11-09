//
// RT64
//
// Path traced materials for Advanced Lighting.
//
// Hit shaders evaluate the game's real material with the raster shader's own code: the draw's RDP tiles are
// sampled from the texture cache (sets 1 and 2), and the color combiner and alpha compare run on the result.
// Alpha tested and blended materials are non-opaque in the acceleration structure; their any-hit shaders
// reject transparent texels for camera, bounce and shadow rays.
//
// Lighting: one distant point light (sun, moon or room light) with ray traced shadows, plus one diffuse bounce.
// The N64 vertex shade is kept inside the albedo (it is the game's baked lighting and painted color), so the
// light factor is normalized to about 1 on fully lit surfaces to stay close to the original brightness.

#include "shared/rt64_blender.h"
#include "shared/rt64_color_combiner.h"

#include "BlueNoise.hlsli"
#include "Color.hlsli"
#include "FbRendererCommon.hlsli"
#include "Random.hlsli"
#include "TextureSampler.hlsli"
#include "shaders/FbRendererRT.hlsli"

// Set to true to light the rasterized frame instead of shading path traced materials (the earlier hybrid mode).
static const bool kHybridLighting = false;

// Light balance. The light factor is direct + (1 - direct) * fill (see ComposePS): 1 on fully lit surfaces, and
// shadowed surfaces keep close to the scene's shadow tint, lifted a little by the colored light bounced off the
// surface a diffuse ray hits (or the sky).
static const float kFillBounce = 0.35;
static const float kSkyBounce = 0.55;
static const float kLightColorTint = 0.25;
static const float kBounceConeSpread = 0.3;

struct HitPayload {
    float px, py, pz;
    float nx, ny, nz;
    float ar, ag, ab;
    float hitT;
    float hit;
    float renderIndex;
    float fr, fg, fb, fogFactor;
};

struct ShadowPayload {
    float shadowed;
    // Render index of the surface being lit. Its own draw does not shadow it: the light facing proxy of a billboard
    // would otherwise cut through the sprite it stands in for.
    float selfRenderIndex;
};

// Instance masks, see RtCameraRayMask and RtShadowRayMask. Camera and bounce rays see the drawn geometry; shadow
// rays see the shadow casters, where camera facing billboards are replaced by light facing proxies.
static const uint kCameraRayMask = 0x1;
static const uint kShadowRayMask = 0x2;

// Diagnostics counters in gHitInstanceId, read back by RaytracingResources::readDiagnosticBuffers.
static const uint kDiagInBounds = 0;
static const uint kDiagOutsideViewport = 1;
static const uint kDiagTraced = 2;
static const uint kDiagHits = 3;
static const uint kDiagBadDirection = 4;
static const uint kDiagBadOrigin = 5;
static const uint kDiagSurfaceIgnores = 6;
static const uint kDiagShadowIgnores = 7;
static const uint kDiagMarker = 8;
static const uint kDiagCompared = 9;
static const uint kDiagMismatched = 10;
static const uint kDiagRecordCount = 11;
static const uint kDiagMaxRecords = 5;
static const uint kDiagShadowRays = 12;
static const uint kDiagShadowBlocked = 13;
static const uint kDiagCulledHits = 14;
static const uint kDiagTieBreaks = 15;

// Surfaces closer than this to the camera ray's closest hit count as coplanar layers.
static const float kTieBreakMinWindow = 0.02;
static const float kTieBreakRelativeWindow = 5e-4;

// Instance ID layout, see RaytracingResources::updateTopLevelASResources.
static const uint kInstanceRenderIndexMask = 0x1FFFFF;
static const uint kInstanceProxyBit = 1u << 21;
static const uint kInstanceCullingBit = 1u << 22;
static const uint kInstanceAlphaTestBit = 1u << 23;

uint hitRenderIndex() {
    return InstanceID() & kInstanceRenderIndexMask;
}

uint2 rayPixel() {
    return DispatchRaysIndex().xy;
}

bool pixelInBounds(uint2 pix) {
    return (pix.x < uint(RtParams.resolution.x)) && (pix.y < uint(RtParams.resolution.y));
}

bool pixelInViewport(uint2 pix) {
    if ((RtParams.viewport.z < 1.0) || (RtParams.viewport.w < 1.0)) {
        return true;
    }

    float2 p = float2(pix) + 0.5;
    return (p.x >= RtParams.viewport.x) && (p.y >= RtParams.viewport.y) &&
        (p.x < (RtParams.viewport.x + RtParams.viewport.z)) &&
        (p.y < (RtParams.viewport.y + RtParams.viewport.w));
}

bool isDiagPixel(uint2 pix) {
    return all(pix == uint2(RtParams.resolution.xy * 0.5));
}

// cameraV.w is 2 on diagnostics capture frames.
bool diagFrame() {
    return RtParams.cameraV.w > 1.5;
}

void diagCount(uint slot, bool condition) {
    uint count = WaveActiveCountBits(condition);
    if (WaveIsFirstLane() && (count > 0)) {
        InterlockedAdd(gHitInstanceId[slot], count);
    }
}

bool isFinite3(float3 v) {
    return all(abs(v) < 1e30);
}

// The rasterized frame as displayed (sky, background and anything the ray tracer does not shade).
float3 rasterColor(uint2 pix) {
    return gBackgroundColor.Load(int3(pix, 0)).rgb;
}

float3 loadPos(uint index) {
    uint byteOffset = index * 16;
    // Load3 crashes the D3D12 driver compiler when this shader is linked into a hit group.
    return float3(
        asfloat(posBuffer.Load(byteOffset)),
        asfloat(posBuffer.Load(byteOffset + 4)),
        asfloat(posBuffer.Load(byteOffset + 8)));
}

float2 loadTexcoord(uint index) {
    uint byteOffset = index * 8;
    return float2(asfloat(genTexCoordBuffer.Load(byteOffset)), asfloat(genTexCoordBuffer.Load(byteOffset + 4)));
}

float4 loadColor(uint index) {
    uint byteOffset = index * 16;
    return float4(
        asfloat(shadedColBuffer.Load(byteOffset)),
        asfloat(shadedColBuffer.Load(byteOffset + 4)),
        asfloat(shadedColBuffer.Load(byteOffset + 8)),
        asfloat(shadedColBuffer.Load(byteOffset + 12)));
}

struct Material {
    float4 color;
    float3 fogColor;
    float fogFactor;
    bool visible;
};

// The raster pixel shader's material path (RasterPS / ShadowCutoutPS) for one ray hit. There are no screen space
// derivatives in ray tracing, so the texture footprint comes from a ray cone: footprintWorld is the width of the
// cone at the hit, divided by the cosine to the surface by the caller.
Material evalMaterial(uint renderIndex, uint primitiveIndex, float2 barycentrics, float footprintWorld) {
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const uint instanceIndex = ri.instanceIndex;
    const RenderParams rp = DynamicRenderParams[instanceIndex];
    const OtherMode otherMode = { rp.omL, rp.omH };
    const ColorCombiner colorCombiner = { rp.ccL, rp.ccH };
    const RDPParams rdp = instanceRDPParams[instanceIndex];

    const uint base = ri.faceIndicesStart + primitiveIndex * 3;
    const uint i0 = indexBuffer.Load(base * 4);
    const uint i1 = indexBuffer.Load((base + 1) * 4);
    const uint i2 = indexBuffer.Load((base + 2) * 4);
    const float w0 = saturate(1.0 - barycentrics.x - barycentrics.y);
    const float2 uv0 = loadTexcoord(i0);
    const float2 uv1 = loadTexcoord(i1);
    const float2 uv2 = loadTexcoord(i2);
    const float2 uv = uv0 * w0 + uv1 * barycentrics.x + uv2 * barycentrics.y;
    const float4 smoothColor = loadColor(i0) * w0 + loadColor(i1) * barycentrics.x + loadColor(i2) * barycentrics.y;
    // Flat shading takes the color of the first vertex, like the raster pipelines' provoking vertex.
    const float4 flatColor = loadColor(i0);
    const float4 shadeColor = renderFlagSmoothShade(rp.flags) ? smoothColor : float4(flatColor.rgb, smoothColor.a);

    // Texels per world unit on this triangle, from the ratio of its texture and world areas.
    const float3 p0 = loadPos(i0);
    const float worldArea = length(cross(loadPos(i1) - p0, loadPos(i2) - p0));
    const float2 duv1 = uv1 - uv0;
    const float2 duv2 = uv2 - uv0;
    const float texelArea = abs(duv1.x * duv2.y - duv1.y * duv2.x);
    const float texelsPerWorld = sqrt(texelArea / max(worldArea, 1e-12));
    const float footprintTexels = max(footprintWorld * texelsPerWorld, 1.0 / 1024.0);

    int tileIndex0 = 0;
    int tileIndex1 = 1;
    float lodFraction;
    computeLOD(otherMode, ri.rdpTileCount, rdp.primLOD, 1.0, footprintTexels, footprintTexels, tileIndex0, tileIndex1, lodFraction);
    const float2 ddxUV = float2(footprintTexels, 0.0);
    const float2 ddyUV = float2(0.0, footprintTexels);

    float4 texVal0 = float4(0.0, 0.0, 0.0, 1.0);
    float4 texVal1 = float4(0.0, 0.0, 0.0, 1.0);
    if (renderFlagUsesTexture0(rp.flags)) {
        const uint globalTileIndex = ri.rdpTileIndex + tileIndex0;
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = renderCMS0(rp.flags);
            rdpTile.cmt = renderCMT0(rp.flags);
            rdpTile.nativeSampler = renderFlagNativeSampler0(rp.flags);
        }

        texVal0 = sampleTexture(otherMode, rp.flags, uv, ddxUV, ddyUV, rdpTile, GPUTiles[globalTileIndex], false);
    }

    if (renderFlagUsesTexture1(rp.flags)) {
        const bool oneCycleHardwareBug = (otherMode.cycleType() == G_CYC_1CYCLE);
        const uint globalTileIndex = ri.rdpTileIndex + (oneCycleHardwareBug ? tileIndex0 : tileIndex1);
        RDPTile rdpTile = RDPTiles[globalTileIndex];
        if (!renderFlagDynamicTiles(rp.flags)) {
            rdpTile.cms = oneCycleHardwareBug ? renderCMS0(rp.flags) : renderCMS1(rp.flags);
            rdpTile.cmt = oneCycleHardwareBug ? renderCMT0(rp.flags) : renderCMT1(rp.flags);
            rdpTile.nativeSampler = oneCycleHardwareBug ? renderFlagNativeSampler0(rp.flags) : renderFlagNativeSampler1(rp.flags);
        }

        texVal1 = sampleTexture(otherMode, rp.flags, uv, ddxUV, ddyUV, rdpTile, GPUTiles[globalTileIndex], oneCycleHardwareBug);
    }

    uint randomSeed = initRand(FrParams.frameCount, renderIndex * 7919 + primitiveIndex, 16);
    float4 combinerColor;
    float alphaCompareValue;
    ColorCombiner::Inputs ccInputs;
    ccInputs.otherMode = otherMode;
    ccInputs.alphaOnly = false;
    ccInputs.texVal0 = texVal0;
    ccInputs.texVal1 = texVal1;
    ccInputs.primColor = rdp.primColor;
    ccInputs.shadeColor = shadeColor;
    ccInputs.envColor = rdp.envColor;
    ccInputs.keyCenter = rdp.keyCenter;
    ccInputs.keyScale = rdp.keyScale;
    ccInputs.lodFraction = lodFraction;
    ccInputs.primLodFrac = rdp.primLOD.x;
    ccInputs.noise = nextRand(randomSeed);
    ccInputs.K4 = (rdp.convertK[4] / 255.0);
    ccInputs.K5 = (rdp.convertK[5] / 255.0);
    colorCombiner.run(ccInputs, combinerColor, alphaCompareValue);

    // Run the blender like the raster pipeline, with its standard fog cycle disabled. Fog is applied after
    // lighting in ComposePS so lit and shadowed surfaces fade to the same fog color.
    Blender::Inputs blInputs;
    blInputs.blendColor = rdp.blendColor;
    blInputs.fogColor = rdp.fogColor;
    blInputs.shadeAlpha = shadeColor.a;

    Material material;
    material.color = float4(Blender::run(otherMode, rp.flags, blInputs, combinerColor, true).rgb, combinerColor.a);
    material.visible = true;
    if ((otherMode.alphaCompare() == G_AC_THRESHOLD) && (alphaCompareValue < rdp.blendColor.a)) {
        material.visible = false;
    }
    else if ((otherMode.alphaCompare() == G_AC_DITHER) && (alphaCompareValue < 0.5)) {
        material.visible = false;
    }
    else if (otherMode.cvgXAlpha() && (combinerColor.a < 0.5)) {
        material.visible = false;
    }
    // Blended surfaces in the acceleration structure are treated as cutouts.
    else if (Blender::usesAlphaBlend(otherMode) && (combinerColor.a < 0.5)) {
        material.visible = false;
    }

    material.fogColor = rdp.fogColor.rgb;
    material.fogFactor = Blender::usesStandardFogCycle(otherMode) ? saturate(shadeColor.a) : 0.0;
    return material;
}

// Ray cone inputs: before tracing, payload.hitT holds the cone width at the ray origin and payload.fogFactor its
// spread angle. The closest hit shader overwrites both.
float coneFootprint(float originWidth, float spreadAngle) {
    return originWidth + RayTCurrent() * spreadAngle;
}

// True when the rasterizer culls this triangle. RT64 culls the triangles of draws with any N64 cull mode that
// appear clockwise on screen. The triangle is projected with the raster's own view-projection; on screen, up is
// +clip Y when the viewport Y scale is positive (RtParams.nearDist carries that sign).
bool rasterCulls(uint renderIndex, uint primitiveIndex) {
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const uint base = ri.faceIndicesStart + primitiveIndex * 3;
    const float4 c0 = mul(RtParams.viewProj, float4(loadPos(indexBuffer.Load(base * 4)), 1.0));
    const float4 c1 = mul(RtParams.viewProj, float4(loadPos(indexBuffer.Load((base + 1) * 4)), 1.0));
    const float4 c2 = mul(RtParams.viewProj, float4(loadPos(indexBuffer.Load((base + 2) * 4)), 1.0));
    // Triangles crossing the camera plane are clipped by the rasterizer; leave them to the depth test.
    if ((c0.w <= 1e-5) || (c1.w <= 1e-5) || (c2.w <= 1e-5)) {
        return false;
    }

    const float2 s0 = c0.xy / c0.w;
    const float2 s1 = c1.xy / c1.w;
    const float2 s2 = c2.xy / c2.w;
    const float2 e1 = s1 - s0;
    const float2 e2 = s2 - s0;
    const float signedArea = e1.x * e2.y - e1.y * e2.x;
    return (signedArea * RtParams.nearDist) < 0.0;
}

// Camera rays start with payload.hit = 2 so the any-hit shader can apply the raster's face culling to them only.
// The camera ray's coplanar tie-break starts with payload.hit = 3: every visible surface in the short ray interval
// is enumerated, and the one with the highest render index (drawn last, so on top in the raster) is recorded in
// px (primitive index), py/pz (barycentrics) and nx (hit distance).
[shader("anyhit")]
void SurfaceAnyHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    const uint instanceID = InstanceID();
    const uint renderIndex = instanceID & kInstanceRenderIndexMask;
    if ((payload.hit > 1.5) && ((instanceID & kInstanceCullingBit) != 0) && rasterCulls(renderIndex, PrimitiveIndex())) {
        if (diagFrame()) {
            InterlockedAdd(gHitInstanceId[kDiagCulledHits], 1);
        }
        IgnoreHit();
    }

    if ((instanceID & kInstanceAlphaTestBit) != 0) {
        Material material = evalMaterial(renderIndex, PrimitiveIndex(), attr.barycentrics, coneFootprint(payload.hitT, payload.fogFactor));
        if (!material.visible) {
            if (diagFrame()) {
                InterlockedAdd(gHitInstanceId[kDiagSurfaceIgnores], 1);
            }
            IgnoreHit();
        }
    }

    if (payload.hit > 2.5) {
        if (float(renderIndex) > payload.renderIndex) {
            payload.renderIndex = float(renderIndex);
            payload.px = asfloat(PrimitiveIndex());
            payload.py = attr.barycentrics.x;
            payload.pz = attr.barycentrics.y;
            payload.nx = RayTCurrent();
        }

        IgnoreHit();
    }
}

[shader("anyhit")]
void ShadowAnyHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    // A billboard proxy never shadows the sprite it stands in for.
    if (((InstanceID() & kInstanceProxyBit) != 0) && (float(hitRenderIndex()) == payload.selfRenderIndex)) {
        IgnoreHit();
    }

    // Shadows ignore face culling; only alpha tested materials can let light through.
    if ((InstanceID() & kInstanceAlphaTestBit) == 0) {
        payload.shadowed = 1.0;
        AcceptHitAndEndSearch();
    }

    // Shadow rays carry no cone; a coarse footprint is enough for an alpha test.
    Material material = evalMaterial(hitRenderIndex(), PrimitiveIndex(), attr.barycentrics, RayTCurrent() * RtParams.pixelSpreadAngle * 4.0);
    if (!material.visible) {
        if (diagFrame()) {
            InterlockedAdd(gHitInstanceId[kDiagShadowIgnores], 1);
        }
        IgnoreHit();
    }
    else {
        payload.shadowed = 1.0;
        AcceptHitAndEndSearch();
    }
}

// Fills the payload for a hit on a triangle of a draw. Used by the closest hit shader and by the camera ray's
// coplanar tie-break in PrimaryRayGen.
void shadeSurface(uint renderIndex, uint primitiveIndex, float2 barycentrics, float3 rayOrigin, float3 rayDirection, float hitT, float coneWidth, inout HitPayload payload) {
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const uint base = ri.faceIndicesStart + primitiveIndex * 3;
    const float3 p0 = loadPos(indexBuffer.Load(base * 4));
    const float3 p1 = loadPos(indexBuffer.Load((base + 1) * 4));
    const float3 p2 = loadPos(indexBuffer.Load((base + 2) * 4));

    // Geometric normal facing the ray. Unlit N64 geometry stores vertex colors where lit geometry stores normals.
    float3 normal = cross(p1 - p0, p2 - p0);
    normal = (dot(normal, normal) > 1e-12) ? normalize(normal) : -rayDirection;
    if (dot(normal, rayDirection) > 0.0) {
        normal = -normal;
    }

    // Grazing angles stretch the cone footprint across the surface.
    const float footprint = coneWidth / max(abs(dot(normal, rayDirection)), 0.2);
    Material material = evalMaterial(renderIndex, primitiveIndex, barycentrics, footprint);
    const float3 position = rayOrigin + rayDirection * hitT;
    payload.px = position.x;
    payload.py = position.y;
    payload.pz = position.z;
    payload.nx = normal.x;
    payload.ny = normal.y;
    payload.nz = normal.z;
    payload.ar = saturate(material.color.r);
    payload.ag = saturate(material.color.g);
    payload.ab = saturate(material.color.b);
    payload.hitT = hitT;
    payload.hit = 1.0;
    payload.renderIndex = float(renderIndex);
    payload.fr = material.fogColor.r;
    payload.fg = material.fogColor.g;
    payload.fb = material.fogColor.b;
    payload.fogFactor = material.fogFactor;
}

[shader("closesthit")]
void SurfaceClosestHit(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    const float coneWidth = coneFootprint(payload.hitT, payload.fogFactor);
    shadeSurface(hitRenderIndex(), PrimitiveIndex(), attr.barycentrics, WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), coneWidth, payload);
}

[shader("miss")]
void PrimaryMiss(inout HitPayload payload) {
    payload.hit = 0.0;
}

[shader("miss")]
void ShadowMiss(inout ShadowPayload payload) {
    payload.shadowed = 0.0;
}

HitPayload emptyPayload() {
    HitPayload payload = (HitPayload)0;
    payload.renderIndex = -1.0;
    return payload;
}

HitPayload traceSurface(float3 origin, float3 direction, float tMin, float tMax, float coneWidth, float coneSpread, bool cameraRay, uint rayMask) {
    HitPayload payload = emptyPayload();
    payload.hitT = coneWidth;
    payload.fogFactor = coneSpread;
    payload.hit = cameraRay ? 2.0 : 0.0;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = tMin;
    ray.TMax = tMax;
    TraceRay(SceneBVH, RAY_FLAG_NONE, rayMask, 0, 0, 0, ray, payload);
    return payload;
}

// Opaque geometry ends the search at once; cutout geometry runs ShadowAnyHit.
float traceShadow(float3 origin, float3 direction, float distance, float selfRenderIndex) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = max(distance, 1.0);

    ShadowPayload payload;
    payload.shadowed = 0.0;
    payload.selfRenderIndex = selfRenderIndex;
    TraceRay(SceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kShadowRayMask, 1, 0, 1, ray, payload);
    return payload.shadowed;
}

// Offset along the normal that grows with distance, so float precision far from the camera does not make a
// surface shadow itself.
float surfaceBias(float hitT) {
    return max(0.15, hitT * 0.004);
}

struct LightSample {
    float3 factor;
    float ndotl;
    float visibility;
    float distance;
    float3 direction;
};

// Direct light factor, using the same model as the shadow map pass:
// lerp(facingFloor, 1, |NdotL|) * lerp(shadowTint, 1, visibility), tinted slightly by the light color.
// Like the shadow map pass, a surface facing away from the light is not dark by itself: single sided N64 walls are
// often seen from their unlit side. The shadow ray starts on the side the light comes from, so the back of a closed
// mesh still finds the mesh in the way.
LightSample sampleLight(float3 position, float3 normal, float hitT, float selfRenderIndex) {
    LightSample result;
    result.factor = 1.0;
    result.ndotl = 1.0;
    result.visibility = 1.0;
    result.distance = 0.0;
    result.direction = float3(0.0, 1.0, 0.0);
    if (RtParams.lightsCount == 0) {
        return result;
    }

    PointLight light = SceneLights[0];
    float3 toLight = light.position - position;
    result.distance = length(toLight);
    result.direction = (result.distance > 1e-4) ? (toLight / result.distance) : float3(0.0, 1.0, 0.0);
    result.ndotl = dot(normal, result.direction);
    const float3 lightSideNormal = (result.ndotl >= 0.0) ? normal : -normal;
    const float bias = surfaceBias(hitT);
    result.visibility = 1.0 - traceShadow(position + lightSideNormal * bias, result.direction, result.distance, selfRenderIndex);

    float3 shadowTint = RtParams.cameraW.xyz;
    float facingFloor = RtParams.cameraW.w;
    float maxColor = max(max(light.diffuseColor.r, light.diffuseColor.g), max(light.diffuseColor.b, 1e-4));
    float3 lightTint = lerp(float3(1.0, 1.0, 1.0), light.diffuseColor / maxColor, kLightColorTint);
    result.factor = lerp(facingFloor.xxx, float3(1.0, 1.0, 1.0), saturate(abs(result.ndotl))) * lerp(shadowTint, float3(1.0, 1.0, 1.0), result.visibility);
    result.factor *= lerp(float3(1.0, 1.0, 1.0), lightTint, result.visibility);
    return result;
}

void writeDiagnostics(float3 origin, float3 direction, HitPayload payload) {
    gHitInstanceId[kDiagMarker] = 0xC0FFEE;
    gHitColor[0] = float4(origin, 1.0);
    gHitColor[1] = float4(direction, payload.hitT);
    gHitColor[2] = float4(payload.px, payload.py, payload.pz, payload.hit);
    gHitColor[3] = RtParams.viewport;
    gHitColor[4] = RtParams.resolution;
    gHitColor[5] = float4(RtParams.nearDist, float(RtParams.lightsCount), RtParams.farDist, payload.renderIndex);
    gHitColor[9] = float4(payload.ar, payload.ag, payload.ab, payload.hit);
    gHitColor[10] = float4(payload.fr, payload.fg, payload.fb, payload.fogFactor);
    if (payload.renderIndex >= 0.0) {
        const RenderIndices ri = instanceRenderIndices[uint(payload.renderIndex)];
        const RenderParams rp = DynamicRenderParams[ri.instanceIndex];
        const OtherMode otherMode = { rp.omL, rp.omH };
        gHitColor[11] = float4(float(ri.instanceIndex), renderFlagUsesTexture0(rp.flags) ? 1.0 : 0.0, renderFlagUsesTexture1(rp.flags) ? 1.0 : 0.0, float(otherMode.alphaCompare()));
        gHitColor[12] = float4(float(ri.rdpTileIndex), float(ri.rdpTileCount), otherMode.cvgXAlpha() ? 1.0 : 0.0, Blender::usesAlphaBlend(otherMode) ? 1.0 : 0.0);
    }
    else {
        gHitColor[11] = -1.0;
        gHitColor[12] = -1.0;
    }

}

uint packColor(float3 color) {
    uint3 c = uint3(saturate(color) * 255.0 + 0.5);
    return c.r | (c.g << 8) | (c.b << 16);
}

// Compares the path traced material with the rasterized pixel of the same surface. Unfogged opaque surfaces should
// match closely; records of a few mismatching pixels (draw, combiner, render mode, first tile) go to the log.
void diagCompareMaterial(uint2 pix, HitPayload payload, float3 raster) {
    const bool comparable = (payload.hit > 0.5) && (payload.fogFactor < 0.05) && (payload.renderIndex >= 0.0);
    const float3 albedo = float3(payload.ar, payload.ag, payload.ab);
    const float3 delta = abs(albedo - raster);
    const bool mismatch = comparable && (max(delta.r, max(delta.g, delta.b)) > 0.2);
    diagCount(kDiagCompared, comparable);
    diagCount(kDiagMismatched, mismatch);
    if (!mismatch || ((pix.x % 16) != 8) || ((pix.y % 16) != 8)) {
        return;
    }

    uint record;
    InterlockedAdd(gHitInstanceId[kDiagRecordCount], 1, record);
    if (record >= kDiagMaxRecords) {
        return;
    }

    const uint renderIndex = uint(payload.renderIndex);
    const RenderIndices ri = instanceRenderIndices[renderIndex];
    const RenderParams rp = DynamicRenderParams[ri.instanceIndex];
    const RDPTile tile = RDPTiles[ri.rdpTileIndex];
    const GPUTile gpuTile = GPUTiles[ri.rdpTileIndex];
    const uint tileBits = uint(tile.fmt & 0xF) | (uint(tile.siz & 0xF) << 4) | (uint(tile.palette & 0xF) << 8) | (uint(tile.cms & 0xF) << 12) | (uint(tile.cmt & 0xF) << 16) | ((ri.rdpTileCount & 0xF) << 20);
    gHitVelocityDistance[record * 3 + 0] = float4(float(renderIndex), asfloat(rp.ccL), asfloat(rp.ccH), asfloat(rp.omL));
    gHitVelocityDistance[record * 3 + 1] = float4(asfloat(rp.omH), asfloat(uint(rp.flags)), asfloat(packColor(albedo)), asfloat(packColor(raster)));
    gHitVelocityDistance[record * 3 + 2] = float4(asfloat(tileBits), asfloat(uint(gpuTile.flags)), asfloat(gpuTile.textureIndex), asfloat(pix.x | (pix.y << 16)));
}

// The camera ray through a pixel. viewI is the inverse of the raster view-projection (row-vector math on the CPU,
// so mul(M, v) here). The camera is the point that projects to clip (0, 0, c, 0); the direction goes through the
// point at NDC depth 0, which is in front of the camera for both depth conventions.
bool cameraRay(uint2 pix, out float3 origin, out float3 direction) {
    float2 p = float2(pix) + 0.5 + RtParams.pixelJitter;
    float2 vpSize = max(RtParams.viewport.zw, float2(1.0, 1.0));
    float2 vp = (p - RtParams.viewport.xy) / vpSize;
    float ndcX = vp.x * 2.0 - 1.0;
    // nearDist carries the sign of the viewport Y scale: +1 when the top of the viewport is clip Y = +1.
    float clipY = -lerp(-RtParams.nearDist, RtParams.nearDist, vp.y);
    float4 midH = mul(RtParams.viewI, float4(ndcX, clipY, 0.0, 1.0));
    float3 midPos = midH.xyz / midH.w;
    float4 originH = mul(RtParams.viewI, float4(0.0, 0.0, 1.0, 0.0));
    if (abs(originH.w) > 1e-12) {
        origin = originH.xyz / originH.w;
    }
    else {
        float4 nearH = mul(RtParams.viewI, float4(ndcX, clipY, -1.0, 1.0));
        origin = nearH.xyz / nearH.w;
    }

    direction = midPos - origin;
    float len2 = dot(direction, direction);
    diagCount(kDiagBadOrigin, !isFinite3(origin));
    if (!isFinite3(origin) || !isFinite3(direction) || (len2 <= 1e-12)) {
        return false;
    }

    direction *= rsqrt(len2);
    return true;
}

void writeMiss(uint2 pix, float3 bg) {
    gDiffuse[pix] = float4(bg, 0.0);
    gShadingPosition[pix] = 0.0;
    gShadingNormal[pix] = 0.0;
    gShadingSpecular[pix] = 0.0;
    gViewDirection[pix] = 0.0;
    gDepth[pix] = 0.0;
    gInstanceId[pix] = -1;
    gNormalRoughness[pix] = 0.0;
}

[shader("raygeneration")]
void PrimaryRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    diagCount(kDiagInBounds, true);
    float3 bg = rasterColor(pix);
    gFlow[pix] = 0.0;
    gTransparent[pix] = 0.0;
    gReflection[pix] = 0.0;
    gRefraction[pix] = 0.0;
    gDirectLightAccum[pix] = 0.0;
    gIndirectLightAccum[pix] = 0.0;

    const bool insideViewport = pixelInViewport(pix);
    diagCount(kDiagOutsideViewport, !insideViewport);
    float3 origin, direction;
    const bool validRay = insideViewport && cameraRay(pix, origin, direction);
    diagCount(kDiagBadDirection, insideViewport && !validRay);
    if (!validRay) {
        writeMiss(pix, bg);
        return;
    }

    HitPayload payload = traceSurface(origin, direction, 1e-2, 1.0e7, 0.0, RtParams.pixelSpreadAngle, true, kCameraRayMask);
    diagCount(kDiagTraced, true);
    diagCount(kDiagHits, payload.hit > 0.5);

    // Coplanar tie-break. Layered character parts (faces, eyes, clothing details) lie on the surface of the base
    // mesh; the raster shows whichever the game drew last, while the closest hit would pick either by float noise and
    // flicker between frames. Enumerate the surfaces just around the hit and keep the latest draw.
    bool tieBreakChanged = false;
    if (payload.hit > 0.5) {
        const float window = max(kTieBreakMinWindow, payload.hitT * kTieBreakRelativeWindow);
        HitPayload tie = emptyPayload();
        tie.hit = 3.0;
        tie.hitT = 0.0;
        tie.fogFactor = RtParams.pixelSpreadAngle;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = direction;
        ray.TMin = max(payload.hitT - window, 1e-2);
        ray.TMax = payload.hitT + window;
        TraceRay(SceneBVH, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, kCameraRayMask, 0, 0, 0, ray, tie);
        if (tie.renderIndex > payload.renderIndex) {
            const uint winner = uint(tie.renderIndex);
            const uint winnerPrimitive = asuint(tie.px);
            const float2 winnerBarycentrics = float2(tie.py, tie.pz);
            const float winnerT = tie.nx;
            shadeSurface(winner, winnerPrimitive, winnerBarycentrics, origin, direction, winnerT, winnerT * RtParams.pixelSpreadAngle, payload);
            tieBreakChanged = true;
        }
    }

    if (diagFrame()) {
        diagCount(kDiagTieBreaks, tieBreakChanged);
    }
    if (isDiagPixel(pix)) {
        writeDiagnostics(origin, direction, payload);
    }

    if (diagFrame()) {
        diagCompareMaterial(pix, payload, bg);
    }

    if (payload.hit > 0.5) {
        float3 normal = float3(payload.nx, payload.ny, payload.nz);
        float3 albedo = kHybridLighting ? bg : float3(payload.ar, payload.ag, payload.ab);
        gDiffuse[pix] = float4(albedo, 1.0);
        gShadingPosition[pix] = float4(payload.px, payload.py, payload.pz, 1.0);
        gShadingNormal[pix] = float4(normal, 0.0);
        gShadingSpecular[pix] = 0.0;
        gViewDirection[pix] = float4(-direction, 0.0);
        gDepth[pix] = payload.hitT;
        gInstanceId[pix] = int(payload.renderIndex);
        gNormalRoughness[pix] = float4(normal * 0.5 + 0.5, 1.0);
        // Fog color and amount of the surface's blender, applied after lighting in ComposePS.
        gTransparent[pix] = kHybridLighting ? float4(0.0, 0.0, 0.0, 0.0) : float4(payload.fr, payload.fg, payload.fb, payload.fogFactor);
    }
    else {
        writeMiss(pix, bg);
    }
}

// Rebuilds the primary hit at full precision from the 32-bit hit distance; the shading position texture is 16-bit.
bool primaryHit(uint2 pix, out float3 position, out float3 normal, out float hitT) {
    position = 0.0;
    normal = float3(0.0, 1.0, 0.0);
    hitT = 0.0;
    float3 origin, direction;
    if ((gDiffuse[pix].a <= EPSILON) || !cameraRay(pix, origin, direction)) {
        return false;
    }

    hitT = gDepth[pix];
    position = origin + direction * hitT;
    normal = gShadingNormal[pix].xyz;
    float normalLen2 = dot(normal, normal);
    normal = (normalLen2 > 1e-8) ? (normal * rsqrt(normalLen2)) : -direction;
    return true;
}

[shader("raygeneration")]
void DirectRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    float3 position, normal;
    float hitT;
    if (!primaryHit(pix, position, normal, hitT)) {
        gDirectLightAccum[pix] = 1.0;
        if (isDiagPixel(pix)) {
            gHitColor[13] = float4(0.0, 0.0, 0.0, 1.0);
        }
        return;
    }

    LightSample light = sampleLight(position, normal, hitT, float(gInstanceId[pix]));
    gDirectLightAccum[pix] = float4(light.factor, 1.0);
    if (diagFrame()) {
        diagCount(kDiagShadowRays, RtParams.lightsCount > 0);
        diagCount(kDiagShadowBlocked, (RtParams.lightsCount > 0) && (light.visibility < 0.5));
    }

    if (isDiagPixel(pix)) {
        gHitColor[13] = float4(light.ndotl, light.visibility, light.distance, light.factor.g);
        // Same ray as the shadow test, but finding the closest blocker.
        const float3 lightSideNormal = (light.ndotl >= 0.0) ? normal : -normal;
        HitPayload blocker = traceSurface(position + lightSideNormal * surfaceBias(hitT), light.direction, 0.0, light.distance, 0.0, RtParams.pixelSpreadAngle, false, kShadowRayMask);
        gHitColor[14] = float4(normal, blocker.renderIndex);
        gHitColor[15] = float4(light.direction, (blocker.hit > 0.5) ? blocker.hitT : -1.0);
    }
}

// One diffuse bounce with the bounce surface's real material and its own shadowed direct light.
[shader("raygeneration")]
void IndirectRayGen() {
    uint2 pix = rayPixel();
    if (!pixelInBounds(pix)) {
        return;
    }

    float3 position, normal;
    float hitT;
    if (kHybridLighting || !primaryHit(pix, position, normal, hitT)) {
        gIndirectLightAccum[pix] = 0.0;
        return;
    }

    // A fixed noise frame keeps the pattern stable between frames; the gaussian filter smooths it.
    float3 bounceDir = getCosHemisphereSampleBlueNoise(gBlueNoise, pix, 17, normal);
    float bias = surfaceBias(hitT);
    // Bounce rays are blurry, so they sample coarse texture levels.
    HitPayload bounce = traceSurface(position + normal * bias, bounceDir, 0.0, 1.0e6, hitT * RtParams.pixelSpreadAngle, kBounceConeSpread, false, kCameraRayMask);
    float3 bounceLight;
    if (bounce.hit > 0.5) {
        float3 bouncePos = float3(bounce.px, bounce.py, bounce.pz);
        float3 bounceNormal = float3(bounce.nx, bounce.ny, bounce.nz);
        LightSample bounceSample = sampleLight(bouncePos, bounceNormal, hitT + bounce.hitT, bounce.renderIndex);
        bounceLight = float3(bounce.ar, bounce.ag, bounce.ab) * bounceSample.factor;
    }
    else {
        bounceLight = kSkyBounce;
    }

    // Fill for the part of the surface the direct light does not reach. ComposePS combines it as
    // direct + (1 - direct) * fill, so fully lit surfaces keep the game's brightness.
    gIndirectLightAccum[pix] = float4(saturate(bounceLight * kFillBounce), 1.0);
}

[shader("raygeneration")]
void ReflectionRayGen() {
    uint2 pix = rayPixel();
    if (pixelInBounds(pix)) {
        gReflection[pix] = 0.0;
    }
}

[shader("raygeneration")]
void RefractionRayGen() {
    uint2 pix = rayPixel();
    if (pixelInBounds(pix)) {
        gRefraction[pix] = 0.0;
    }
}
