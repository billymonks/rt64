//
// RT64
//

#pragma once

namespace RT64 {
    // Directional light used by the advanced lighting shadow pass. The direction is in the game's world space
    // (before the camera) and points from the surfaces towards the light.
    struct SceneLight {
        float direction[3] = { 0.55f, 0.82f, 0.32f };

        // Color multiplier for a surface fully in shadow.
        float shadowTint[3] = { 0.40f, 0.40f, 0.40f };

        // Brightness multiplier for a surface facing away from the light.
        float facingFloor = 0.74f;

        // Direct color of the ray-traced sun, moon or room light. Already includes intensity.
        float lightColor[3] = { 1.15f, 1.02f, 0.88f };

        // Distance from the world origin to the point light that stands in for this directional light.
        float lightDistance = 50000.0f;

        // When false, the shadow pass is skipped for this frame.
        bool enabled = true;
    };
};
