#pragma once

#include <glm/glm.hpp>

struct DirectionalLight
{
    // Direction in which the light rays travel. The shader uses its negation
    // as the direction from the shaded point toward the light.
    glm::vec3 direction;
    glm::vec3 color;
    float illuminance;
};

struct AmbientLight
{
    glm::vec3 color;
    float intensity;
};

struct PointLight
{
    glm::vec3 position;
    glm::vec3 color;
    float intensity; // luminous intensity in candela
    float range;
};

struct SpotLight
{
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float intensity; // luminous intensity in candela
    float range;
    float innerAngle; // radians
    float outerAngle; // radians
};
