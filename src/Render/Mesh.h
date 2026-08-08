#pragma once

#include <cstdint>
#include <vector>
#include <filesystem>
#include <memory>
#include <glm/glm.hpp>

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv;
    glm::vec3 tangent;
};

struct SubMesh
{
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t vertexOffset;
    uint32_t materialIndex;
};

struct SceneMaterial
{
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    uint32_t baseColorTextureIndex = 0;
};

struct Mesh
{
    std::unique_ptr<std::vector<Vertex>> vertices;
    std::unique_ptr<std::vector<uint32_t>> indices;
    std::vector<SubMesh> submeshes;

    Mesh()
        : vertices(std::make_unique<std::vector<Vertex>>())
        , indices(std::make_unique<std::vector<uint32_t>>())
    {
    }

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;
};

struct RenderObject
{
    glm::mat4 transform;
};

struct RenderPrimitive
{
    uint32_t submeshIndex;
    uint32_t objectIndex;
};
