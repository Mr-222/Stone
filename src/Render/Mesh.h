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
    uint32_t materialIndex;
};

struct Mesh
{
    glm::mat4 transform;
    std::unique_ptr<std::vector<Vertex>> vertices;
    std::unique_ptr<std::vector<uint16_t>> indices;
    std::vector<SubMesh> submeshes;

    Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Mesh(Mesh&&) noexcept = default;
    Mesh& operator=(Mesh&&) noexcept = default;
};

std::vector<Mesh> LoadGltf(std::filesystem::path path);