#pragma once

#include <cstddef>
#include <vector>
#include <filesystem>
#include <memory>
#include <string>

#include "Mesh.h"
#include "Core/Buffer.h"
#include "Core/RenderGraph.h"
#include "Core/Texture.h"

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

class Scene {
public:
    void LoadGltf(std::filesystem::path path);
    void CommitToGPU(MTL::Device* device, CommandBufferPool& commandBufferPool, MTL4::CommandQueue* queue);
    void RegisterBuffers(RenderGraph& graph);

    const std::vector<Texture>& GetTextures() const { return m_textures; }

    std::vector<Vertex> opaqueVertices;
    std::vector<uint32_t> opaqueIndices;
    std::vector<SubMesh> opaqueSubmeshes;
    std::vector<RenderPrimitive> opaqueRenderPrimitives;

    std::vector<Vertex> transparentVertices;
    std::vector<uint32_t> transparentIndices;
    std::vector<SubMesh> transparentSubmeshes;
    std::vector<RenderPrimitive> transparentRenderPrimitives;

    std::vector<RenderObject> objects;
    std::vector<SceneMaterial> materials;
    std::vector<DirectionalLight> directionalLights;
    AmbientLight ambientLight;

private:
    struct MeshRange
    {
        uint32_t firstSubmesh;
        uint32_t submeshCount;
    };

    struct MeshRanges
    {
        MeshRange opaqueRange;
        MeshRange transparentRange;
    };

    struct TextureSource
    {
        uint32_t imageIndex;
        bool sRGB;
        std::string name;
    };

    static MeshRange MergeMesh(Mesh&& mesh, std::vector<Vertex>& targetVertices, std::vector<uint32_t>& targetIndices, std::vector<SubMesh>& targetSubmeshes);

    std::unique_ptr<Buffer> m_opaqueVertexBuffer;
    std::unique_ptr<Buffer> m_opaqueIndexBuffer;
    std::unique_ptr<Buffer> m_opaqueRenderPrimitiveBuffer;
    std::unique_ptr<Buffer> m_opaqueIndexBufferInfoBuffer;

    std::unique_ptr<Buffer> m_transparentVertexBuffer;
    std::unique_ptr<Buffer> m_transparentIndexBuffer;
    std::unique_ptr<Buffer> m_transparentRenderPrimitiveBuffer;
    std::unique_ptr<Buffer> m_transparentIndexBufferInfoBuffer;

    std::unique_ptr<Buffer> m_materialBuffer;
    std::unique_ptr<Buffer> m_lightListInfoBuffer;
    std::unique_ptr<Buffer> m_directionalLightBuffer;

    std::vector<std::vector<std::byte>> m_encodedImages;
    std::vector<TextureSource> m_textureSources;
    std::vector<Texture> m_textures;
};
