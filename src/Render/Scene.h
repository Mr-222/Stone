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

class Scene {
public:
    void LoadGltf(std::filesystem::path path);
    void CommitToGPU(MTL::Device* device, CommandBufferPool& commandBufferPool, MTL4::CommandQueue* queue);
    void RegisterBuffers(RenderGraph& graph);

    const std::vector<Texture>& GetTextures() const { return m_textures; }

    std::vector<Vertex> globalVertices;
    std::vector<uint32_t> globalIndices;
    std::vector<SubMesh> submeshes;
    std::vector<RenderObject> objects;
    std::vector<RenderPrimitive> renderPrimitives;
    std::vector<SceneMaterial> materials;

private:
    struct MeshRange
    {
        uint32_t firstSubmesh;
        uint32_t submeshCount;
    };

    struct TextureSource
    {
        uint32_t imageIndex;
        bool sRGB;
        std::string name;
    };

    MeshRange MergeMesh(Mesh&& mesh);

    std::unique_ptr<Buffer> m_vertexBuffer;
    std::unique_ptr<Buffer> m_indexBuffer;
    std::unique_ptr<Buffer> m_renderPrimitiveBuffer;
    std::unique_ptr<Buffer> m_indexBufferInfoBuffer;
    std::unique_ptr<Buffer> m_materialBuffer;

    std::vector<std::vector<std::byte>> m_encodedImages;
    std::vector<TextureSource> m_textureSources;
    std::vector<Texture> m_textures;
};
