#pragma once

#include <vector>
#include <filesystem>
#include <memory>

#include "Mesh.h"
#include "Core/Buffer.h"
#include "Core/RenderGraph.h"

class Scene {
public:
    void LoadGltf(std::filesystem::path path);
    void CommitToGPU(MTL::Device* device, CommandBufferPool& commandBufferPool, MTL4::CommandQueue* queue);
    void RegisterBuffers(RenderGraph& graph);

    std::vector<Vertex> globalVertices;
    std::vector<uint32_t> globalIndices;
    std::vector<SubMesh> submeshes;
    std::vector<RenderObject> objects;
    std::vector<RenderPrimitive> renderPrimitives;

private:
    struct MeshRange
    {
        uint32_t firstSubmesh;
        uint32_t submeshCount;
    };

    MeshRange MergeMesh(Mesh&& mesh);

    std::unique_ptr<Buffer> m_vertexBuffer;
    std::unique_ptr<Buffer> m_indexBuffer;
    std::unique_ptr<Buffer> m_renderPrimitiveBuffer;
    std::unique_ptr<Buffer> m_indexBufferInfoBuffer;
};
