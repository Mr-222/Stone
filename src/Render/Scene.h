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
    void CommitToGPU(MTL::Device* device);
    void RegisterBuffers(RenderGraph& graph);

public:
    struct MeshRange
    {
        uint32_t firstSubmesh;
        uint32_t submeshCount;
    };

    MeshRange MergeMesh(Mesh&& mesh);

    std::vector<Vertex> m_globalVertices;
    std::vector<uint32_t> m_globalIndices;
    std::vector<SubMesh> m_submeshes;
    std::vector<RenderObject> m_objects;

    std::unique_ptr<Buffer> m_vertexBuffer;
    std::unique_ptr<Buffer> m_indexBuffer;
    std::unique_ptr<Buffer> m_renderObjBuffer;
    std::unique_ptr<Buffer> m_indexBufferInfoBuffer;
};
