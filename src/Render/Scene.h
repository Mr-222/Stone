#pragma once

#include <vector>
#include <filesystem>
#include <memory>

#include "Mesh.h"
#include "Core/Buffer.h"

class Scene {
public:
    void Init();
    void LoadGltf(std::filesystem::path path);
    void CommitToGPU(MTL::Device* device);

private:
    struct MeshRange
    {
        uint32_t firstSubmesh;
        uint32_t submeshCount;
    };

    MeshRange MergeMesh(Mesh&& mesh);

    std::vector<Vertex> globalVertices;
    std::vector<uint32_t> globalIndices;
    std::vector<SubMesh> submeshes;
    std::vector<SceneObject> objects;
    bool gltfLoaded = false;

    std::unique_ptr<Buffer> vertexBuffer;
    std::unique_ptr<Buffer> indexBuffer;
};
