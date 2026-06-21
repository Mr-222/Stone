#include "Scene.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "Utility/Logger.h"
#include "GeometryGenerator.h"

struct RenderObjectGPUEntry {
    uint32_t baseVertexOffset;
    uint32_t firstIndex;
    uint32_t indexCount;
    // TODO: Add properties for compute shader culling (CellBound, MaterialID, etc.)
    uint32_t pad0;
    glm::mat4 transform;
};

glm::mat4 ToGlm(const fastgltf::math::fmat4x4& matrix) {
    glm::mat4 result(1.f);
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t row = 0; row < 4; ++row) {
            result[column][row] = matrix[column][row];
        }
    }
    return result;
}

// TODO: Implement submesh so that renderer support materials
Mesh LoadMesh(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh) {
    Mesh outputMesh;

    for (auto it = mesh.primitives.begin(); it != mesh.primitives.end(); ++it) {
        auto* positionIt = it->findAttribute("POSITION");
        const bool hasPosition = positionIt != it->attributes.end();
        LOG_ERROR_IF(!hasPosition, "Failed to find position attribute");

        const bool hasIndices = it->indicesAccessor.has_value();
        LOG_ERROR_IF(!hasIndices, "Failed to find index accessor");

        const size_t vertexOffset = outputMesh.vertices->size();

        // Position
        {
            auto& positionAccessor = asset.accessors[positionIt->accessorIndex];
            outputMesh.vertices->resize(vertexOffset + positionAccessor.count);
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, positionAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec3 pos, std::size_t idx) {
                auto& vertex = outputMesh.vertices->at(vertexOffset + idx);
                vertex.position = glm::vec3(pos.x(), pos.y(), pos.z());
                vertex.normal = glm::vec3(0.f, 1.f, 0.f);
                vertex.uv = glm::vec2(0.f, 0.f);
                vertex.tangent = glm::vec3(1.f, 0.f, 0.f);
            });
        }

        // Normal
        {
            const auto* normalIt = it->findAttribute("NORMAL");
            const bool hasNormal = normalIt != it->attributes.end();
            LOG_WARN_IF(!hasNormal, "Failed to find normal attribute");
            if (hasNormal) {
                auto& normalAccessor = asset.accessors[normalIt->accessorIndex];
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, normalAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec3 normal, std::size_t idx) {
                    outputMesh.vertices->at(vertexOffset + idx).normal = glm::vec3(normal.x(), normal.y(), normal.z());
                });
            }
        }

        // Tex coord
        {
            // TODO: support baseColorTextureIndex
            const auto* texcoordIt = it->findAttribute("TEXCOORD_0");
            const bool hasTexcoord = texcoordIt != it->attributes.end();
            LOG_WARN_IF(!hasTexcoord, "Failed to find texcoord attribute");
            if (hasTexcoord) {
                auto& texcoordAccessor = asset.accessors[texcoordIt->accessorIndex];
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(asset, texcoordAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec2 uv, std::size_t idx) {
                    outputMesh.vertices->at(vertexOffset + idx).uv = glm::vec2(uv.x(), uv.y());
                });
            }
        }

        // Tangent
        {
            const auto* tangentIt = it->findAttribute("TANGENT");
            const bool hasTangent = tangentIt != it->attributes.end();
            LOG_WARN_IF(!hasTangent, "Failed to find tangent attribute");
            if (hasTangent) {
                auto& tangentAccessor = asset.accessors[tangentIt->accessorIndex];
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(asset, tangentAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec4 tangent, std::size_t idx) {
                    outputMesh.vertices->at(vertexOffset + idx).tangent = glm::vec3(tangent.x(), tangent.y(), tangent.z());
                });
            }
        }

        // Indices
        {
            auto& indexAccessor = asset.accessors[it->indicesAccessor.value()];
            const auto indexOffset = outputMesh.indices->size();
            outputMesh.indices->resize(indexOffset + indexAccessor.count);
            fastgltf::iterateAccessorWithIndex<std::uint32_t>(asset, indexAccessor, [&outputMesh, vertexOffset, indexOffset](std::uint32_t index, std::size_t idx) {
                const auto meshIndex = vertexOffset + index;
                outputMesh.indices->at(indexOffset + idx) = static_cast<uint32_t>(meshIndex);
            });

            outputMesh.submeshes.push_back(SubMesh{
                static_cast<uint32_t>(indexOffset),
                static_cast<uint32_t>(indexAccessor.count),
                0,
            });
        }
    }

    return outputMesh;
}

Scene::MeshRange Scene::MergeMesh(Mesh&& mesh) {
    const auto vertexOffset = static_cast<uint32_t>(m_globalVertices.size());
    const auto indexOffset = static_cast<uint32_t>(m_globalIndices.size());
    const auto firstSubmesh = static_cast<uint32_t>(m_submeshes.size());
    const auto submeshCount = static_cast<uint32_t>(mesh.submeshes.size());

    m_globalVertices.insert(m_globalVertices.end(), mesh.vertices->begin(), mesh.vertices->end());
    m_globalIndices.insert(m_globalIndices.end(), mesh.indices->begin(), mesh.indices->end());

    for (const auto& submesh : mesh.submeshes) {
        m_submeshes.push_back(SubMesh{
            indexOffset + submesh.firstIndex,
            submesh.indexCount,
            vertexOffset + submesh.vertexOffset,
        });
    }

    return MeshRange{firstSubmesh, submeshCount};
}

void Scene::LoadGltf(std::filesystem::path path) {
    const bool fileExists = std::filesystem::is_regular_file(path);
    LOG_ERROR_IF(!fileExists, "Failed to load glTF file {}", path.string());

    LOG_INFO("Loading {}", path.string());

    fastgltf::Parser parser;
    constexpr auto gltfOptions =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble |
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::LoadExternalImages |
        fastgltf::Options::GenerateMeshIndices;
    auto gltfFile = fastgltf::MappedGltfFile::FromPath(path);
    LOG_ERROR_IF(!bool(gltfFile), "Failed to load glTF file {}", path.string());

    auto asset = parser.loadGltf(gltfFile.get(), path.parent_path(), gltfOptions);
    const bool loadFailed = asset.error() != fastgltf::Error::None;
    LOG_ERROR_IF(loadFailed, "Failed to load glTF: {}", fastgltf::getErrorMessage(asset.error()));

    std::vector<MeshRange> meshRanges;
    meshRanges.reserve(asset->meshes.size());
    for (auto& gltfMesh : asset->meshes) {
        meshRanges.emplace_back(MergeMesh(LoadMesh(asset.get(), gltfMesh)));
    }

    const bool hasScenes = !asset->scenes.empty();
    LOG_ERROR_IF(!hasScenes, "glTF file {} has no scenes.", path.string());
    const auto sceneIndex = asset->defaultScene.value_or(0);
    fastgltf::iterateSceneNodes(asset.get(), sceneIndex, fastgltf::math::fmat4x4(), [this, &meshRanges](fastgltf::Node& node, const fastgltf::math::fmat4x4& transform) {
        if (!node.meshIndex.has_value())
            return;

        const bool invalidMeshIndex = node.meshIndex.value() >= meshRanges.size();
        LOG_ERROR_IF(invalidMeshIndex, "Mesh index {} exceeds mesh range", node.meshIndex.value());

        const auto& meshRange = meshRanges[node.meshIndex.value()];
        m_objects.push_back(RenderObject{
            meshRange.firstSubmesh,
            meshRange.submeshCount,
            ToGlm(transform),
        });
    });
}

void Scene::CommitToGPU(MTL::Device* device) {
    LOG_ERROR_IF(!device, "Failed to commit scene to GPU: device is null");

    LOG_WARN_IF(m_globalVertices.empty(), "Scene has no vertices to commit");
    m_vertexBuffer = std::make_unique<Buffer>(
        device,
        m_globalVertices.data(),
        m_globalVertices.size() * sizeof(Vertex),
        MTL::ResourceStorageModeShared);

    LOG_WARN_IF(m_globalIndices.empty(), "Scene has no indices to commit");
    m_indexBuffer = std::make_unique<Buffer>(
        device,
        m_globalIndices.data(),
        m_globalIndices.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);

    const uint64_t indexBufferAddress = m_indexBuffer->GetGPUAddress();
    m_indexBufferInfoBuffer = std::make_unique<Buffer>(
        device,
        &indexBufferAddress,
        sizeof(indexBufferAddress),
        MTL::ResourceStorageModeShared
    );

    std::vector<RenderObjectGPUEntry> roEntries{};
    roEntries.reserve(m_objects.size());
    for (const RenderObject& renderObj : m_objects) {
        RenderObjectGPUEntry entry{};

        entry.transform = renderObj.transform;
        entry.baseVertexOffset = m_submeshes[renderObj.firstSubmesh].vertexOffset;
        entry.firstIndex = m_submeshes[renderObj.firstSubmesh].firstIndex;

        entry.indexCount = 0;
        for (int i = 0; i < renderObj.submeshCount; ++i)
            entry.indexCount += m_submeshes[renderObj.firstSubmesh + i].indexCount;

        roEntries.push_back(entry);
    }
    m_renderObjBuffer = std::make_unique<Buffer>(
        device,
        roEntries.data(),
        roEntries.size() * sizeof(RenderObjectGPUEntry),
        MTL::ResourceStorageModeShared);
}

void Scene::RegisterBuffers(RenderGraph &graph) {
    graph.RegisterBuffer("GlobalVertexBuffer", *m_vertexBuffer);
    graph.RegisterBuffer("GlobalIndexBuffer", *m_indexBuffer);
    graph.RegisterBuffer("IndexBufferInfoBuffer", *m_indexBufferInfoBuffer);
    graph.RegisterBuffer("RenderObjectBuffer", *m_renderObjBuffer);
}
