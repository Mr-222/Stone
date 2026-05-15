#include "Scene.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "Utility/Logger.h"
#include "GeometryGenerator.h"

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
    const auto vertexOffset = static_cast<uint32_t>(globalVertices.size());
    const auto indexOffset = static_cast<uint32_t>(globalIndices.size());
    const auto firstSubmesh = static_cast<uint32_t>(submeshes.size());
    const auto submeshCount = static_cast<uint32_t>(mesh.submeshes.size());

    globalVertices.insert(globalVertices.end(), mesh.vertices->begin(), mesh.vertices->end());
    globalIndices.insert(globalIndices.end(), mesh.indices->begin(), mesh.indices->end());

    for (const auto& submesh : mesh.submeshes) {
        submeshes.push_back(SubMesh{
            indexOffset + submesh.firstIndex,
            submesh.indexCount,
            vertexOffset + submesh.vertexOffset,
        });
    }

    return MeshRange{firstSubmesh, submeshCount};
}

void Scene::Init() {
    auto sphere = MergeMesh(GeometryGenerator::Sphere(
        {0.f, 0.f, 0.f},
        1.0f,
        32));
    objects.push_back({
        sphere.firstSubmesh,
        sphere.submeshCount,
        glm::mat4(1.f),
    });

    LoadGltf("path/to/scene.gltf");
}

void Scene::LoadGltf(std::filesystem::path path) {
    const bool alreadyLoaded = gltfLoaded;
    LOG_ERROR_IF(alreadyLoaded, "Scene::LoadGltf can only be called once");

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
    gltfLoaded = true;

    const bool hasScenes = !asset->scenes.empty();
    LOG_ERROR_IF(!hasScenes, "glTF file {} has no scenes.", path.string());
    const auto sceneIndex = asset->defaultScene.value_or(0);
    fastgltf::iterateSceneNodes(asset.get(), sceneIndex, fastgltf::math::fmat4x4(), [this, &meshRanges](fastgltf::Node& node, const fastgltf::math::fmat4x4& transform) {
        if (!node.meshIndex.has_value())
            return;

        const bool invalidMeshIndex = node.meshIndex.value() >= meshRanges.size();
        LOG_ERROR_IF(invalidMeshIndex, "Mesh index {} exceeds mesh range", node.meshIndex.value());

        const auto& meshRange = meshRanges[node.meshIndex.value()];
        objects.push_back(SceneObject{
            meshRange.firstSubmesh,
            meshRange.submeshCount,
            ToGlm(transform),
        });
    });
}

void Scene::CommitToGPU(MTL::Device* device) {
    LOG_ERROR_IF(!device, "Failed to commit scene to GPU: device is null");

    const bool hasVertices = !globalVertices.empty();
    LOG_WARN_IF(!hasVertices, "Scene has no vertices to commit");
    if (hasVertices) {
        vertexBuffer = std::make_unique<Buffer>(
            device,
            globalVertices.data(),
            globalVertices.size() * sizeof(Vertex),
            MTL::ResourceStorageModeShared);
    }
    else {
        vertexBuffer.reset();
    }

    const bool hasIndices = !globalIndices.empty();
    LOG_WARN_IF(!hasIndices, "Scene has no indices to commit");
    if (hasIndices) {
        indexBuffer = std::make_unique<Buffer>(
            device,
            globalIndices.data(),
            globalIndices.size() * sizeof(uint32_t),
            MTL::ResourceStorageModeShared);
    }
    else {
        indexBuffer.reset();
    }
}
