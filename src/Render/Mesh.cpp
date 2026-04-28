#include "Mesh.h"

#include <cassert>
#include <limits>
#include <stdexcept>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "Utility/Logger.h"

namespace {

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
Mesh LoadMesh(const fastgltf::Asset& asset, const fastgltf::Mesh& mesh, const glm::mat4& transform, const std::filesystem::path& path) {
    Mesh outputMesh;
    outputMesh.transform = transform;

    for (auto it = mesh.primitives.begin(); it != mesh.primitives.end(); ++it) {
        auto* positionIt = it->findAttribute("POSITION");
        LOG_ERROR_IF(positionIt == it->attributes.end(), "Failed to find position attribute"); // A mesh primitive is required to hold the POSITION attribute
        assert(it->indicesAccessor.has_value()); // We specify GenerateMeshIndices, so we should always have indices

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
            if (const auto* normalIt = it->findAttribute("NORMAL"); normalIt != it->attributes.end()) {
                auto& normalAccessor = asset.accessors[normalIt->accessorIndex];
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(asset, normalAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec3 normal, std::size_t idx) {
                    outputMesh.vertices->at(vertexOffset + idx).normal = glm::vec3(normal.x(), normal.y(), normal.z());
                });
            }
            else {
                LOG_WARN("Failed to find normal attribute");
            }
        }

        // Tex coord
        {
            // TODO: support baseColorTextureIndex
            if (const auto* texcoordIt = it->findAttribute("TEXCOORD0"); texcoordIt != it->attributes.end()) {
                auto& texcoordAccessor = asset.accessors[texcoordIt->accessorIndex];
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(asset, texcoordAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec2 uv, std::size_t idx) {
                    outputMesh.vertices->at(vertexOffset + idx).uv = glm::vec2(uv.x(), uv.y());
                });
            }
            else {
                LOG_WARN("Failed to find texcoord attribute");
            }
        }

        // Tangent
        {
            if (const auto* tangentIt = it->findAttribute("TANGENT"); tangentIt != it->attributes.end()) {
                auto& tangentAccessor = asset.accessors[tangentIt->accessorIndex];
                fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(asset, tangentAccessor, [&outputMesh, vertexOffset](fastgltf::math::fvec4 tangent, std::size_t idx) {
                    outputMesh.vertices->at(vertexOffset + idx).tangent = glm::vec3(tangent.x(), tangent.y(), tangent.z());
                });
            }
            else {
                LOG_WARN("Failed to find tangent attribute");
            }
        }

        // Indices
        {
            auto& indexAccessor = asset.accessors[it->indicesAccessor.value()];
            const auto indexOffset = outputMesh.indices->size();
            outputMesh.indices->resize(indexOffset + indexAccessor.count);
            fastgltf::iterateAccessorWithIndex<std::uint32_t>(asset, indexAccessor, [&outputMesh, &path, vertexOffset, indexOffset](std::uint32_t index, std::size_t idx) {
                const auto meshIndex = vertexOffset + index;
                if (meshIndex > std::numeric_limits<std::uint16_t>::max()) {
                    LOG_ERROR("Index {} in {} exceeds uint16_t range", meshIndex, path.string());
                    throw std::runtime_error("glTF index exceeds uint16_t range");
                }
                outputMesh.indices->at(indexOffset + idx) = static_cast<std::uint16_t>(meshIndex);
            });
        }
    }

    return outputMesh;
}

}

Mesh::Mesh() : transform(glm::mat4(1.f)) {
    vertices = std::make_unique<std::vector<Vertex>>();
    indices  = std::make_unique<std::vector<uint16_t>>();
}

std::vector<Mesh> LoadGltf(std::filesystem::path path) {
    LOG_ERROR_IF(!std::filesystem::is_regular_file(path), "Failed to load glTF file {}", path.string());
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
    LOG_ERROR_IF(asset.error() != fastgltf::Error::None, "Failed to load glTF: {}", fastgltf::getErrorMessage(asset.error()));

    std::vector<Mesh> outputMeshes;

    if (!asset->scenes.empty()) {
        const auto sceneIndex = asset->defaultScene.value_or(0);
        fastgltf::iterateSceneNodes(asset.get(), sceneIndex, fastgltf::math::fmat4x4(), [&outputMeshes, &asset, &path](fastgltf::Node& node, const fastgltf::math::fmat4x4& transform) {
            if (!node.meshIndex.has_value()) {
                return;
            }

            outputMeshes.emplace_back(LoadMesh(asset.get(), asset->meshes[node.meshIndex.value()], ToGlm(transform), path));
        });
    }
    else {
        LOG_WARN("glTF file {} has no scenes; loading meshes with identity transforms", path.string());
        for (const auto& mesh : asset->meshes) {
            outputMeshes.emplace_back(LoadMesh(asset.get(), mesh, glm::mat4(1.f), path));
        }
    }

    return outputMeshes;
}
