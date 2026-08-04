#include "Scene.h"

#include <fstream>
#include <optional>
#include <span>
#include <type_traits>

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include "Core/CommandBuffer.h"
#include "Shader/ShaderTypes.h"
#include "Utility/ImageLoader.h"
#include "Utility/Logger.h"
#include "GeometryGenerator.h"

std::vector<std::byte> ReadFileBytes(const fastgltf::sources::URI& source) {
    const bool isLocalPath = source.uri.isLocalPath();
    LOG_ERROR_IF(!isLocalPath, "Only local glTF image URIs are supported.");

    const std::filesystem::path path = source.uri.fspath();
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    LOG_ERROR_IF(!file, "Failed to open glTF image {}.", path.string());

    const auto fileSize = static_cast<size_t>(file.tellg());
    LOG_ERROR_IF(source.fileByteOffset > fileSize,
        "Image byte offset {} exceeds file size {} for {}.",
        source.fileByteOffset,
        fileSize,
        path.string());

    std::vector<std::byte> bytes(fileSize - source.fileByteOffset);
    file.seekg(static_cast<std::streamoff>(source.fileByteOffset), std::ios::beg);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    LOG_ERROR_IF(!file, "Failed to read glTF image {}.", path.string());
    return bytes;
}

std::vector<std::byte> CopyImageBytes(const fastgltf::Asset& asset, const fastgltf::Image& image) {
    std::vector<std::byte> bytes;

    std::visit([&](const auto& source) {
        using Source = std::decay_t<decltype(source)>;

        if constexpr (
            std::is_same_v<Source, fastgltf::sources::Array> ||
            std::is_same_v<Source, fastgltf::sources::Vector> ||
            std::is_same_v<Source, fastgltf::sources::ByteView>)
        {
            bytes.assign(source.bytes.begin(), source.bytes.end());
        } else if constexpr (std::is_same_v<Source, fastgltf::sources::BufferView>) {
            const auto view =
                fastgltf::DefaultBufferDataAdapter{}(asset, source.bufferViewIndex);
            bytes.assign(view.begin(), view.end());
        } else if constexpr (std::is_same_v<Source, fastgltf::sources::URI>) {
            bytes = ReadFileBytes(source);
        }
    }, image.data);

    LOG_ERROR_IF(bytes.empty(), "glTF image '{}' has no supported encoded data.", image.name);
    return bytes;
}

size_t AlignUp(const size_t value, const size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void UploadTexture(
    Texture& texture,
    const ImageData& image,
    CommandBufferPool& commandBufferPool,
    MTL4::CommandQueue* queue)
{
    MTL::Texture* nativeTexture = texture.GetNative();
    MTL::Device* device = nativeTexture->device();
    const size_t unalignedBytesPerRow = static_cast<size_t>(image.width) * 4;
    const size_t alignment = device->minimumTextureBufferAlignmentForPixelFormat(nativeTexture->pixelFormat());
    LOG_ERROR_IF(alignment == 0, "Metal returned an invalid texture-buffer alignment.");

    const size_t bytesPerRow = AlignUp(unalignedBytesPerRow, alignment);
    const size_t uploadSize = bytesPerRow * image.height;
    std::vector<uint8_t> uploadData(uploadSize);
    for (size_t row = 0; row < image.height; ++row) {
        std::memcpy(
            uploadData.data() + row * bytesPerRow,
            image.pixels.data() + row * unalignedBytesPerRow,
            unalignedBytesPerRow);
    }

    Buffer stagingBuffer(device, uploadData.data(), uploadData.size(), MTL::ResourceStorageModeShared);
    CommandBuffer commandBuffer = commandBufferPool.AcquireFlushGPU();
    commandBuffer.AddResource(stagingBuffer.GetNative());
    commandBuffer.AddResource(nativeTexture);

    MTL4::ComputeCommandEncoder* encoder = commandBuffer.BeginComputePass();
    encoder->copyFromBuffer(
        stagingBuffer.GetNative(),
        0,
        bytesPerRow,
        uploadSize,
        MTL::Size(image.width, image.height, 1),
        nativeTexture,
        0,
        0,
        MTL::Origin(0, 0, 0));
    if (nativeTexture->mipmapLevelCount() > 1)
        encoder->generateMipmaps(nativeTexture);
    encoder->endEncoding();
    commandBuffer.SubmitTo(queue);
}

glm::mat4 ToGlm(const fastgltf::math::fmat4x4& matrix) {
    glm::mat4 result(1.f);
    for (std::size_t column = 0; column < 4; ++column) {
        for (std::size_t row = 0; row < 4; ++row) {
            result[column][row] = matrix[column][row];
        }
    }
    return result;
}

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
                it->materialIndex.has_value() ? static_cast<uint32_t>(it->materialIndex.value()) : 0,
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
            submesh.materialIndex,
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

    m_encodedImages.reserve(asset->images.size());
    for (const fastgltf::Image& image : asset->images)
        m_encodedImages.emplace_back(CopyImageBytes(asset.get(), image));

    m_textureSources.reserve(asset->textures.size());
    for (size_t textureIndex = 0; textureIndex < asset->textures.size(); ++textureIndex) {
        const fastgltf::Texture& texture = asset->textures[textureIndex];

        LOG_ERROR_IF(!texture.imageIndex.has_value(), "glTF texture {} does not reference a supported image.", textureIndex);
        LOG_ERROR_IF(texture.imageIndex.value() >= asset->images.size(), "glTF texture {} references invalid image {}.", textureIndex, texture.imageIndex.value());

        std::string name(texture.name.begin(), texture.name.end());
        if (name.empty())
            name = "Scene Texture " + std::to_string(textureIndex + 1);

        m_textureSources.push_back(TextureSource{
            static_cast<uint32_t>(texture.imageIndex.value()),
            false,
            std::move(name),
        });
    }

    materials.reserve(asset->materials.size());
    for (size_t materialIndex = 0; materialIndex < asset->materials.size(); ++materialIndex) {
        const fastgltf::Material& gltfMaterial = asset->materials[materialIndex];
        const auto& factor = gltfMaterial.pbrData.baseColorFactor;

        SceneMaterial material{
            .baseColorFactor = glm::vec4(factor.x(), factor.y(), factor.z(), factor.w()),
            .baseColorTextureIndex = 0,
        };

        if (gltfMaterial.pbrData.baseColorTexture.has_value()) {
            const fastgltf::TextureInfo& textureInfo = gltfMaterial.pbrData.baseColorTexture.value();
            LOG_ERROR_IF(textureInfo.textureIndex >= asset->textures.size(),
                "glTF material {} references invalid base-color texture {}.",
                materialIndex,
                textureInfo.textureIndex);
            LOG_WARN_IF(textureInfo.texCoordIndex != 0,
                "glTF material {} uses TEXCOORD_{}, but Stone currently samples TEXCOORD_0.",
                materialIndex,
                textureInfo.texCoordIndex);
            LOG_WARN_IF(textureInfo.transform != nullptr,
                "glTF material {} has a base-color texture transform that is not yet supported.",
                materialIndex);

            material.baseColorTextureIndex = static_cast<uint32_t>(textureInfo.textureIndex) + 1; // texture 0 is fallback white texture
            m_textureSources[textureInfo.textureIndex].sRGB = true;
        }

        materials.push_back(material);
    }

    std::vector<MeshRange> meshRanges;
    meshRanges.reserve(asset->meshes.size());
    for (auto& gltfMesh : asset->meshes) {
        for (const fastgltf::Primitive& primitive : gltfMesh.primitives) {
            LOG_ERROR_IF(primitive.materialIndex.has_value() && primitive.materialIndex.value() >= asset->materials.size(),
                "glTF primitive references invalid material {}.",
                primitive.materialIndex.value());
        }
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
        const auto objectIndex = static_cast<uint32_t>(objects.size());
        objects.push_back(RenderObject{
            ToGlm(transform),
        });

        for (uint32_t i = 0; i < meshRange.submeshCount; ++i) {
            renderPrimitives.push_back(RenderPrimitive{
                meshRange.firstSubmesh + i,
                objectIndex,
            });
        }
    });
}

void Scene::CommitToGPU(MTL::Device* device, CommandBufferPool& commandBufferPool, MTL4::CommandQueue* queue) {
    LOG_ERROR_IF(!device, "Failed to commit scene to GPU: device is null");
    LOG_ERROR_IF(!queue, "Failed to commit scene to GPU: command queue is null");

    LOG_ERROR_IF(m_textureSources.size() + 1 > kMaxBindlessTextureCount,
        "Scene needs {} textures, but the bindless table supports {}.",
        m_textureSources.size() + 1,
        kMaxBindlessTextureCount);

    m_textures.reserve(m_textureSources.size() + 1);
    std::vector<std::optional<ImageData>> decodedImages(m_encodedImages.size());

    const auto createTexture = [&](const ImageData& image, const MTL::PixelFormat format, const std::string& name) {
        MTL::TextureDescriptor* descriptor = MTL::TextureDescriptor::texture2DDescriptor(
            format,
            image.width,
            image.height,
            true);
        descriptor->setStorageMode(MTL::StorageModePrivate);
        descriptor->setUsage(MTL::TextureUsageShaderRead);

        m_textures.emplace_back(device, descriptor);
        Texture& texture = m_textures.back();
        texture.GetNative()->setLabel(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
        UploadTexture(texture, image, commandBufferPool, queue);
    };

    const ImageData defaultWhiteTexture {
        .width = 1,
        .height = 1,
        .pixels = { 255, 255, 255, 255 },
    };
    createTexture(defaultWhiteTexture, MTL::PixelFormatRGBA8Unorm_sRGB, "Default White Texture");

    for (const TextureSource& textureSource : m_textureSources) {
        LOG_ERROR_IF(textureSource.imageIndex >= m_encodedImages.size(),
            "Texture '{}' references invalid image {}.",
            textureSource.name,
            textureSource.imageIndex);

        std::optional<ImageData>& decodedImage = decodedImages[textureSource.imageIndex];
        if (!decodedImage.has_value()) {
            const std::vector<std::byte>& encodedImage = m_encodedImages[textureSource.imageIndex];
            decodedImage = DecodeImageRGBA8(std::span(encodedImage.data(), encodedImage.size()));
        }

        createTexture(
            decodedImage.value(),
            textureSource.sRGB ? MTL::PixelFormatRGBA8Unorm_sRGB : MTL::PixelFormatRGBA8Unorm,
            textureSource.name);
    }

    LOG_WARN_IF(globalVertices.empty(), "Scene has no vertices to commit");
    const size_t vertexBufferSize = globalVertices.size() * sizeof(Vertex);
    m_vertexBuffer = std::make_unique<Buffer>(
        device,
        vertexBufferSize,
        MTL::ResourceStorageModePrivate);
    m_vertexBuffer->GetNative()->setLabel(NS::String::string("Global Vertex Buffer", NS::UTF8StringEncoding));
    if (vertexBufferSize > 0)
        m_vertexBuffer->UpdateStaged(globalVertices.data(), vertexBufferSize, 0, commandBufferPool, queue);

    LOG_WARN_IF(globalIndices.empty(), "Scene has no indices to commit");
    const size_t indexBufferSize = globalIndices.size() * sizeof(uint32_t);
    m_indexBuffer = std::make_unique<Buffer>(
        device,
        indexBufferSize,
        MTL::ResourceStorageModePrivate);
    m_indexBuffer->GetNative()->setLabel(NS::String::string("Global Index Buffer", NS::UTF8StringEncoding));
    if (indexBufferSize > 0)
        m_indexBuffer->UpdateStaged(globalIndices.data(), indexBufferSize, 0, commandBufferPool, queue);

    const uint64_t indexBufferAddress = m_indexBuffer->GetGPUAddress();
    m_indexBufferInfoBuffer = std::make_unique<Buffer>(
        device,
        &indexBufferAddress,
        sizeof(indexBufferAddress),
        MTL::ResourceStorageModeShared
    );
    m_indexBufferInfoBuffer->GetNative()->setLabel(NS::String::string("Index Buffer Info Buffer", NS::UTF8StringEncoding));

    std::vector<GPURenderPrimitive> primitiveEntries{};
    primitiveEntries.reserve(renderPrimitives.size());
    for (const RenderPrimitive& renderPrimitive : renderPrimitives) {
        const RenderObject& renderObject = objects[renderPrimitive.objectIndex];
        const SubMesh& submesh = submeshes[renderPrimitive.submeshIndex];

        GPURenderPrimitive entry{};
        entry.worldMat = renderObject.transform;
        entry.baseVertex = submesh.vertexOffset;
        entry.firstIndex = submesh.firstIndex;
        entry.indexCount = submesh.indexCount;
        entry.materialIndex = submesh.materialIndex;
        primitiveEntries.push_back(entry);
    }
    m_renderPrimitiveBuffer = std::make_unique<Buffer>(
        device,
        primitiveEntries.size() * sizeof(GPURenderPrimitive),
        MTL::ResourceStorageModePrivate);
    m_renderPrimitiveBuffer->GetNative()->setLabel(NS::String::string("Render Primitive Buffer", NS::UTF8StringEncoding));
    if (!primitiveEntries.empty())
        m_renderPrimitiveBuffer->UpdateStaged(primitiveEntries.data(), primitiveEntries.size() * sizeof(GPURenderPrimitive), 0, commandBufferPool, queue);

    std::vector<GPUMaterial> materialEntries{};
    materialEntries.reserve(materials.size());
    for (const SceneMaterial& material : materials) {
        materialEntries.push_back(GPUMaterial{
            .baseColorFactor = material.baseColorFactor,
            .baseColorTextureIndex = material.baseColorTextureIndex,
            .pad0 = 0,
            .pad1 = 0,
            .pad2 = 0,
        });
    }
    m_materialBuffer = std::make_unique<Buffer>(
        device,
        materialEntries.size() * sizeof(GPUMaterial),
        MTL::ResourceStorageModePrivate);
    m_materialBuffer->GetNative()->setLabel(NS::String::string("Material Buffer", NS::UTF8StringEncoding));
    m_materialBuffer->UpdateStaged(
        materialEntries.data(),
        materialEntries.size() * sizeof(GPUMaterial),
        0,
        commandBufferPool,
        queue);
}

void Scene::RegisterBuffers(RenderGraph &graph) {
    graph.RegisterBuffer("GlobalVertexBuffer", *m_vertexBuffer);
    graph.RegisterBuffer("GlobalIndexBuffer", *m_indexBuffer);
    graph.RegisterBuffer("IndexBufferInfoBuffer", *m_indexBufferInfoBuffer);
    graph.RegisterBuffer("RenderPrimitiveBuffer", *m_renderPrimitiveBuffer);
    graph.RegisterBuffer("MaterialBuffer", *m_materialBuffer);
}
