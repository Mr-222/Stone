#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <Metal/Metal.hpp>

#include "Texture.h"
#include "Buffer.h"
#include "IndirectCommandBuffer.h"

constexpr std::string kSwapchainImageName = "swapchain_image";
constexpr std::string kSceneDepthImageName = "scene_depth";

enum class RenderGraphResourceType {
    Texture,
    Buffer,
    IndirectCommandBuffer,
};

struct RenderGraphResourceHandle {
    static constexpr uint32_t InvalidIndex = std::numeric_limits<uint32_t>::max();

    uint32_t index = InvalidIndex;

    bool IsValid() const { return index != InvalidIndex; }
};

struct RenderGraphResource {
    std::string name;
    RenderGraphResourceType type;
    union {
        MTL::Texture* texture;
        MTL::Buffer* buffer;
        MTL::IndirectCommandBuffer* indirectCB;
    };

    RenderGraphResource(std::string n, RenderGraphResourceType t)
        : name(std::move(n)), type(t), texture(nullptr) {}
};

class RenderGraphResources {
public:
    void Clear();

    RenderGraphResourceHandle DeclareTexture(std::string name);
    RenderGraphResourceHandle RegisterTexture(std::string name, const Texture& texture);
    MTL::Texture* GetTexture(RenderGraphResourceHandle handle) const;
    RenderGraphResourceHandle GetTextureHandle(const std::string& name) const;

    RenderGraphResourceHandle DeclareBuffer(std::string name);
    RenderGraphResourceHandle RegisterBuffer(std::string name, const Buffer& buffer);
    MTL::Buffer* GetBuffer(RenderGraphResourceHandle handle) const;
    RenderGraphResourceHandle GetBufferHandle(const std::string& name) const;

    RenderGraphResourceHandle DeclareIndirectCommandBuffer(std::string name);
    RenderGraphResourceHandle RegisterIndirectCommandBuffer(std::string name, const IndirectCommandBuffer& indirectCB);
    MTL::IndirectCommandBuffer* GetIndirectCommandBuffer(RenderGraphResourceHandle handle) const;
    RenderGraphResourceHandle GetIndirectCommandBufferHandle(const std::string& name) const;

    bool IsTextureHandleValid(RenderGraphResourceHandle handle) const;
    bool IsTextureHandleDeclared(RenderGraphResourceHandle handle) const;
    bool IsBufferHandleValid(RenderGraphResourceHandle handle) const;
    bool IsBufferHandleDeclared(RenderGraphResourceHandle handle) const;
    bool IsIndirectCommandBufferHandleValid(RenderGraphResourceHandle handle) const;
    bool IsIndirectCommandBufferHandleDeclared(RenderGraphResourceHandle handle) const;

    const std::string& GetName(RenderGraphResourceHandle handle) const;

private:
    std::vector<RenderGraphResource> m_resources;
    std::unordered_map<std::string, RenderGraphResourceHandle> m_resourcesLUT;
};
