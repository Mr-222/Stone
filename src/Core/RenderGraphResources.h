#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <Metal/Metal.hpp>

#include "Texture.h"

constexpr std::string kSwapchainImageName = "swapchain_image";

enum class RenderGraphResourceType {
    Texture,
    Buffer,
};

struct RenderGraphResourceHandle {
    static constexpr uint32_t InvalidIndex = std::numeric_limits<uint32_t>::max();

    uint32_t index = InvalidIndex;

    bool IsValid() const { return index != InvalidIndex; }
};

struct RenderGraphResource {
    std::string name;
    RenderGraphResourceType type;
    MTL::Texture* texture = nullptr;
    MTL::Buffer* buffer = nullptr;
};

class RenderGraphResources {
public:
    void Clear();

    RenderGraphResourceHandle DeclareTexture(std::string name);
    RenderGraphResourceHandle RegisterTexture(std::string name, const Texture& texture);

    MTL::Texture* GetTexture(RenderGraphResourceHandle handle) const;
    RenderGraphResourceHandle GetTextureHandle(const std::string& name) const;

    bool IsTextureHandleValid(RenderGraphResourceHandle handle) const;
    bool IsTextureHandleDeclared(RenderGraphResourceHandle handle) const;
    const std::string& GetTextureName(RenderGraphResourceHandle handle) const;

private:
    std::vector<RenderGraphResource> m_resources;
    std::unordered_map<std::string, RenderGraphResourceHandle> m_resourcesLUT;
};
