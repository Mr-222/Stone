#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unordered_map>
#include <Metal/Metal.hpp>

#include "Core/Texture.h"

constexpr std::string kSwapchainImageName = "swapchain_image";

struct RenderGraphTextureHandle {
    static constexpr uint32_t InvalidIndex = std::numeric_limits<uint32_t>::max();

    uint32_t index = InvalidIndex;

    bool IsValid() const { return index != InvalidIndex; }
};

class RenderGraphResources {
public:
    void Clear();

    RenderGraphTextureHandle ImportTexture(std::string name, MTL::Texture* texture);
    RenderGraphTextureHandle ImportTexture(std::string name, const Texture& texture);

    MTL::Texture* GetTexture(RenderGraphTextureHandle handle) const;
    RenderGraphTextureHandle GetTextureHandle(std::string_view name) const;

    bool IsTextureHandleValid(RenderGraphTextureHandle handle) const;
    const std::string& GetTextureName(RenderGraphTextureHandle handle) const;

private:
    struct TextureResource {
        std::string name;
        MTL::Texture* texture = nullptr;
    };

    std::vector<TextureResource> m_textures;
    std::unordered_map<std::string, RenderGraphTextureHandle> m_textureLUT;
};
