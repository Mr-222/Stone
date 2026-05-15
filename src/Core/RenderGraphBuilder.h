#pragma once

#include <vector>
#include <Metal/Metal.hpp>

#include "RenderGraphResources.h"

enum class RenderGraphTextureAccessType {
    Read,
    Write,
};

struct RenderGraphTextureAccess {
    RenderGraphTextureHandle texture;
    RenderGraphTextureAccessType type;
};

struct RenderGraphColorAttachmentDesc {
    MTL::LoadAction loadAction;
    MTL::StoreAction storeAction;
    MTL::ClearColor clearColor;
};

struct RenderGraphColorAttachment {
    RenderGraphTextureHandle texture;
    RenderGraphColorAttachmentDesc desc;
};

class RenderGraphBuilder {
public:
    void ReadTexture(RenderGraphTextureHandle texture) {
        m_textureAccesses.push_back(RenderGraphTextureAccess{
            .texture = texture,
            .type = RenderGraphTextureAccessType::Read,
        });
    }

    RenderGraphColorAttachment WriteColor(RenderGraphTextureHandle texture, const RenderGraphColorAttachmentDesc& desc) {
        m_textureAccesses.push_back(RenderGraphTextureAccess{
            .texture = texture,
            .type = RenderGraphTextureAccessType::Write,
        });

        return RenderGraphColorAttachment{
            .texture = texture,
            .desc = desc,
        };
    }

    const std::vector<RenderGraphTextureAccess>& GetTextureAccesses() const {
        return m_textureAccesses;
    }

private:
    std::vector<RenderGraphTextureAccess> m_textureAccesses;
};
