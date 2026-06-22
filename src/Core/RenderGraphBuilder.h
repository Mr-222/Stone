#pragma once

#include <vector>
#include <Metal/Metal.hpp>

#include "RenderGraphResources.h"

enum class RenderGraphResourceAccessType {
    Read,
    Write,
};

struct RenderGraphResourceAccess {
    RenderGraphResourceHandle resource;
    RenderGraphResourceType resourceType;
    RenderGraphResourceAccessType accessType;
};

struct RenderGraphColorAttachmentDesc {
    MTL::LoadAction loadAction;
    MTL::StoreAction storeAction;
    MTL::ClearColor clearColor;
};

struct RenderGraphColorAttachment {
    RenderGraphResourceHandle texture;
    RenderGraphColorAttachmentDesc desc;
};

class RenderGraphBuilder {
public:
    void ReadTexture(RenderGraphResourceHandle texture) {
        m_resourceAccesses.push_back(RenderGraphResourceAccess{
            .resource = texture,
            .resourceType = RenderGraphResourceType::Texture,
            .accessType = RenderGraphResourceAccessType::Read,
        });
    }

    void ReadBuffer(RenderGraphResourceHandle buffer) {
        m_resourceAccesses.push_back(RenderGraphResourceAccess{
            .resource = buffer,
            .resourceType = RenderGraphResourceType::Buffer,
            .accessType = RenderGraphResourceAccessType::Read,
        });
    }

    void WriteBuffer(RenderGraphResourceHandle buffer) {
         m_resourceAccesses.push_back(RenderGraphResourceAccess{
            .resource = buffer,
            .resourceType = RenderGraphResourceType::Buffer,
            .accessType = RenderGraphResourceAccessType::Write,
         });
    }

    void ReadIndirectCommandBuffer(RenderGraphResourceHandle indirectCB) {
        m_resourceAccesses.push_back(RenderGraphResourceAccess{
            .resource = indirectCB,
            .resourceType = RenderGraphResourceType::IndirectCommandBuffer,
            .accessType = RenderGraphResourceAccessType::Read,
        });
    }

    void WriteIndirectCommandBuffer(RenderGraphResourceHandle indirectCB) {
         m_resourceAccesses.push_back(RenderGraphResourceAccess{
            .resource = indirectCB,
            .resourceType = RenderGraphResourceType::IndirectCommandBuffer,
            .accessType = RenderGraphResourceAccessType::Write,
         });
    }

    RenderGraphColorAttachment WriteColor(RenderGraphResourceHandle texture, const RenderGraphColorAttachmentDesc& desc) {
        m_resourceAccesses.push_back(RenderGraphResourceAccess{
            .resource = texture,
            .resourceType = RenderGraphResourceType::Texture,
            .accessType = RenderGraphResourceAccessType::Write,
        });

        return RenderGraphColorAttachment{
            .texture = texture,
            .desc = desc,
        };
    }

    const std::vector<RenderGraphResourceAccess>& GetResourceAccesses() const {
        return m_resourceAccesses;
    }

private:
    std::vector<RenderGraphResourceAccess> m_resourceAccesses;
};
