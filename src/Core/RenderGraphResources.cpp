#include "RenderGraphResources.h"

#include "Utility/Logger.h"

void RenderGraphResources::Clear() {
    m_resources.clear();
    m_resourcesLUT.clear();
}

RenderGraphResourceHandle RenderGraphResources::DeclareTexture(std::string name) {
    if (m_resourcesLUT.contains(name)) {
        LOG_ERROR_IF(m_resources[m_resourcesLUT[name].index].type != RenderGraphResourceType::Texture,
            "'{}' already declared and it's not a texture.", name);
        return m_resourcesLUT[name];
    }

    m_resources.push_back(RenderGraphResource{
       .name = name,
       .type = RenderGraphResourceType::Texture,
       .texture = nullptr,
    });

    RenderGraphResourceHandle handle { static_cast<uint32_t>(m_resources.size() - 1) };
    m_resourcesLUT[name] = handle;
    return handle;
}

RenderGraphResourceHandle RenderGraphResources::RegisterTexture(std::string name, const Texture& texture) {
    MTL::Texture* nativeTexture = texture.GetNative();
    LOG_ERROR_IF(!nativeTexture, "Cannot register null texture '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareTexture(name);
    m_resources[handle.index].texture = nativeTexture;

    return handle;
}

MTL::Texture* RenderGraphResources::GetTexture(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!IsTextureHandleValid(handle), "Invalid render graph texture handle {}.", handle.index);
    return m_resources[handle.index].texture;
}

RenderGraphResourceHandle RenderGraphResources::GetTextureHandle(const std::string& name) const {
    const auto it = m_resourcesLUT.find(name);
    LOG_ERROR_IF(it == m_resourcesLUT.end(), "Texture '{}' does not exist.", name);
    assert(m_resources[it->second.index].type == RenderGraphResourceType::Texture);
    return it->second;
}

RenderGraphResourceHandle RenderGraphResources::DeclareBuffer(std::string name) {
    if (m_resourcesLUT.contains(name)) {
        LOG_ERROR_IF(m_resources[m_resourcesLUT[name].index].type != RenderGraphResourceType::Buffer,
            "'{}' already declared and it's not a buffer.", name);
        return m_resourcesLUT[name];
    }

    m_resources.push_back(RenderGraphResource{
        .name = name,
        .type = RenderGraphResourceType::Buffer,
        .buffer = nullptr,
    });

    RenderGraphResourceHandle handle { static_cast<uint32_t>(m_resources.size() - 1) };
    m_resourcesLUT[name] = handle;
    return handle;
}

RenderGraphResourceHandle RenderGraphResources::RegisterBuffer(std::string name, const Buffer& buffer) {
    MTL::Buffer* nativeBuffer = buffer.GetNative();
    LOG_ERROR_IF(!nativeBuffer, "Cannot register null buffer '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareBuffer(name);
    m_resources[handle.index].buffer = nativeBuffer;

    return handle;
}

MTL::Buffer *RenderGraphResources::GetBuffer(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!IsBufferHandleValid(handle), "Invalid render graph buffer handle {}.", handle.index);
    return m_resources[handle.index].buffer;
}

RenderGraphResourceHandle RenderGraphResources::GetBufferHandle(const std::string &name) const {
    const auto it = m_resourcesLUT.find(name);
    LOG_ERROR_IF(it == m_resourcesLUT.end(), "Buffer '{}' does not exist.", name);
    assert(m_resources[it->second.index].type == RenderGraphResourceType::Buffer);
    return it->second;
}

bool RenderGraphResources::IsTextureHandleValid(RenderGraphResourceHandle handle) const {
    return handle.IsValid() &&
           handle.index < m_resources.size() &&
           m_resources[handle.index].texture &&
           m_resources[handle.index].type == RenderGraphResourceType::Texture;
}

bool RenderGraphResources::IsTextureHandleDeclared(RenderGraphResourceHandle handle) const {
    return handle.IsValid() &&
           handle.index < m_resources.size() &&
           m_resources[handle.index].type == RenderGraphResourceType::Texture;
}

bool RenderGraphResources::IsBufferHandleValid(RenderGraphResourceHandle handle) const {
    return handle.IsValid() &&
           handle.index < m_resources.size() &&
           m_resources[handle.index].buffer &&
           m_resources[handle.index].type == RenderGraphResourceType::Buffer;
}

bool RenderGraphResources::IsBufferHandleDeclared(RenderGraphResourceHandle handle) const {
    return handle.IsValid() &&
           handle.index < m_resources.size() &&
           m_resources[handle.index].type == RenderGraphResourceType::Buffer;
}

const std::string& RenderGraphResources::GetTextureName(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!IsTextureHandleDeclared(handle), "Invalid render graph texture handle {}.", handle.index);
    return m_resources[handle.index].name;
}
