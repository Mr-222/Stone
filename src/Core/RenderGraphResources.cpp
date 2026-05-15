#include "RenderGraphResources.h"

#include "Utility/Logger.h"

void RenderGraphResources::Clear() {
    m_textures.clear();
    m_textureLUT.clear();
}

RenderGraphTextureHandle RenderGraphResources::DeclareTexture(std::string name) {
    if (m_textureLUT.contains(name))
        return m_textureLUT[name];

    m_textures.push_back(TextureResource{
       .name = name,
       .texture = nullptr,
    });

    RenderGraphTextureHandle handle { static_cast<uint32_t>(m_textures.size() - 1) };
    m_textureLUT[name] = handle;
    return handle;
}

RenderGraphTextureHandle RenderGraphResources::RegisterTexture(std::string name, const Texture& texture) {
    MTL::Texture* nativeTexture = texture.GetNative();
    LOG_ERROR_IF(!nativeTexture, "Cannot register null texture '{}' into render graph.", name);

    RenderGraphTextureHandle handle = DeclareTexture(name);
    m_textures[handle.index].texture = nativeTexture;

    return handle;
}

MTL::Texture* RenderGraphResources::GetTexture(RenderGraphTextureHandle handle) const {
    LOG_ERROR_IF(!IsTextureHandleValid(handle), "Invalid render graph texture handle {}.", handle.index);
    return m_textures[handle.index].texture;
}

RenderGraphTextureHandle RenderGraphResources::GetTextureHandle(std::string_view name) const {
    LOG_ERROR_IF(!m_textureLUT.contains(name.data()), "Texture '{}' does not exist.", name);
    return m_textureLUT.at(name.data());
}

bool RenderGraphResources::IsTextureHandleValid(RenderGraphTextureHandle handle) const {
    return handle.IsValid() && handle.index < m_textures.size() && m_textures[handle.index].texture;
}

bool RenderGraphResources::IsTextureHandleDeclared(RenderGraphTextureHandle handle) const {
    return handle.IsValid() && handle.index < m_textures.size();
}

const std::string& RenderGraphResources::GetTextureName(RenderGraphTextureHandle handle) const {
    LOG_ERROR_IF(!IsTextureHandleDeclared(handle), "Invalid render graph texture handle {}.", handle.index);
    return m_textures[handle.index].name;
}
