#include "RenderGraphResources.h"

#include "Utility/Logger.h"

void RenderGraphResources::Clear() {
    m_textures.clear();
}

RenderGraphTextureHandle RenderGraphResources::ImportTexture(std::string name, MTL::Texture* texture) {
    LOG_ERROR_IF(!texture, "Cannot import null texture '{}' into render graph.", name);

    RenderGraphTextureHandle handle;

    if (m_textureLUT.contains(name)) {
        handle = m_textureLUT[name];
        m_textures[handle.index].texture = texture;
    }
    else {
        m_textures.push_back(TextureResource{
           .name = name,
           .texture = texture,
        });
        handle = RenderGraphTextureHandle { static_cast<uint32_t>(m_textures.size() - 1) };
        m_textureLUT[name] = handle;
    }

    return handle;
}

RenderGraphTextureHandle RenderGraphResources::ImportTexture(std::string name, const Texture& texture) {
    return ImportTexture(std::move(name), texture.GetNative());
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

const std::string& RenderGraphResources::GetTextureName(RenderGraphTextureHandle handle) const {
    LOG_ERROR_IF(!IsTextureHandleValid(handle), "Invalid render graph texture handle {}.", handle.index);
    return m_textures[handle.index].name;
}
