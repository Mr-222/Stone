#include "Texture.h"

#include <utility>

#include "Utility/Logger.h"

Texture Texture::Borrowed(MTL::Texture* texture) {
    LOG_ERROR_IF(!texture, "Cannot borrow a null texture.");
    return Texture(texture, false);
}

Texture::Texture(MTL::Device* device, const MTL::TextureDescriptor* descriptor)
    : m_texture(device->newTexture(descriptor))
{
    LOG_ERROR_IF(!m_texture, "Failed to create texture.");
}

Texture::Texture(const Heap& heap, const MTL::TextureDescriptor* descriptor)
    : m_texture(heap.GetNative()->newTexture(descriptor))
{
    LOG_ERROR_IF(!m_texture, "Failed to create heap texture.");
}

Texture::Texture(MTL::Texture* texture, bool ownsTexture)
    : m_texture(texture)
    , m_ownsTexture(ownsTexture)
{
}

Texture::~Texture() {
    Release();
}

Texture::Texture(Texture&& other) noexcept
    : m_texture(std::exchange(other.m_texture, nullptr))
    , m_ownsTexture(std::exchange(other.m_ownsTexture, false))
{
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        Release();
        m_texture = std::exchange(other.m_texture, nullptr);
        m_ownsTexture = std::exchange(other.m_ownsTexture, false);
    }

    return *this;
}

void Texture::Release() {
    if (m_ownsTexture && m_texture)
        m_texture->release();

    m_texture = nullptr;
    m_ownsTexture = false;
}
