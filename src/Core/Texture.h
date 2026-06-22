#pragma once

#include <Metal/Metal.hpp>

#include "Heap.h"

class Texture {
public:
    static Texture Borrowed(MTL::Texture* texture);

    Texture(MTL::Device* device, const MTL::TextureDescriptor* descriptor);
    Texture(const Heap& heap, const MTL::TextureDescriptor* descriptor);
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    MTL::Texture* GetNative() const { return m_texture; }
    size_t GetWidth() const { return m_texture ? m_texture->width() : 0; }
    size_t GetHeight() const { return m_texture ? m_texture->height() : 0; }

private:
    Texture(MTL::Texture* texture, bool ownsTexture);

    void Release();

    MTL::Texture* m_texture = nullptr;
    bool m_ownsTexture = true;
};
