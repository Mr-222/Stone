#include "ImageLoader.h"

#include <memory>

#include "Logger.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct StbiImageDeleter {
    void operator()(stbi_uc* pixels) const {
        stbi_image_free(pixels);
    }
};

ImageData DecodeImageRGBA8(std::span<const std::byte> encodedImage) {
    LOG_ERROR_IF(encodedImage.empty(), "Cannot decode an empty image.");
    LOG_ERROR_IF(encodedImage.size() > INT_MAX, "Encoded image is too large for stb_image.");

    int width = 0;
    int height = 0;
    std::unique_ptr<stbi_uc, StbiImageDeleter> pixels(
        stbi_load_from_memory(
            reinterpret_cast<const stbi_uc*>(encodedImage.data()),
            static_cast<int>(encodedImage.size()),
            &width,
            &height,
            nullptr,
            STBI_rgb_alpha));

    LOG_ERROR_IF(!pixels, "Failed to decode image: {}", stbi_failure_reason());
    LOG_ERROR_IF(width <= 0 || height <= 0, "Decoded image has invalid dimensions {}x{}.", width, height);

    const size_t byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * STBI_rgb_alpha;
    return ImageData{
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        std::vector<uint8_t>(pixels.get(), pixels.get() + byteCount),
    };
}
