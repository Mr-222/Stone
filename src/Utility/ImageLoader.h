#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

struct ImageData {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> pixels;
};

ImageData DecodeImageRGBA8(std::span<const std::byte> encodedImage);
