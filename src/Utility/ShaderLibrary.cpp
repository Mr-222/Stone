#include "ShaderLibrary.h"

#include <utility>

#include "Utility/Logger.h"

namespace {
NS::String* MakeNSString(std::string_view string) {
    const std::string copy(string);
    return NS::String::string(copy.c_str(), NS::UTF8StringEncoding);
}
}

ShaderLibrary::ShaderLibrary(MTL::Library* library)
    : m_library(library)
{
}

ShaderLibrary::~ShaderLibrary() {
    Release();
}

ShaderLibrary::ShaderLibrary(ShaderLibrary&& other) noexcept
    : m_library(std::exchange(other.m_library, nullptr))
{
    m_functions.swap(other.m_functions);
}

ShaderLibrary& ShaderLibrary::operator=(ShaderLibrary&& other) noexcept {
    if (this == &other)
        return *this;

    Release();
    m_library = std::exchange(other.m_library, nullptr);
    m_functions.swap(other.m_functions);
    return *this;
}

MTL::Function* ShaderLibrary::GetFunction(std::string_view functionName) const {
    const auto it = m_functions.find(std::string(functionName));
    LOG_ERROR_IF(it == m_functions.end(), "Shader function {} was not loaded", functionName);
    return it->second;
}

void ShaderLibrary::AddFunction(std::string_view functionName, MTL::Function* function) {
    m_functions.emplace(std::string(functionName), function);
}

void ShaderLibrary::Release() {
    for (auto& [_, function] : m_functions) {
        if (function)
            function->release();
    }
    m_functions.clear();

    if (m_library)
        m_library->release();
    m_library = nullptr;
}

ShaderLibrary LoadShaderLibrary(
    MTL::Device* device,
    std::string_view libraryPath,
    std::initializer_list<std::string_view> functionNames
) {
    LOG_ERROR_IF(!device, "Cannot load shader library without a Metal device");

    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(MakeNSString(libraryPath), &error);
    LOG_ERROR_IF(
        !library,
        "Failed to load shader library {}: {}",
        libraryPath,
        error ? error->localizedDescription()->utf8String() : "unknown error"
    );

    ShaderLibrary shaderLibrary(library);
    for (std::string_view functionName : functionNames) {
        MTL::Function* function = library->newFunction(MakeNSString(functionName));
        LOG_ERROR_IF(!function, "Failed to load shader function {} from {}", functionName, libraryPath);
        shaderLibrary.AddFunction(functionName, function);
    }

    return shaderLibrary;
}

MTL4::LibraryFunctionDescriptor* MakeLibraryFunctionDescriptor(
    MTL::Library* library,
    std::string_view functionName
) {
    LOG_ERROR_IF(!library, "Cannot create function descriptor {} without a shader library", functionName);

    MTL4::LibraryFunctionDescriptor* descriptor = MTL4::LibraryFunctionDescriptor::alloc()->init()->autorelease();
    descriptor->setLibrary(library);
    descriptor->setName(MakeNSString(functionName));
    return descriptor;
}
