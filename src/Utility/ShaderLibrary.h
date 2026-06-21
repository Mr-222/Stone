#pragma once

#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>

#include <Metal/Metal.hpp>

class ShaderLibrary {
public:
    ShaderLibrary() = default;
    explicit ShaderLibrary(MTL::Library* library);
    ~ShaderLibrary();

    ShaderLibrary(const ShaderLibrary&) = delete;
    ShaderLibrary& operator=(const ShaderLibrary&) = delete;

    ShaderLibrary(ShaderLibrary&& other) noexcept;
    ShaderLibrary& operator=(ShaderLibrary&& other) noexcept;

    MTL::Library* GetLibrary() const { return m_library; }
    MTL::Function* GetFunction(std::string_view functionName) const;
    const std::unordered_map<std::string, MTL::Function*>& GetFunctions() const { return m_functions; }

private:
    void AddFunction(std::string_view functionName, MTL::Function* function);
    void Release();

    MTL::Library* m_library = nullptr;
    std::unordered_map<std::string, MTL::Function*> m_functions;

    friend ShaderLibrary LoadShaderLibrary(
        MTL::Device* device,
        std::string_view libraryPath,
        std::initializer_list<std::string_view> functionNames
    );
};

ShaderLibrary LoadShaderLibrary(
    MTL::Device* device,
    std::string_view libraryPath,
    std::initializer_list<std::string_view> functionNames
);

MTL4::LibraryFunctionDescriptor* MakeLibraryFunctionDescriptor(
    MTL::Library* library,
    std::string_view functionName
);
