#pragma once

#include <memory>

class MetalContext;
class Window;
class CommandBufferPool;
namespace MTL4 {
    class CommandBuffer;
}

class Renderer {
public:
    Renderer();
    ~Renderer();

    void Run();

private:
    void Setup();
    void DoRender();

    std::unique_ptr<MetalContext> m_metalContext;
    std::unique_ptr<Window> m_window;
    std::unique_ptr<CommandBufferPool> m_commandBufferPool;
};
