#pragma once

#include <memory>

class MetalContext;
class Window;
class CommandBufferPool;
class RenderGraph;
class TrianglePass;
class Camera;
class Buffer;
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

    std::shared_ptr<MetalContext> m_metalContext;
    std::unique_ptr<Window> m_window;
    std::shared_ptr<CommandBufferPool> m_commandBufferPool;

    std::unique_ptr<RenderGraph> m_renderGraph;
    std::unique_ptr<Buffer> m_indirectArgBuffer;
    std::unique_ptr<TrianglePass> m_trianglePass;

    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<Buffer> m_frameUniform;

    bool m_firstMouse = true;
    double m_lastX;
    double m_lastY;
};
