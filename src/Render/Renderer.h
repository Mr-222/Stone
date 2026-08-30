#pragma once

#include <memory>
#include <vector>

class MetalContext;
class Window;
class CommandBufferPool;
class RenderGraph;
class ObjectCullingPass;
class OpaqueDirectLightingPass;
class TransparentObjectCullingPass;
class TransparentDirectLightingPass;
class TransparentCompositePass;
class Scene;
class Camera;
class Buffer;
class Texture;
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
    std::unique_ptr<ObjectCullingPass> m_objectCullingPass;
    std::unique_ptr<OpaqueDirectLightingPass> m_opaqueDirectLightingPass;
    std::unique_ptr<TransparentObjectCullingPass> m_transparentObjectCullingPass;
    std::unique_ptr<TransparentDirectLightingPass> m_transparentDirectLightingPass;
    std::unique_ptr<TransparentCompositePass> m_transparentCompositePass;
    std::unique_ptr<Scene> m_scene;

    std::unique_ptr<Camera> m_camera;
    std::vector<std::unique_ptr<Buffer>> m_frameUniforms;
    std::vector<std::unique_ptr<Texture>> m_depthTextures;

    bool m_firstMouse = true;
    double m_lastX;
    double m_lastY;
};
