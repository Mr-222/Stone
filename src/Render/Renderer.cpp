#include "Renderer.h"

#include <memory>

#include "Core/CommandBufferPool.h"
#include "Core/MetalContext.h"
#include "Core/Texture.h"
#include "Core/Window.h"
#include "Core/RenderGraph.h"
#include "Camera.h"
#include "TrianglePass.h"

struct FrameUniform {
    glm::mat4 viewProjection;
};

Renderer::~Renderer() = default;

Renderer::Renderer() {
    Setup();
}

void Renderer::Setup() {
    m_window = std::make_unique<Window>(1600, 900);
    m_metalContext = std::make_shared<MetalContext>(m_window->GetCAMetalLayer());
    m_commandBufferPool = std::make_shared<CommandBufferPool>(64, *m_metalContext);
    m_renderGraph = std::make_unique<RenderGraph>(m_metalContext, m_commandBufferPool);

    CameraConfig config = {
        .position = glm::vec3(0.0f, 0.0f, -3.0f),
        .yaw = 90.f,
        .pitch = 0.f,
        .fov = 45.f,
        .aspectRatio = 16.0f / 9.0f,
        .nearPlane = 0.01f,
        .farPlane = 100.0f
    };
    m_camera = std::make_unique<Camera>(config);
    m_frameUniform = std::make_unique<Buffer>(m_metalContext->GetDevice(), sizeof(FrameUniform),
                        MTL::ResourceStorageModeShared);
    m_renderGraph->RegisterBuffer("frameUniform", *m_frameUniform);

    m_trianglePass = std::make_unique<TrianglePass>();
    m_trianglePass->Setup(*m_metalContext, *m_commandBufferPool);
    m_trianglePass->AddToGraph(*m_renderGraph);

    m_renderGraph->Compile();
}

void Renderer::Run() {
    while (!m_window->ShouldClose()) {
        m_window->PollEvents();

        m_metalContext->BeginFrame();

        Texture backbuffer = Texture::Borrowed(m_metalContext->GetCurrentDrawable()->texture());
        m_renderGraph->RegisterTexture(kSwapchainImageName, backbuffer);

        FrameUniform frameUniform = {
            .viewProjection = m_camera->GetProjectionMatrix() * m_camera->GetViewMatrix()
        };
        m_frameUniform->Update(&frameUniform, sizeof(FrameUniform));

        DoRender();

        m_metalContext->EndFrame();
    }
}

void Renderer::DoRender() {
    m_renderGraph->Execute(m_metalContext->GetCommandQueue());
}
