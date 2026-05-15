#include "Renderer.h"

#include <memory>

#include "Core/CommandBufferPool.h"
#include "Core/MetalContext.h"
#include "Core/Texture.h"
#include "Core/Window.h"
#include "Core/RenderGraph.h"
#include "Render/TrianglePass.h"

Renderer::~Renderer() = default;

Renderer::Renderer() {
    Setup();
}

void Renderer::Setup() {
    m_window = std::make_unique<Window>(500, 500);
    m_metalContext = std::make_shared<MetalContext>(m_window->GetCAMetalLayer());
    m_commandBufferPool = std::make_shared<CommandBufferPool>(64, *m_metalContext);
    m_renderGraph = std::make_unique<RenderGraph>(m_metalContext, m_commandBufferPool);

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
        m_renderGraph->ImportTexture(kSwapchainImageName, backbuffer);
        DoRender();
        m_metalContext->EndFrame();
    }
}

void Renderer::DoRender() {
    m_renderGraph->Execute(m_metalContext->GetCommandQueue());
}
