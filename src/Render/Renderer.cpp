#include "Render/Renderer.h"

#include "Core/MetalContext.h"
#include "Core/Window.h"

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

void Renderer::Init() {
    m_window = std::make_unique<Window>(500, 500);

    m_metalContext = std::make_unique<MetalContext>();
    m_metalContext->Init(m_window->GetCAMetalLayer());
}

void Renderer::Run() {
    while (!m_window->ShouldClose()) {
        m_window->PollEvents();

        m_metalContext->BeginFrame();

        m_metalContext->EndFrame();
    }
}
