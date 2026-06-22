#include "Renderer.h"

#include <memory>
#include <GLFW/glfw3.h>

#include "Core/CommandBufferPool.h"
#include "Core/MetalContext.h"
#include "Core/Texture.h"
#include "Core/Window.h"
#include "Core/RenderGraph.h"
#include "Render/Camera.h"
#include "Render/TrianglePass.h"
#include "Render/ObjectCullingPass.h"
#include "Render/Scene.h"

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

    m_scene = std::make_unique<Scene>();
    m_scene->LoadGltf("./Models/FlightHelmet/glTF/FlightHelmet.gltf");
    m_scene->CommitToGPU(m_metalContext->GetDevice());
    m_scene->RegisterBuffers(*m_renderGraph);

    m_trianglePass = std::make_unique<TrianglePass>();
    m_trianglePass->Setup(*m_metalContext, *m_commandBufferPool);
    m_trianglePass->AddToGraph(*m_renderGraph);

    m_objectCullingPass = std::make_unique<ObjectCullingPass>();
    m_objectCullingPass->Setup(*m_metalContext, m_scene->objects.size());
    m_objectCullingPass->AddToGraph(*m_renderGraph);

    m_renderGraph->Compile();
}

void Renderer::Run() {
    glfwSetWindowUserPointer(m_window->GetGLFWWindow(), this);
    glfwSetInputMode(m_window->GetGLFWWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(m_window->GetGLFWWindow(), [](GLFWwindow* window, double xpos, double ypos) {
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
            return;
        }

        Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));

        // Prevent the camera from abruptly jumping or spinning wildly the very first time
        // user move mouse after the application starts.
        if (renderer->m_firstMouse) {
            renderer->m_lastX = xpos;
            renderer->m_lastY = ypos;
            renderer->m_firstMouse = false;
        }

        float xoffset = static_cast<float>(xpos - renderer->m_lastX);
        float yoffset = static_cast<float>(ypos - renderer->m_lastY);

        renderer->m_lastX = xpos;
        renderer->m_lastY = ypos;

        renderer->m_camera->ProcessMouseMovement(xoffset, yoffset);
    });

    glfwSetMouseButtonCallback(m_window->GetGLFWWindow(), [](GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    });

    float lastFrameTime = glfwGetTime();

    while (!m_window->ShouldClose()) {
        m_window->PollEvents();

        float currentFrameTime = glfwGetTime();
        float deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        GLFWwindow* glfwWindow = m_window->GetGLFWWindow();
        if (glfwGetKey(glfwWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            m_firstMouse = true;
        }

        if (glfwGetInputMode(glfwWindow, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
            if (glfwGetKey(glfwWindow, GLFW_KEY_W) == GLFW_PRESS)
                m_camera->ProcessKeyboard(Movement::FORWARD, deltaTime);
            if (glfwGetKey(glfwWindow, GLFW_KEY_S) == GLFW_PRESS)
                m_camera->ProcessKeyboard(Movement::BACKWARD, deltaTime);
            if (glfwGetKey(glfwWindow, GLFW_KEY_A) == GLFW_PRESS)
                m_camera->ProcessKeyboard(Movement::LEFT, deltaTime);
            if (glfwGetKey(glfwWindow, GLFW_KEY_D) == GLFW_PRESS)
                m_camera->ProcessKeyboard(Movement::RIGHT, deltaTime);
        }

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
