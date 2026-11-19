#include "Renderer.h"

#include <array>
#include <memory>
#include <GLFW/glfw3.h>

#include "Core/CommandBufferPool.h"
#include "Core/MetalContext.h"
#include "Core/Texture.h"
#include "Core/Window.h"
#include "Core/RenderGraph.h"
#include "Render/Camera.h"
#include "Render/ObjectCullingPass.h"
#include "Render/OpaqueDirectLightingPass.h"
#include "Render/TransparentObjectCullingPass.h"
#include "Render/TransparentDirectLightingPass.h"
#include "Render/TransparentCompositePass.h"
#include "Render/AtmosphereLUT.h"
#include "Render/AtmosphereScatteringPass.h"
#include "Render/FFTBloomPass.h"
#include "Render/TonemappingPass.h"
#include "Render/Scene.h"
#include "Shader/ShaderTypes.h"

Renderer::~Renderer() = default;

Renderer::Renderer() {
    Setup();
}

void Renderer::Setup() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    m_window = std::make_unique<Window>(1600, 900);
    m_metalContext = std::make_shared<MetalContext>(m_window->GetCAMetalLayer());
    m_commandBufferPool = std::make_shared<CommandBufferPool>(64, *m_metalContext);
    m_renderGraph = std::make_unique<RenderGraph>(m_metalContext, m_commandBufferPool);
    const uint32_t frameSlotCount = m_metalContext->GetFrameSlotCount();
    m_depthTextures.resize(frameSlotCount);
    m_sceneColorTextures.resize(frameSlotCount);

    CameraConfig config = {
        .position = glm::vec3(-11.0f, 1.5f, -0.4f),
        .yaw = 0.f,
        .pitch = 0.f,
        .fov = 45.f,
        .aspectRatio = 16.0f / 9.0f,
        .nearPlane = 0.01f,
        .farPlane = 100.0f
    };
    m_camera = std::make_unique<Camera>(config);
    m_frameUniforms.resize(frameSlotCount);
    for (uint32_t frameSlot = 0; frameSlot < m_frameUniforms.size(); ++frameSlot) {
        m_frameUniforms[frameSlot] = std::make_unique<Buffer>(
            m_metalContext->GetDevice(),
            sizeof(FrameUniform),
            MTL::ResourceStorageModeShared);
        m_renderGraph->RegisterFrameLocalBuffer("frameUniform", frameSlot, *m_frameUniforms[frameSlot]);
    }

    m_scene = std::make_unique<Scene>();
    m_scene->LoadGltf("./Models/Sponza/glTF/Sponza.gltf");
    m_scene->directionalLights = {
        DirectionalLight{
            .direction = glm::vec3(0.25f, -1.0f, 0.15f),
            .color = glm::vec3(1.0f, 0.98f, 0.92f),
            .illuminance = 10.0f,
        },
    };
    m_scene->ambientLight = AmbientLight{
        .color = glm::vec3(1.0f),
        .intensity = 0.1f,
    };
    const std::array<glm::vec3, 6> lightPalette = {
        glm::vec3(1.0f, 0.78f, 0.45f), // warm amber
        glm::vec3(1.0f, 0.52f, 0.22f), // torch orange
        glm::vec3(0.42f, 0.85f, 1.0f), // cyan
        glm::vec3(0.45f, 0.95f, 0.62f), // emerald
        glm::vec3(0.75f, 0.52f, 1.0f), // violet
        glm::vec3(1.0f, 0.96f, 0.88f), // soft white
    };
    const std::array floorHeights = { 1.6f, 5.6f };
    const std::array aisleZs = { -4.2f, -0.3f, 3.6f };
    m_scene->pointLights.clear();
    m_scene->pointLights.reserve(floorHeights.size() * 10 * aisleZs.size());
    for (size_t floorIdx = 0; floorIdx < floorHeights.size(); ++floorIdx) {
        for (size_t col = 0; col < 10; ++col) {
            const float x = -10.5f + static_cast<float>(col) * (20.0f / 9.0f);
            for (size_t aisleIdx = 0; aisleIdx < aisleZs.size(); ++aisleIdx) {
                const size_t paletteIdx = (floorIdx * 7 + col * 3 + aisleIdx) % lightPalette.size();
                m_scene->pointLights.push_back(PointLight{
                    .position = glm::vec3(x, floorHeights[floorIdx], aisleZs[aisleIdx]),
                    .color = lightPalette[paletteIdx],
                    .intensity = 6.0f,
                    .range = 3.8f,
                });
            }
        }
    }

    const std::array spotHeights = { 2.8f, 6.8f };
    const std::array spotXs = { -8.0f, -3.0f, 2.0f, 7.0f };
    const std::array spotZs = { -3.2f, 2.5f };
    m_scene->spotLights.clear();
    m_scene->spotLights.reserve(spotHeights.size() * spotXs.size() * spotZs.size());
    for (size_t floorIdx = 0; floorIdx < spotHeights.size(); ++floorIdx) {
        for (size_t xIdx = 0; xIdx < spotXs.size(); ++xIdx) {
            for (size_t zIdx = 0; zIdx < spotZs.size(); ++zIdx) {
                const float z = spotZs[zIdx];
                const glm::vec3 pos(spotXs[xIdx], spotHeights[floorIdx], z);
                const glm::vec3 target(spotXs[xIdx], spotHeights[floorIdx] - 2.6f, -0.3f);
                const size_t paletteIdx = (floorIdx * 3 + xIdx + zIdx * 2) % lightPalette.size();
                m_scene->spotLights.push_back(SpotLight{
                    .position = pos,
                    .direction = glm::normalize(target - pos),
                    .color = lightPalette[paletteIdx],
                    .intensity = 20.0f,
                    .range = 7.5f,
                    .innerAngle = glm::radians(15.0f),
                    .outerAngle = glm::radians(30.0f),
                });
            }
        }
    }
    m_scene->CommitToGPU(m_metalContext->GetDevice(), *m_commandBufferPool, m_metalContext->GetComputeCommandQueue());
    m_scene->RegisterBuffers(*m_renderGraph);

    m_renderGraph->AddPassNode<AtmosphereLUT>("AtmosphereLUT", *m_metalContext);
    m_renderGraph->AddPassNode<ObjectCullingPass>("ObjectCulling", *m_metalContext, m_scene->opaqueRenderPrimitives.size(), m_scene->opaqueMaskedPrimitiveStart);
    m_renderGraph->AddPassNode<OpaqueDirectLightingPass>("OpaqueDirectLighting", *m_metalContext, m_scene->opaqueRenderPrimitives.size(), m_scene->opaqueMaskedPrimitiveStart, m_scene->GetTextures(), m_scene->GetTLAS(), m_scene->GetAccelerationStructureResidencySet());
    m_renderGraph->AddPassNode<AtmosphereScatteringPass>("AtmosphereScattering", *m_metalContext);
    m_renderGraph->AddPassNode<TransparentObjectCullingPass>("TransparentObjectCulling", *m_metalContext, m_scene->transparentRenderPrimitives.size());
    m_renderGraph->AddPassNode<TransparentDirectLightingPass>("TransparentDirectLighting", *m_metalContext, m_scene->transparentRenderPrimitives.size(), m_scene->GetTextures(), m_scene->GetTLAS(), m_scene->GetAccelerationStructureResidencySet());
    m_renderGraph->AddPassNode<TransparentCompositePass>("TransparentComposite", *m_metalContext);
    m_renderGraph->AddPassNode<FFTBloomPass>("FFTBloom", *m_metalContext);
    m_renderGraph->AddPassNode<TonemappingPass>("Tonemapping", *m_metalContext, *m_window);

    m_renderGraph->SetDependencyGraph({
        { "OpaqueDirectLighting", { "ObjectCulling", "AtmosphereLUT" } },
        { "AtmosphereScattering", { "OpaqueDirectLighting" } },
        { "TransparentObjectCulling", { "ObjectCulling" } },
        { "TransparentDirectLighting", { "AtmosphereScattering", "TransparentObjectCulling" } },
        { "TransparentComposite", { "TransparentDirectLighting" } },
        { "FFTBloom", { "TransparentComposite" } },
        { "Tonemapping", { "FFTBloom" } }
    });

    m_renderGraph->Compile();

    pool->release();
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
    int frameCount = 0;
    float fpsAccumulator = 0.0f;

    while (!m_window->ShouldClose()) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        m_window->PollEvents();

        float currentFrameTime = glfwGetTime();
        float deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        frameCount++;
        fpsAccumulator += deltaTime;
        if (fpsAccumulator >= 0.5f) {
            float fps = frameCount / fpsAccumulator;
            m_window->SetTitle("Stone | " + std::to_string(static_cast<int>(fps)) + " FPS");
            frameCount = 0;
            fpsAccumulator = 0.0f;
        }

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

        // register frame buffer
        MTL::Texture* backbufferTexture = m_metalContext->GetCurrentDrawable()->texture();
        Texture backbuffer = Texture::Borrowed(backbufferTexture);
        m_renderGraph->RegisterTexture(kSwapchainImageName, backbuffer);

        // register depth buffer
        const uint32_t frameSlot = m_metalContext->GetCurrentFrameSlot();
        std::unique_ptr<Texture>& depthTexture = m_depthTextures[frameSlot];
        if (!depthTexture || depthTexture->GetWidth() != backbufferTexture->width() || depthTexture->GetHeight() != backbufferTexture->height()) {
            MTL::TextureDescriptor* depthDescriptor = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, backbufferTexture->width(), backbufferTexture->height(), false);
            depthDescriptor->setStorageMode(MTL::StorageModePrivate);
            depthDescriptor->setUsage(MTL::TextureUsageRenderTarget);

            depthTexture = std::make_unique<Texture>(m_metalContext->GetDevice(), depthDescriptor);
            depthTexture->GetNative()->setLabel(NS::String::string("Scene Depth", NS::UTF8StringEncoding));
        }
        m_renderGraph->RegisterFrameLocalTexture(kSceneDepthImageName, frameSlot, *depthTexture);

        // register scene color buffer
        std::unique_ptr<Texture>& sceneColorTexture = m_sceneColorTextures[frameSlot];
        if (!sceneColorTexture || sceneColorTexture->GetWidth() != backbufferTexture->width() || sceneColorTexture->GetHeight() != backbufferTexture->height()) {
            MTL::TextureDescriptor* sceneColorDescriptor = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, backbufferTexture->width(), backbufferTexture->height(), false);
            sceneColorDescriptor->setStorageMode(MTL::StorageModePrivate);
            sceneColorDescriptor->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);

            sceneColorTexture = std::make_unique<Texture>(m_metalContext->GetDevice(), sceneColorDescriptor);
            sceneColorTexture->GetNative()->setLabel(NS::String::string("Scene Color", NS::UTF8StringEncoding));
        }
        m_renderGraph->RegisterFrameLocalTexture(kSceneColorImageName, frameSlot, *sceneColorTexture);

        const glm::mat4 view = m_camera->GetViewMatrix();
        const glm::mat4 proj = m_camera->GetProjectionMatrix();
        const glm::mat4 viewProj = proj * view;
        const uint32_t width = static_cast<uint32_t>(backbufferTexture->width());
        const uint32_t height = static_cast<uint32_t>(backbufferTexture->height());
        const uint32_t tileCountX = (width + kLightTileSize - 1) / kLightTileSize;
        const uint32_t tileCountY = (height + kLightTileSize - 1) / kLightTileSize;

        FrameUniform frameUniform = {
            .view = view,
            .proj = proj,
            .viewProjection = viewProj,
            .invViewProj = glm::inverse(viewProj),
            .cameraPosition = glm::vec4(m_camera->GetPosition(), 1.0f),
            .sunTransmittance = glm::vec4(1.0f),
            .screenSizeAndTileCount = glm::uvec4(width, height, tileCountX, tileCountY),
        };
        m_frameUniforms[frameSlot]->Update(&frameUniform, sizeof(FrameUniform));

        DoRender();

        m_metalContext->EndFrame();

        pool->release();
    }
}

void Renderer::DoRender() {
    m_renderGraph->Execute(m_metalContext->GetRenderCommandQueue(), m_metalContext->GetComputeCommandQueue());
}
