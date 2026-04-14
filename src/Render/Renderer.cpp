#include "Renderer.h"

#include <array>
#include <glm/glm.hpp>

#include "Core/MetalContext.h"
#include "Core/Window.h"
#include "Core/CommandBufferPool.h"
#include "Core/Buffer.h"
#include "Utility/Logger.h"

constexpr const char* kTriangleShaderLibrary = STONE_SHADER_DIR "/Triangle.metallib";

MTL::RenderPipelineState* m_pipelineState = nullptr;
MTL4::ArgumentTable* m_argumentTable = nullptr;
MTL::ResidencySet* m_residencySet = nullptr;
MTL::Buffer* m_argumentBuffer = nullptr;
Buffer* m_positionBuffer = nullptr;
Buffer* m_colorBuffer = nullptr;

Renderer::~Renderer() = default;

Renderer::Renderer() {
    m_window = std::make_unique<Window>(500, 500);
    m_metalContext = std::make_unique<MetalContext>(m_window->GetCAMetalLayer());
    m_commandBufferPool = std::make_unique<CommandBufferPool>(64, *m_metalContext);

    Setup();
}

void Renderer::Run() {
    while (!m_window->ShouldClose()) {
        m_window->PollEvents();

        m_metalContext->BeginFrame();
        DoRender();
        m_metalContext->EndFrame();
    }
}

void Renderer::Setup() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = m_metalContext->GetDevice();

    // Create PSO
    NS::Error* error = nullptr;
    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");

    NS::String* libraryPath = NS::String::string(kTriangleShaderLibrary, NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(libraryPath, &error);
    LOG_ERROR_IF(!library, "Failed to load shader library {}: {}", kTriangleShaderLibrary, error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();

    MTL4::LibraryFunctionDescriptor* vertexFunc = MTL4::LibraryFunctionDescriptor::alloc()->init()->autorelease();
    vertexFunc->setLibrary(library);
    vertexFunc->setName(NS::String::string("vertex_main", NS::UTF8StringEncoding));

    MTL4::LibraryFunctionDescriptor* fragmentFunc = MTL4::LibraryFunctionDescriptor::alloc()->init()->autorelease();
    fragmentFunc->setLibrary(library);
    fragmentFunc->setName(NS::String::string("fragment_main", NS::UTF8StringEncoding));

    pipelineDescriptor->setVertexFunctionDescriptor(vertexFunc);
    pipelineDescriptor->setFragmentFunctionDescriptor(fragmentFunc);
    pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::Function* vertexFunction = library->newFunction(NS::String::string("vertex_main", NS::UTF8StringEncoding));
    LOG_ERROR_IF(!vertexFunction, "Failed to load vertex function for argument encoding");

    MTL::ArgumentEncoder* argumentEncoder = vertexFunction->newArgumentEncoder(0);
    LOG_ERROR_IF(!argumentEncoder, "Failed to create argument encoder for triangle bindings");

    // Create geometry buffers and populate the argument buffer they live behind.
    std::array<glm::vec4, 3> positions = {{
        { -0.5f, -0.5f, 0.f, 1.f },
        { 0.5f, -0.5f, 0.f, 1.f },
        { 0.f, 0.5f, 0.f, 1.f },
    }};
    std::array<glm::vec4, 3> colors = {{
        { 1.f, 0.f, 0.f, 1.f },
        { 0.f, 1.f, 0.f, 1.f },
        { 0.f, 0.f, 1.f, 1.f },
    }};

    Buffer positionBufferCopy = Buffer(m_metalContext->GetDevice(), positions.data(), sizeof(positions), MTL::ResourceStorageModeShared);
    Buffer colorBufferCopy = Buffer(m_metalContext->GetDevice(), colors.data(), sizeof(colors), MTL::ResourceStorageModeShared);
    m_positionBuffer = new Buffer(m_metalContext->GetDevice(), sizeof(positions), MTL::ResourceStorageModePrivate);
    m_colorBuffer = new Buffer(m_metalContext->GetDevice(), sizeof(colors), MTL::ResourceStorageModePrivate);
    m_positionBuffer->UploadFromFlush(positionBufferCopy, *m_commandBufferPool, m_metalContext->GetCommandQueue());
    m_colorBuffer->UploadFromFlush(colorBufferCopy, *m_commandBufferPool, m_metalContext->GetCommandQueue());
    LOG_ERROR_IF(!m_positionBuffer->GetNativeBuffer() || !m_colorBuffer->GetNativeBuffer(), "Failed to allocate triangle buffers");

    m_argumentBuffer = device->newBuffer(argumentEncoder->encodedLength(), MTL::ResourceStorageModeShared);

    argumentEncoder->setArgumentBuffer(m_argumentBuffer, 0);
    argumentEncoder->setBuffer(m_positionBuffer->GetNativeBuffer(), 0, 0);
    argumentEncoder->setBuffer(m_colorBuffer->GetNativeBuffer(), 0, 1);

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("Triangle Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(1);
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_argumentBuffer->gpuAddress(), 0);

    // Create residency set
    MTL::ResidencySetDescriptor* rsDesc = MTL::ResidencySetDescriptor::alloc()->init()->autorelease();
    m_residencySet = device->newResidencySet(rsDesc, &error);
    m_residencySet->addAllocation(m_argumentBuffer);
    m_residencySet->addAllocation(m_positionBuffer->GetNativeBuffer());
    m_residencySet->addAllocation(m_colorBuffer->GetNativeBuffer());
    m_residencySet->commit();

    argumentEncoder->release();
    vertexFunction->release();
    library->release();
    compiler->release();

    pool->release();
}

void Renderer::DoRender() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    CommandBuffer cmd = m_commandBufferPool->Acquire();

    MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
    MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);

    colorAttachment->setTexture(m_metalContext->GetCurrentDrawable()->texture());
    colorAttachment->setLoadAction(MTL::LoadActionClear);
    colorAttachment->setClearColor(MTL::ClearColor::Make(0.1, 0.2, 0.3, 1.0));
    colorAttachment->setStoreAction(MTL::StoreActionStore);

    MTL4::RenderCommandEncoder* commandEncoder = cmd.BeginRenderPass(passDescriptor, m_residencySet);
    MTL::Texture* drawableTexture = m_metalContext->GetCurrentDrawable()->texture();
    MTL::Viewport viewport { 0.0, 0.0, static_cast<double>(drawableTexture->width()), static_cast<double>(drawableTexture->height()), 0.0, 1.0 };

    LOG_ERROR_IF(!commandEncoder, "Failed to create render command encoder");
    LOG_ERROR_IF(!m_pipelineState, "Render pipeline state is null");
    LOG_ERROR_IF(!m_argumentTable, "Argument table is null");

    commandEncoder->setRenderPipelineState(m_pipelineState);
    commandEncoder->setViewport(viewport);
    commandEncoder->setCullMode(MTL::CullModeBack);
    commandEncoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
    commandEncoder->setArgumentTable(m_argumentTable, MTL::RenderStageVertex);
    commandEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
    commandEncoder->endEncoding();

    cmd.SubmitTo(m_metalContext->GetCommandQueue());

    pool->release();
}
