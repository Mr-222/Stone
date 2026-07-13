#include "TrianglePass.h"

#include <array>
#include <glm/glm.hpp>

#include "Core/Buffer.h"
#include "Core/Heap.h"
#include "Core/MetalContext.h"
#include "Core/RenderGraph.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kTriangleShaderLibrary = STONE_SHADER_DIR "/Triangle.metallib";

struct TrianglePassData {
    RenderGraphColorAttachment colorAttachment;
    MTL::RenderPipelineState* pipelineState = nullptr;
    MTL4::ArgumentTable* argumentTable = nullptr;
    MTL::Heap* privateHeap = nullptr;
    MTL::Heap* sharedHeap = nullptr;
    RenderGraphResourceHandle frameUniformHandle;
};

TrianglePass::TrianglePass() = default;

TrianglePass::~TrianglePass() {
    if (m_argumentTable)
        m_argumentTable->release();
    if (m_pipelineState)
        m_pipelineState->release();
}

void TrianglePass::Setup(MetalContext& context, CommandBufferPool& commandBufferPool) {
    MTL::Device* device = context.GetDevice();

    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kTriangleShaderLibrary, {
        "vertex_main",
        "fragment_main",
    });

    MTL4::RenderPipelineDescriptor* pipelineDescriptor = MTL4::RenderPipelineDescriptor::alloc()->init()->autorelease();

    MTL4::LibraryFunctionDescriptor* vertexFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "vertex_main");
    MTL4::LibraryFunctionDescriptor* fragmentFunc = MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "fragment_main");

    pipelineDescriptor->setVertexFunctionDescriptor(vertexFunc);
    pipelineDescriptor->setFragmentFunctionDescriptor(fragmentFunc);
    pipelineDescriptor->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    pipelineDescriptor->setInputPrimitiveTopology(MTL::PrimitiveTopologyClassTriangle);

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newRenderPipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create render pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::Function* vertexFunction = shaderLibrary.GetFunction("vertex_main");

    MTL::ArgumentEncoder* argumentEncoder = vertexFunction->newArgumentEncoder(static_cast<NS::UInteger>(TriangleBufferIndex::BindlessArguments));
    LOG_ERROR_IF(!argumentEncoder, "Failed to create argument encoder for triangle bindings");

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

    m_privateHeap = std::make_unique<Heap>(device, 1024 * 1024, MTL::StorageModePrivate);
    m_sharedHeap = std::make_unique<Heap>(device, 1024, MTL::StorageModeShared);

    Buffer positionBufferCopy(device, positions.data(), sizeof(positions), MTL::ResourceStorageModeShared);
    Buffer colorBufferCopy(device, colors.data(), sizeof(colors), MTL::ResourceStorageModeShared);
    m_positionBuffer = std::make_unique<Buffer>(*m_privateHeap, sizeof(positions), MTL::ResourceStorageModePrivate);
    m_colorBuffer = std::make_unique<Buffer>(*m_privateHeap, sizeof(colors), MTL::ResourceStorageModePrivate);
    m_positionBuffer->UploadFromFlush(positionBufferCopy, commandBufferPool, context.GetRenderCommandQueue());
    m_colorBuffer->UploadFromFlush(colorBufferCopy, commandBufferPool, context.GetRenderCommandQueue());
    LOG_ERROR_IF(!m_positionBuffer->GetNative() || !m_colorBuffer->GetNative(), "Failed to allocate triangle buffers");

    m_argumentBuffer = std::make_unique<Buffer>(*m_sharedHeap, argumentEncoder->encodedLength(), MTL::ResourceStorageModeShared);

    argumentEncoder->setArgumentBuffer(m_argumentBuffer->GetNative(), 0);
    argumentEncoder->setBuffer(m_positionBuffer->GetNative(), 0, static_cast<NS::UInteger>(TriangleBindlessArgumentID::Positions));
    argumentEncoder->setBuffer(m_colorBuffer->GetNative(), 0, static_cast<NS::UInteger>(TriangleBindlessArgumentID::Colors));

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("Triangle Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(TriangleBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_argumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(TriangleBufferIndex::BindlessArguments));

    argumentEncoder->release();
    compiler->release();
}

void TrianglePass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle swapchainHandle = graph.DeclareTexture(kSwapchainImageName);
    RenderGraphResourceHandle frameUniformHandle = graph.DeclareBuffer("frameUniform");

    graph.AddPass<TrianglePassData>(
        "Triangle",
        IsCompute,
        [this, swapchainHandle, frameUniformHandle](RenderGraphBuilder& builder, TrianglePassData& data, RenderGraphResources& resources) {
            data.colorAttachment = builder.WriteColor(swapchainHandle, RenderGraphColorAttachmentDesc{
                .loadAction = MTL::LoadActionClear,
                .storeAction = MTL::StoreActionStore,
                .clearColor = MTL::ClearColor::Make(0.1, 0.2, 0.3, 1.0),
            });
            data.pipelineState = m_pipelineState;
            data.argumentTable = m_argumentTable;
            data.privateHeap = m_privateHeap->GetNative();
            data.sharedHeap = m_sharedHeap->GetNative();
            data.frameUniformHandle = frameUniformHandle;
            builder.ReadBuffer(frameUniformHandle);

            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(frameUniformHandle);
            data.argumentTable->setAddress(frameUniformBuffer->gpuAddress(), static_cast<NS::UInteger>(TriangleBufferIndex::FrameUniform));
        },
        [](const TrianglePassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* frameUniformBuffer = resources.GetBuffer(data.frameUniformHandle);
            cmd.AddResource(frameUniformBuffer);
            cmd.AddResource(data.privateHeap);
            cmd.AddResource(data.sharedHeap);

            MTL::Texture* colorTexture = resources.GetTexture(data.colorAttachment.texture);
            LOG_ERROR_IF(!colorTexture, "Triangle pass has no color target.");
            LOG_ERROR_IF(!data.pipelineState, "Triangle pass pipeline state is null.");
            LOG_ERROR_IF(!data.argumentTable, "Triangle pass argument table is null.");

            MTL4::RenderPassDescriptor* passDescriptor = MTL4::RenderPassDescriptor::alloc()->init()->autorelease();
            MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);
            colorAttachment->setTexture(colorTexture);
            colorAttachment->setLoadAction(data.colorAttachment.desc.loadAction);
            colorAttachment->setClearColor(data.colorAttachment.desc.clearColor);
            colorAttachment->setStoreAction(data.colorAttachment.desc.storeAction);

            MTL4::RenderCommandEncoder* renderEncoder = cmd.BeginRenderPass(passDescriptor);
            LOG_ERROR_IF(!renderEncoder, "Failed to create render command encoder");

            MTL::Viewport viewport {
                0.0,
                0.0,
                static_cast<double>(colorTexture->width()),
                static_cast<double>(colorTexture->height()),
                0.0,
                1.0
            };

            renderEncoder->setRenderPipelineState(data.pipelineState);
            renderEncoder->setViewport(viewport);
            renderEncoder->setCullMode(MTL::CullModeNone);
            renderEncoder->setFrontFacingWinding(MTL::WindingCounterClockwise);
            renderEncoder->setArgumentTable(data.argumentTable, MTL::RenderStageVertex);
            renderEncoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
            renderEncoder->endEncoding();
        });
}
