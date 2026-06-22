#include "ObjectCullingPass.h"

#include "Core/Buffer.h"
#include "Core/IndirectCommandBuffer.h"
#include "Core/RenderGraph.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

constexpr const char* kObjetcCullingShaderLibrary = STONE_SHADER_DIR "/ObjectCulling.metallib";

struct ObjectCullingPassData {
    RenderGraphResourceHandle globalIndexBufferHandle;
    RenderGraphResourceHandle indexBufferInfoBufferHandle;
    RenderGraphResourceHandle renderObjectsBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
};

ObjectCullingPass::ObjectCullingPass() = default;

ObjectCullingPass::~ObjectCullingPass() = default;

void ObjectCullingPass::Setup(MetalContext& context, const int numObjects) {
    m_numObjects = numObjects;

    MTL::Device* device = context.GetDevice();

    m_visibilityBuffer = std::make_unique<Buffer>(
        device,
        numObjects * sizeof(uint32_t),
        MTL::ResourceStorageModeShared
    );
    m_visibilityBuffer->GetNative()->setLabel(NS::String::string("Visibility Buffer", NS::UTF8StringEncoding));

    m_ICBExecutionRangeBuffer = std::make_unique<Buffer>(
        device,
        2 * sizeof(uint32_t),
        MTL::ResourceStorageModeShared
    );
    m_ICBExecutionRangeBuffer->GetNative()->setLabel(NS::String::string("Execution Range Buffer", NS::UTF8StringEncoding));

    auto icbDescriptor = MTL::IndirectCommandBufferDescriptor::alloc()->init()->autorelease();
    icbDescriptor->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    icbDescriptor->setInheritBuffers(true);
    icbDescriptor->setInheritPipelineState(true);
    m_indirectCB = std::make_unique<IndirectCommandBuffer>(device, icbDescriptor, m_numObjects, MTL::ResourceStorageModePrivate);
    m_indirectCB->GetNative()->setLabel(NS::String::string("Scene ICB", NS::UTF8StringEncoding));

    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kObjetcCullingShaderLibrary, {
        "objectCulling_main",
    });

    MTL4::ComputePipelineDescriptor* pipelineDescriptor = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();

    pipelineDescriptor->setComputeFunctionDescriptor(
        MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "objectCulling_main")
    );

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newComputePipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create compute pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::Function* computeFunction = shaderLibrary.GetFunction("objectCulling_main");

    MTL::ArgumentEncoder* argumentEncoder = computeFunction->newArgumentEncoder(static_cast<NS::UInteger>(ObjectCullingBufferIndex::ICBContainer));
    LOG_ERROR_IF(!argumentEncoder, "Failed to create argument encoder for object culling's ICB");
    m_icbArgumentBuffer = std::make_unique<Buffer>(
        device,
        argumentEncoder->encodedLength(),
        MTL::ResourceStorageModeShared
    );
    m_icbArgumentBuffer->GetNative()->setLabel(NS::String::string("ICB Argument Buffer", NS::UTF8StringEncoding));

    argumentEncoder->setArgumentBuffer(m_icbArgumentBuffer->GetNative(), 0);
    argumentEncoder->setIndirectCommandBuffer(m_indirectCB->GetNative(), 0);
    argumentEncoder->release();
    compiler->release();

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("Object Culling pass argument table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(ObjectCullingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_ICBExecutionRangeBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::ExecutionRange));
    m_argumentTable->setAddress(m_visibilityBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::Visibilities));
    m_argumentTable->setAddress(m_icbArgumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::ICBContainer));
}

void ObjectCullingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle visibilitiesBufferHandle = graph.RegisterBuffer("VisibilityBuffer", *m_visibilityBuffer);
    RenderGraphResourceHandle executionRangeBufferHandle  = graph.RegisterBuffer("ExecutionRangeBuffer", *m_ICBExecutionRangeBuffer);
    RenderGraphResourceHandle indirectCBHandle = graph.RegisterIndirectCommandBuffer("IndirectCommandBuffer", *m_indirectCB);
    RenderGraphResourceHandle globalIndexBufferHandle = graph.DeclareBuffer("GlobalIndexBuffer");
    RenderGraphResourceHandle indexBufferInfoBufferHandle = graph.DeclareBuffer("IndexBufferInfoBuffer");
    RenderGraphResourceHandle renderObjectsBufferHandle = graph.DeclareBuffer("RenderObjectBuffer");

    graph.AddPass<ObjectCullingPassData>(
        "ObjectCulling",
        [=, this](RenderGraphBuilder& builder, ObjectCullingPassData& data, RenderGraphResources& resources) {
            data.globalIndexBufferHandle = globalIndexBufferHandle;
            data.indexBufferInfoBufferHandle = indexBufferInfoBufferHandle;
            data.renderObjectsBufferHandle = renderObjectsBufferHandle;
            data.indirectCBHandle = indirectCBHandle;

            builder.ReadBuffer(data.globalIndexBufferHandle);
            builder.ReadBuffer(data.indexBufferInfoBufferHandle);
            builder.ReadBuffer(data.renderObjectsBufferHandle);
            builder.WriteBuffer(visibilitiesBufferHandle);
            builder.WriteBuffer(executionRangeBufferHandle);
            builder.WriteIndirectCommandBuffer(data.indirectCBHandle);

            MTL::Buffer* indexBufferInfoBuffer = resources.GetBuffer(indexBufferInfoBufferHandle);
            m_argumentTable->setAddress(indexBufferInfoBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::IndexBufferInfo));

            MTL::Buffer* renderObjectBuffer = resources.GetBuffer(renderObjectsBufferHandle);
            m_argumentTable->setAddress(renderObjectBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::RenderObjects));
        },
        [this](const ObjectCullingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* globalIndexBuffer = resources.GetBuffer(data.globalIndexBufferHandle);
            LOG_ERROR_IF(!globalIndexBuffer, "Failed to get global index buffer");
            MTL::Buffer* indexBufferInfoBuffer = resources.GetBuffer(data.indexBufferInfoBufferHandle);
            LOG_ERROR_IF(!indexBufferInfoBuffer, "Failed to get index buffer info");
            MTL::Buffer* renderObjectsBuffer = resources.GetBuffer(data.renderObjectsBufferHandle);
            LOG_ERROR_IF(!renderObjectsBuffer, "Failed to get render objects buffer");
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);
            LOG_ERROR_IF(!indirectCB, "Failed to get indirect command buffer");

            cmd.AddResource(globalIndexBuffer);
            cmd.AddResource(indexBufferInfoBuffer);
            cmd.AddResource(renderObjectsBuffer);
            cmd.AddResource(m_ICBExecutionRangeBuffer->GetNative());
            cmd.AddResource(m_visibilityBuffer->GetNative());
            cmd.AddResource(indirectCB);
            cmd.AddResource(m_icbArgumentBuffer->GetNative());

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();
            LOG_ERROR_IF(!computeEncoder, "Failed to create compute command encoder");

            // Reset ICB
            memset(m_visibilityBuffer->GetNative()->contents(), 0, m_visibilityBuffer->GetSize());
            memset(m_ICBExecutionRangeBuffer->GetNative()->contents(), 0, m_ICBExecutionRangeBuffer->GetSize());
            computeEncoder->resetCommandsInBuffer(indirectCB, NS::Range(0, m_numObjects));

            computeEncoder->setComputePipelineState(m_pipelineState);
            computeEncoder->setArgumentTable(m_argumentTable);
            NS::UInteger width = m_pipelineState->threadExecutionWidth();
            MTL::Size gridSize = MTL::Size(m_numObjects, 1, 1);
            MTL::Size threadGroupSize = MTL::Size(width, 1, 1);
            computeEncoder->dispatchThreads(gridSize, threadGroupSize);
            computeEncoder->endEncoding();
        });
}
