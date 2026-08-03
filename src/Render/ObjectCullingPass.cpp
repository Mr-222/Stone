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
    RenderGraphResourceHandle renderPrimitivesBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
};

ObjectCullingPass::ObjectCullingPass() = default;

ObjectCullingPass::~ObjectCullingPass() = default;

void ObjectCullingPass::Setup(MetalContext& context, const int numPrimitives) {
    m_numPrimitives = numPrimitives;

    MTL::Device* device = context.GetDevice();

    m_visibilityBuffer = std::make_unique<Buffer>(
        device,
        numPrimitives * sizeof(Visibility),
        MTL::ResourceStorageModePrivate
    );
    m_visibilityBuffer->GetNative()->setLabel(NS::String::string("Visibility Buffer", NS::UTF8StringEncoding));

    const ObjectCullingParams cullingParams {
        .primitiveCount = static_cast<uint32_t>(numPrimitives),
    };
    m_cullingParamsBuffer = std::make_unique<Buffer>(
        device,
        &cullingParams,
        sizeof(cullingParams),
        MTL::ResourceStorageModeShared
    );
    m_cullingParamsBuffer->GetNative()->setLabel(NS::String::string("Object Culling Params Buffer", NS::UTF8StringEncoding));

    m_ICBExecutionRangeBuffer = std::make_unique<Buffer>(
        device,
        2 * sizeof(uint32_t),
        MTL::ResourceStorageModePrivate
    );
    m_ICBExecutionRangeBuffer->GetNative()->setLabel(NS::String::string("Execution Range Buffer", NS::UTF8StringEncoding));

    auto icbDescriptor = MTL::IndirectCommandBufferDescriptor::alloc()->init()->autorelease();
    icbDescriptor->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    icbDescriptor->setInheritBuffers(true);
    icbDescriptor->setInheritPipelineState(true);
    m_indirectCB = std::make_unique<IndirectCommandBuffer>(device, icbDescriptor, m_numPrimitives, MTL::ResourceStorageModePrivate);
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
    m_argumentTable->setAddress(m_cullingParamsBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::CullingParams));
    m_argumentTable->setAddress(m_visibilityBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::Visibilities));
    m_argumentTable->setAddress(m_icbArgumentBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::ICBContainer));
}

void ObjectCullingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle visibilitiesBufferHandle = graph.RegisterBuffer("VisibilityBuffer", *m_visibilityBuffer);
    RenderGraphResourceHandle executionRangeBufferHandle  = graph.RegisterBuffer("ExecutionRangeBuffer", *m_ICBExecutionRangeBuffer);
    RenderGraphResourceHandle indirectCBHandle = graph.RegisterIndirectCommandBuffer("IndirectCommandBuffer", *m_indirectCB);
    RenderGraphResourceHandle globalIndexBufferHandle = graph.DeclareBuffer("GlobalIndexBuffer");
    RenderGraphResourceHandle indexBufferInfoBufferHandle = graph.DeclareBuffer("IndexBufferInfoBuffer");
    RenderGraphResourceHandle renderPrimitivesBufferHandle = graph.DeclareBuffer("RenderPrimitiveBuffer");

    graph.AddPass<ObjectCullingPassData>(
        "ObjectCulling",
        IsCompute,
        [=, this](RenderGraphBuilder& builder, ObjectCullingPassData& data, RenderGraphResources& resources) {
            data.globalIndexBufferHandle = globalIndexBufferHandle;
            data.indexBufferInfoBufferHandle = indexBufferInfoBufferHandle;
            data.renderPrimitivesBufferHandle = renderPrimitivesBufferHandle;
            data.indirectCBHandle = indirectCBHandle;

            builder.ReadBuffer(data.globalIndexBufferHandle);
            builder.ReadBuffer(data.indexBufferInfoBufferHandle);
            builder.ReadBuffer(data.renderPrimitivesBufferHandle);
            builder.WriteBuffer(visibilitiesBufferHandle);
            builder.WriteBuffer(executionRangeBufferHandle);
            builder.WriteIndirectCommandBuffer(data.indirectCBHandle);

            MTL::Buffer* indexBufferInfoBuffer = resources.GetBuffer(indexBufferInfoBufferHandle);
            m_argumentTable->setAddress(indexBufferInfoBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::IndexBufferInfo));

            MTL::Buffer* renderPrimitiveBuffer = resources.GetBuffer(renderPrimitivesBufferHandle);
            m_argumentTable->setAddress(renderPrimitiveBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::RenderPrimitives));
        },
        [this](const ObjectCullingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* globalIndexBuffer = resources.GetBuffer(data.globalIndexBufferHandle);
            LOG_ERROR_IF(!globalIndexBuffer, "Failed to get global index buffer");
            MTL::Buffer* indexBufferInfoBuffer = resources.GetBuffer(data.indexBufferInfoBufferHandle);
            LOG_ERROR_IF(!indexBufferInfoBuffer, "Failed to get index buffer info");
            MTL::Buffer* renderPrimitivesBuffer = resources.GetBuffer(data.renderPrimitivesBufferHandle);
            LOG_ERROR_IF(!renderPrimitivesBuffer, "Failed to get render primitives buffer");
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);
            LOG_ERROR_IF(!indirectCB, "Failed to get indirect command buffer");

            cmd.AddResource(globalIndexBuffer);
            cmd.AddResource(indexBufferInfoBuffer);
            cmd.AddResource(renderPrimitivesBuffer);
            cmd.AddResource(m_ICBExecutionRangeBuffer->GetNative());
            cmd.AddResource(m_visibilityBuffer->GetNative());
            cmd.AddResource(m_cullingParamsBuffer->GetNative());
            cmd.AddResource(indirectCB);
            cmd.AddResource(m_icbArgumentBuffer->GetNative());

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();
            LOG_ERROR_IF(!computeEncoder, "Failed to create compute command encoder");

            computeEncoder->fillBuffer(m_visibilityBuffer->GetNative(), NS::Range(0, m_visibilityBuffer->GetSize()), 0);
            computeEncoder->fillBuffer(m_ICBExecutionRangeBuffer->GetNative(), NS::Range(0, m_ICBExecutionRangeBuffer->GetSize()), 0);
            computeEncoder->resetCommandsInBuffer(indirectCB, NS::Range(0, m_numPrimitives));

            computeEncoder->setComputePipelineState(m_pipelineState);
            computeEncoder->setArgumentTable(m_argumentTable);
            NS::UInteger width = m_pipelineState->threadExecutionWidth();
            MTL::Size gridSize = MTL::Size(m_numPrimitives, 1, 1);
            MTL::Size threadGroupSize = MTL::Size(width, 1, 1);
            computeEncoder->dispatchThreads(gridSize, threadGroupSize);
            computeEncoder->optimizeIndirectCommandBuffer(indirectCB, NS::Range(0, m_numPrimitives));
            computeEncoder->endEncoding();
        });
}
