#include "ObjectCullingPass.h"

#include <string>

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
    RenderGraphResourceHandle visibilityBufferHandle;
    RenderGraphResourceHandle executionRangeBufferHandle;
    RenderGraphResourceHandle icbArgumentBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
};

NS::String* MakeFrameResourceLabel(const char* name, const size_t frameSlot) {
    const std::string label = std::string(name) + " [frame slot " + std::to_string(frameSlot) + "]";
    return NS::String::string(label.c_str(), NS::UTF8StringEncoding);
}

ObjectCullingPass::ObjectCullingPass() = default;

ObjectCullingPass::~ObjectCullingPass() = default;

void ObjectCullingPass::Setup(MetalContext& context, const int numPrimitives) {
    m_numPrimitives = numPrimitives;

    MTL::Device* device = context.GetDevice();

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

    auto icbDescriptor = MTL::IndirectCommandBufferDescriptor::alloc()->init()->autorelease();
    icbDescriptor->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    icbDescriptor->setInheritBuffers(true);
    icbDescriptor->setInheritPipelineState(true);
    icbDescriptor->setInheritDepthStencilState(true);

    m_frameResources.resize(context.GetFrameSlotCount());
    for (size_t frameSlot = 0; frameSlot < m_frameResources.size(); ++frameSlot) {
        FrameResources& resources = m_frameResources[frameSlot];

        resources.visibilityBuffer = std::make_unique<Buffer>(
            device,
            numPrimitives * sizeof(Visibility),
            MTL::ResourceStorageModePrivate);
        resources.visibilityBuffer->GetNative()->setLabel(MakeFrameResourceLabel("Visibility Buffer", frameSlot));

        resources.executionRangeBuffer = std::make_unique<Buffer>(
            device,
            sizeof(IndirectCommandBufferExecutionRange),
            MTL::ResourceStorageModePrivate);
        resources.executionRangeBuffer->GetNative()->setLabel(MakeFrameResourceLabel("Execution Range Buffer", frameSlot));

        resources.indirectCB = std::make_unique<IndirectCommandBuffer>(
            device,
            icbDescriptor,
            m_numPrimitives,
            MTL::ResourceStorageModePrivate);
        resources.indirectCB->GetNative()->setLabel(MakeFrameResourceLabel("Scene ICB", frameSlot));

        resources.icbArgumentBuffer = std::make_unique<Buffer>(
            device,
            argumentEncoder->encodedLength(),
            MTL::ResourceStorageModeShared);
        resources.icbArgumentBuffer->GetNative()->setLabel(MakeFrameResourceLabel("ICB Argument Buffer", frameSlot));

        argumentEncoder->setArgumentBuffer(resources.icbArgumentBuffer->GetNative(), 0);
        argumentEncoder->setIndirectCommandBuffer(resources.indirectCB->GetNative(), 0);
    }

    argumentEncoder->release();
    compiler->release();

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("Object Culling pass argument table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(ObjectCullingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_cullingParamsBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::CullingParams));
}

void ObjectCullingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle visibilitiesBufferHandle;
    RenderGraphResourceHandle executionRangeBufferHandle;
    RenderGraphResourceHandle icbArgumentBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
    for (uint32_t frameSlot = 0; frameSlot < m_frameResources.size(); ++frameSlot) {
        const FrameResources& resources = m_frameResources[frameSlot];
        visibilitiesBufferHandle = graph.RegisterFrameLocalBuffer(
            "VisibilityBuffer", frameSlot, *resources.visibilityBuffer);
        executionRangeBufferHandle = graph.RegisterFrameLocalBuffer(
            "ExecutionRangeBuffer", frameSlot, *resources.executionRangeBuffer);
        icbArgumentBufferHandle = graph.RegisterFrameLocalBuffer(
            "ObjectCullingICBArgumentBuffer", frameSlot, *resources.icbArgumentBuffer);
        indirectCBHandle = graph.RegisterFrameLocalIndirectCommandBuffer(
            "IndirectCommandBuffer", frameSlot, *resources.indirectCB);
    }

    RenderGraphResourceHandle globalIndexBufferHandle = graph.DeclareBuffer("GlobalIndexBuffer");
    RenderGraphResourceHandle indexBufferInfoBufferHandle = graph.DeclareBuffer("IndexBufferInfoBuffer");
    RenderGraphResourceHandle renderPrimitivesBufferHandle = graph.DeclareBuffer("RenderPrimitiveBuffer");

    graph.AddPass<ObjectCullingPassData>(
        "ObjectCulling",
        IsCompute,
        [=](RenderGraphBuilder& builder, ObjectCullingPassData& data, RenderGraphResources&) {
            data.globalIndexBufferHandle = globalIndexBufferHandle;
            data.indexBufferInfoBufferHandle = indexBufferInfoBufferHandle;
            data.renderPrimitivesBufferHandle = renderPrimitivesBufferHandle;
            data.visibilityBufferHandle = visibilitiesBufferHandle;
            data.executionRangeBufferHandle = executionRangeBufferHandle;
            data.icbArgumentBufferHandle = icbArgumentBufferHandle;
            data.indirectCBHandle = indirectCBHandle;

            builder.ReadBuffer(data.globalIndexBufferHandle);
            builder.ReadBuffer(data.indexBufferInfoBufferHandle);
            builder.ReadBuffer(data.renderPrimitivesBufferHandle);
            builder.ReadBuffer(data.icbArgumentBufferHandle);
            builder.WriteBuffer(data.visibilityBufferHandle);
            builder.WriteBuffer(data.executionRangeBufferHandle);
            builder.WriteIndirectCommandBuffer(data.indirectCBHandle);
        },
        [this](const ObjectCullingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            MTL::Buffer* globalIndexBuffer = resources.GetBuffer(data.globalIndexBufferHandle);
            LOG_ERROR_IF(!globalIndexBuffer, "Failed to get global index buffer");
            MTL::Buffer* indexBufferInfoBuffer = resources.GetBuffer(data.indexBufferInfoBufferHandle);
            LOG_ERROR_IF(!indexBufferInfoBuffer, "Failed to get index buffer info");
            MTL::Buffer* renderPrimitivesBuffer = resources.GetBuffer(data.renderPrimitivesBufferHandle);
            LOG_ERROR_IF(!renderPrimitivesBuffer, "Failed to get render primitives buffer");
            MTL::Buffer* visibilityBuffer = resources.GetBuffer(data.visibilityBufferHandle);
            LOG_ERROR_IF(!visibilityBuffer, "Failed to get visibility buffer");
            MTL::Buffer* executionRangeBuffer = resources.GetBuffer(data.executionRangeBufferHandle);
            LOG_ERROR_IF(!executionRangeBuffer, "Failed to get ICB execution range buffer");
            MTL::Buffer* icbArgumentBuffer = resources.GetBuffer(data.icbArgumentBufferHandle);
            LOG_ERROR_IF(!icbArgumentBuffer, "Failed to get ICB argument buffer");
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);
            LOG_ERROR_IF(!indirectCB, "Failed to get indirect command buffer");

            m_argumentTable->setAddress(
                executionRangeBuffer->gpuAddress(),
                static_cast<NS::UInteger>(ObjectCullingBufferIndex::ExecutionRange));
            m_argumentTable->setAddress(
                indexBufferInfoBuffer->gpuAddress(),
                static_cast<NS::UInteger>(ObjectCullingBufferIndex::IndexBufferInfo));
            m_argumentTable->setAddress(
                renderPrimitivesBuffer->gpuAddress(),
                static_cast<NS::UInteger>(ObjectCullingBufferIndex::RenderPrimitives));
            m_argumentTable->setAddress(
                visibilityBuffer->gpuAddress(),
                static_cast<NS::UInteger>(ObjectCullingBufferIndex::Visibilities));
            m_argumentTable->setAddress(
                icbArgumentBuffer->gpuAddress(),
                static_cast<NS::UInteger>(ObjectCullingBufferIndex::ICBContainer));

            cmd.AddResource(globalIndexBuffer);
            cmd.AddResource(indexBufferInfoBuffer);
            cmd.AddResource(renderPrimitivesBuffer);
            cmd.AddResource(executionRangeBuffer);
            cmd.AddResource(visibilityBuffer);
            cmd.AddResource(m_cullingParamsBuffer->GetNative());
            cmd.AddResource(indirectCB);
            cmd.AddResource(icbArgumentBuffer);

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();
            LOG_ERROR_IF(!computeEncoder, "Failed to create compute command encoder");

            computeEncoder->fillBuffer(visibilityBuffer, NS::Range(0, visibilityBuffer->length()), 0);
            computeEncoder->fillBuffer(executionRangeBuffer, NS::Range(0, executionRangeBuffer->length()), 0);
            computeEncoder->resetCommandsInBuffer(indirectCB, NS::Range(0, m_numPrimitives));

            computeEncoder->barrierAfterEncoderStages(MTL::StageBlit, MTL::StageDispatch, MTL4::VisibilityOptionDevice);

            computeEncoder->setComputePipelineState(m_pipelineState);
            computeEncoder->setArgumentTable(m_argumentTable);
            NS::UInteger width = m_pipelineState->threadExecutionWidth();
            MTL::Size gridSize = MTL::Size(m_numPrimitives, 1, 1);
            MTL::Size threadGroupSize = MTL::Size(width, 1, 1);
            computeEncoder->dispatchThreads(gridSize, threadGroupSize);

            computeEncoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);

            computeEncoder->optimizeIndirectCommandBuffer(indirectCB, NS::Range(0, m_numPrimitives));
            computeEncoder->endEncoding();
        });
}
