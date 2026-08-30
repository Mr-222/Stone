#include "TransparentObjectCullingPass.h"

#include <string>

#include "Core/Buffer.h"
#include "Core/IndirectCommandBuffer.h"
#include "Core/RenderGraph.h"
#include "Shader/ShaderTypes.h"
#include "Utility/Logger.h"
#include "Utility/ShaderLibrary.h"

// Reuses the same culling shader as the opaque path – the logic is identical,
// only the buffers bound through the argument table differ.
constexpr const char* kTransparentObjectCullingShaderLibrary = STONE_SHADER_DIR "/ObjectCulling.metallib";

struct TransparentObjectCullingPassData {
    RenderGraphResourceHandle transparentIndexBufferHandle;
    RenderGraphResourceHandle transparentIndexBufferInfoHandle;
    RenderGraphResourceHandle transparentRenderPrimitivesHandle;
    RenderGraphResourceHandle visibilityBufferHandle;
    RenderGraphResourceHandle executionRangeBufferHandle;
    RenderGraphResourceHandle icbArgumentBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
};

static NS::String* MakeTransparentFrameLabel(const char* name, const size_t frameSlot) {
    const std::string label = std::string(name) + " [frame slot " + std::to_string(frameSlot) + "]";
    return NS::String::string(label.c_str(), NS::UTF8StringEncoding);
}

void TransparentObjectCullingPass::Setup(MetalContext& context, const int numPrimitives) {
    m_numPrimitives = numPrimitives;
    // Ensure at least 1 so we always create valid GPU resources even with
    // zero transparent primitives in the scene.
    const int effectiveCount = std::max(numPrimitives, 1);

    MTL::Device* device = context.GetDevice();

    const ObjectCullingParams cullingParams {
        .primitiveCount = static_cast<uint32_t>(numPrimitives),
    };
    m_cullingParamsBuffer = std::make_unique<Buffer>(device, &cullingParams, sizeof(cullingParams), MTL::ResourceStorageModeShared);
    m_cullingParamsBuffer->GetNative()->setLabel(NS::String::string("Transparent Culling Params Buffer", NS::UTF8StringEncoding));

    NS::Error* error = nullptr;

    ShaderLibrary shaderLibrary = LoadShaderLibrary(device, kTransparentObjectCullingShaderLibrary, {
        "objectCulling_main",
    });

    MTL4::ComputePipelineDescriptor* pipelineDescriptor = MTL4::ComputePipelineDescriptor::alloc()->init()->autorelease();
    pipelineDescriptor->setComputeFunctionDescriptor(MakeLibraryFunctionDescriptor(shaderLibrary.GetLibrary(), "objectCulling_main"));

    MTL4::Compiler* compiler = device->newCompiler(MTL4::CompilerDescriptor::alloc()->init()->autorelease(), &error);
    LOG_ERROR_IF(!compiler, "Failed to create MTL::Compiler");
    MTL4::CompilerTaskOptions* taskOptions = MTL4::CompilerTaskOptions::alloc()->init()->autorelease();
    m_pipelineState = compiler->newComputePipelineState(pipelineDescriptor, taskOptions, &error);
    LOG_ERROR_IF(!m_pipelineState, "Failed to create transparent culling compute pipeline: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    MTL::Function* computeFunction = shaderLibrary.GetFunction("objectCulling_main");
    MTL::ArgumentEncoder* argumentEncoder = computeFunction->newArgumentEncoder(static_cast<NS::UInteger>(ObjectCullingBufferIndex::ICBContainer));
    LOG_ERROR_IF(!argumentEncoder, "Failed to create argument encoder for transparent culling ICB");

    auto icbDescriptor = MTL::IndirectCommandBufferDescriptor::alloc()->init()->autorelease();
    icbDescriptor->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    icbDescriptor->setInheritBuffers(true);
    icbDescriptor->setInheritPipelineState(true);
    icbDescriptor->setInheritDepthStencilState(true);

    m_frameResources.resize(context.GetFrameSlotCount());
    for (size_t frameSlot = 0; frameSlot < m_frameResources.size(); ++frameSlot) {
        FrameResources& resources = m_frameResources[frameSlot];

        resources.visibilityBuffer = std::make_unique<Buffer>(device, effectiveCount * sizeof(Visibility), MTL::ResourceStorageModePrivate);
        resources.visibilityBuffer->GetNative()->setLabel(MakeTransparentFrameLabel("Transparent Visibility Buffer", frameSlot));

        resources.executionRangeBuffer = std::make_unique<Buffer>(device, sizeof(IndirectCommandBufferExecutionRange), MTL::ResourceStorageModePrivate);
        resources.executionRangeBuffer->GetNative()->setLabel(MakeTransparentFrameLabel("Transparent Execution Range Buffer", frameSlot));

        resources.indirectCB = std::make_unique<IndirectCommandBuffer>(device, icbDescriptor, effectiveCount, MTL::ResourceStorageModePrivate);
        resources.indirectCB->GetNative()->setLabel(MakeTransparentFrameLabel("Transparent Scene ICB", frameSlot));

        resources.icbArgumentBuffer = std::make_unique<Buffer>(device, argumentEncoder->encodedLength(), MTL::ResourceStorageModeShared);
        resources.icbArgumentBuffer->GetNative()->setLabel(MakeTransparentFrameLabel("Transparent ICB Argument Buffer", frameSlot));

        argumentEncoder->setArgumentBuffer(resources.icbArgumentBuffer->GetNative(), 0);
        argumentEncoder->setIndirectCommandBuffer(resources.indirectCB->GetNative(), 0);
    }

    argumentEncoder->release();
    compiler->release();

    MTL4::ArgumentTableDescriptor* argumentTableDescriptor = MTL4::ArgumentTableDescriptor::alloc()->init()->autorelease();
    argumentTableDescriptor->setLabel(NS::String::string("Transparent Culling Argument Table", NS::UTF8StringEncoding));
    argumentTableDescriptor->setInitializeBindings(true);
    argumentTableDescriptor->setMaxBufferBindCount(static_cast<NS::UInteger>(ObjectCullingBufferIndex::MaxBufferBindCount));
    m_argumentTable = device->newArgumentTable(argumentTableDescriptor, &error);
    LOG_ERROR_IF(!m_argumentTable, "Failed to create argument table: {}", error ? error->localizedDescription()->utf8String() : "unknown error");

    m_argumentTable->setAddress(m_cullingParamsBuffer->GetGPUAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::CullingParams));
}

void TransparentObjectCullingPass::AddToGraph(RenderGraph& graph) {
    RenderGraphResourceHandle visibilityBufferHandle;
    RenderGraphResourceHandle executionRangeBufferHandle;
    RenderGraphResourceHandle icbArgumentBufferHandle;
    RenderGraphResourceHandle indirectCBHandle;
    for (uint32_t frameSlot = 0; frameSlot < m_frameResources.size(); ++frameSlot) {
        const FrameResources& resources = m_frameResources[frameSlot];
        visibilityBufferHandle = graph.RegisterFrameLocalBuffer("TransparentVisibilityBuffer", frameSlot, *resources.visibilityBuffer);
        executionRangeBufferHandle = graph.RegisterFrameLocalBuffer("TransparentExecutionRangeBuffer", frameSlot, *resources.executionRangeBuffer);
        icbArgumentBufferHandle = graph.RegisterFrameLocalBuffer("TransparentICBArgumentBuffer", frameSlot, *resources.icbArgumentBuffer);
        indirectCBHandle = graph.RegisterFrameLocalIndirectCommandBuffer("TransparentIndirectCommandBuffer", frameSlot, *resources.indirectCB);
    }

    RenderGraphResourceHandle transparentIndexBufferHandle = graph.DeclareBuffer("TransparentIndexBuffer");
    RenderGraphResourceHandle transparentIndexBufferInfoHandle = graph.DeclareBuffer("TransparentIndexBufferInfoBuffer");
    RenderGraphResourceHandle transparentRenderPrimitivesHandle = graph.DeclareBuffer("TransparentRenderPrimitiveBuffer");

    graph.AddPass<TransparentObjectCullingPassData>(
        "TransparentObjectCulling",
        IsCompute,
        [=](RenderGraphBuilder& builder, TransparentObjectCullingPassData& data, RenderGraphResources&) {
            data.transparentIndexBufferHandle = transparentIndexBufferHandle;
            data.transparentIndexBufferInfoHandle = transparentIndexBufferInfoHandle;
            data.transparentRenderPrimitivesHandle = transparentRenderPrimitivesHandle;
            data.visibilityBufferHandle = visibilityBufferHandle;
            data.executionRangeBufferHandle = executionRangeBufferHandle;
            data.icbArgumentBufferHandle = icbArgumentBufferHandle;
            data.indirectCBHandle = indirectCBHandle;

            builder.ReadBuffer(data.transparentIndexBufferHandle);
            builder.ReadBuffer(data.transparentIndexBufferInfoHandle);
            builder.ReadBuffer(data.transparentRenderPrimitivesHandle);
            builder.ReadBuffer(data.icbArgumentBufferHandle);
            builder.WriteBuffer(data.visibilityBufferHandle);
            builder.WriteBuffer(data.executionRangeBufferHandle);
            builder.WriteIndirectCommandBuffer(data.indirectCBHandle);
        },
        [this](const TransparentObjectCullingPassData& data, RenderGraphResources& resources, CommandBuffer& cmd) {
            if (m_numPrimitives == 0) return;

            MTL::Buffer* transparentIndexBuffer = resources.GetBuffer(data.transparentIndexBufferHandle);
            MTL::Buffer* transparentIndexBufferInfo = resources.GetBuffer(data.transparentIndexBufferInfoHandle);
            MTL::Buffer* transparentRenderPrimitives = resources.GetBuffer(data.transparentRenderPrimitivesHandle);
            MTL::Buffer* visibilityBuffer = resources.GetBuffer(data.visibilityBufferHandle);
            MTL::Buffer* executionRangeBuffer = resources.GetBuffer(data.executionRangeBufferHandle);
            MTL::Buffer* icbArgumentBuffer = resources.GetBuffer(data.icbArgumentBufferHandle);
            MTL::IndirectCommandBuffer* indirectCB = resources.GetIndirectCommandBuffer(data.indirectCBHandle);

            LOG_ERROR_IF(!transparentIndexBuffer, "TransparentCulling: Failed to get transparent index buffer");
            LOG_ERROR_IF(!transparentIndexBufferInfo, "TransparentCulling: Failed to get transparent index buffer info");
            LOG_ERROR_IF(!transparentRenderPrimitives, "TransparentCulling: Failed to get transparent render primitives");
            LOG_ERROR_IF(!visibilityBuffer, "TransparentCulling: Failed to get visibility buffer");
            LOG_ERROR_IF(!executionRangeBuffer, "TransparentCulling: Failed to get execution range buffer");
            LOG_ERROR_IF(!icbArgumentBuffer, "TransparentCulling: Failed to get ICB argument buffer");
            LOG_ERROR_IF(!indirectCB, "TransparentCulling: Failed to get indirect command buffer");

            m_argumentTable->setAddress(executionRangeBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::ExecutionRange));
            m_argumentTable->setAddress(transparentIndexBufferInfo->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::IndexBufferInfo));
            m_argumentTable->setAddress(transparentRenderPrimitives->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::RenderPrimitives));
            m_argumentTable->setAddress(visibilityBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::Visibilities));
            m_argumentTable->setAddress(icbArgumentBuffer->gpuAddress(), static_cast<NS::UInteger>(ObjectCullingBufferIndex::ICBContainer));

            cmd.AddResource(transparentIndexBuffer);
            cmd.AddResource(transparentIndexBufferInfo);
            cmd.AddResource(transparentRenderPrimitives);
            cmd.AddResource(executionRangeBuffer);
            cmd.AddResource(visibilityBuffer);
            cmd.AddResource(m_cullingParamsBuffer->GetNative());
            cmd.AddResource(indirectCB);
            cmd.AddResource(icbArgumentBuffer);

            MTL4::ComputeCommandEncoder* computeEncoder = cmd.BeginComputePass();
            LOG_ERROR_IF(!computeEncoder, "TransparentCulling: Failed to create compute encoder");

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
