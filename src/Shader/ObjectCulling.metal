#include "ShaderTypes.h"

// TODO: Implement actual culling logic
bool IsVisible() {
    return true;
}

kernel void objectCulling_main(
    uint primitiveID [[thread_position_in_grid]],
    device IndirectCommandBufferExecutionRange& executionRange [[buffer(ObjectCullingBufferIndex::ExecutionRange)]],
    constant ObjectCullingParams& params [[buffer(ObjectCullingBufferIndex::CullingParams)]],
    constant IndexBufferInfo& indexBufferInfo [[buffer(ObjectCullingBufferIndex::IndexBufferInfo)]],
    device GPURenderPrimitive* primitives [[buffer(ObjectCullingBufferIndex::RenderPrimitives)]],
    device Visibility* visibilities [[buffer(ObjectCullingBufferIndex::Visibilities)]],
    device ObjectCullingICBContainer& icb [[buffer(ObjectCullingBufferIndex::ICBContainer)]])
{
    if (primitiveID >= params.primitiveCount)
        return;

    device GPURenderPrimitive& primitive = primitives[primitiveID];

    if (IsVisible()) {
        uint localSlot = atomic_fetch_add_explicit(&executionRange.length, 1, memory_order_relaxed);
        uint slot = localSlot + executionRange.location;

        visibilities[localSlot].primitiveID = primitiveID;

        render_command cmd(icb.commandBuffer, slot);
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    primitive.indexCount,
                                    (const device uint*)(indexBufferInfo.addr + primitive.firstIndex * sizeof(uint32_t)),
                                    1,
                                    primitive.baseVertex,
                                    primitiveID);
    }
}
