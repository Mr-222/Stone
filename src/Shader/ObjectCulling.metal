#include "ShaderTypes.h"

// TODO: Implement actual culling logic
bool IsVisible() {
    return true;
}

kernel void objectCulling_main(
    uint objID [[thread_position_in_grid]],
    device IndirectCommandBufferExecutionRange& executionRange [[buffer(ObjectCullingBufferIndex::ExecutionRange)]],
    constant ObjectCullingParams& params [[buffer(ObjectCullingBufferIndex::CullingParams)]],
    constant IndexBufferInfo& indexBufferInfo [[buffer(ObjectCullingBufferIndex::IndexBufferInfo)]],
    device GPURenderObject* objects [[buffer(ObjectCullingBufferIndex::RenderObjects)]],
    device Visibility* visibilities [[buffer(ObjectCullingBufferIndex::Visibilities)]],
    device ObjectCullingICBContainer& icb [[buffer(ObjectCullingBufferIndex::ICBContainer)]])
{
    if (objID >= params.objectCount)
        return;

    device GPURenderObject& obj = objects[objID];

    if (IsVisible()) {
        uint localSlot = atomic_fetch_add_explicit(&executionRange.length, 1, memory_order_relaxed);
        uint slot = localSlot + executionRange.location;

        visibilities[localSlot].objID = objID;

        render_command cmd(icb.commandBuffer, slot);
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    obj.indexCount,
                                    (const device uint*)(indexBufferInfo.addr + obj.firstIndex * sizeof(uint32_t)),
                                    1,
                                    obj.baseVertex);
    }
}
