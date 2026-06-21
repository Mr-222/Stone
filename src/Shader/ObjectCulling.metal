#include <metal_stdlib>
using namespace metal;

struct RenderObject
{
    uint32_t baseVertex;
    uint32_t firstIndex;
    uint32_t indexCount;
    uint32_t pad0;
    float4x4 worldMat;
};

struct IndirectCommandBufferExecutionRange
{
    uint32_t location;
    atomic_uint length;
};

struct IndexBufferInfo
{
    uint64_t addr;
};

struct Visibility
{
    uint32_t objID;
};

struct ICBContainer
{
    command_buffer commandBuffer [[id(0)]];
};

// TODO: Implement actual culling logic
bool IsVisible() {
    return true;
}

kernel void objectCulling_main(
    uint objID [[thread_position_in_grid]],
    device IndirectCommandBufferExecutionRange& executionRange [[buffer(0)]],
    constant IndexBufferInfo& indexBufferInfo [[buffer(1)]],
    device RenderObject* objects [[buffer(2)]],
    device Visibility* visibilities[[buffer(3)]],
    device ICBContainer& icb [[buffer(4)]])
{
    device RenderObject& obj = objects[objID];

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
