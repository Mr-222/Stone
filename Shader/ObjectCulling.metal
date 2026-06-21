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
    uint32_t length;
    uint32_t pad0;
};

// TODO: Implement actual culling logic
bool IsVisible() {
    return true;
}

kernel void main_cull(
    uint objID [[thread_position_in_grid]],
    device IndirectCommandBufferExecutionRange& executionRange [[buffer(0)]],
    constant IndexBufferInfo& indexBufferinfo [[buffer(1)]],
    device RenderObject* objects [[buffer(2)]],
    indirect_command_buffer icb [[buffer(3)]])
{
    device RenderObject& obj = objects[objID];
    
    if (IsVisible()) {
        uint localSlot = atomic_fetch_add_explicit(&executionRange.length, 1, memory_order_relaxed);
        uint slot = localSlot + executionRange.location;
        
        render_command cmd(icb, slot);
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    obj.indexCount,
                                    index_type::uint32,
                                    indexBufferinfo.addr + obj.firstIndex * sizeof(uint32_t),
                                    indexBufferinfo.length,
                                    1,
                                    obj.baseVertex,
                                    0);
    }
}
