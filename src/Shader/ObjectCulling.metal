#include "ShaderTypes.h"

bool IsVisible(const float4x4 viewProjection, const float3 center, const float3 extents) {
    const float4x4 rows = transpose(viewProjection);
    const float4 planes[6] = {
        rows[3] + rows[0], // left
        rows[3] - rows[0], // right
        rows[3] + rows[1], // bottom
        rows[3] - rows[1], // top
        rows[2],           // near
        rows[3] - rows[2], // far
    };

    for (uint i = 0; i < 6; ++i) {
        if (dot(planes[i].xyz, center) + planes[i].w + dot(abs(planes[i].xyz), extents) < 0.0f)
            return false;
    }
    return true;
}

kernel void objectCulling_main(
    uint primitiveID [[thread_position_in_grid]],
    constant FrameUniform& frameUniform [[buffer(ObjectCullingBufferIndex::FrameUniform)]],
    constant ObjectCullingParams& params [[buffer(ObjectCullingBufferIndex::CullingParams)]],
    constant IndexBufferInfo& indexBufferInfo [[buffer(ObjectCullingBufferIndex::IndexBufferInfo)]],
    const device GPURenderPrimitive* primitives [[buffer(ObjectCullingBufferIndex::RenderPrimitives)]],
    device ObjectCullingICBContainer& icb [[buffer(ObjectCullingBufferIndex::ICBContainer)]])
{
    if (primitiveID >= params.primitiveCount)
        return;

    const device GPURenderPrimitive& primitive = primitives[primitiveID];

    if (!IsVisible(frameUniform.viewProjection, primitive.boundsCenter.xyz, primitive.boundsExtents.xyz))
        return;

    render_command cmd(icb.commandBuffer, primitiveID);
    cmd.draw_indexed_primitives(primitive_type::triangle, primitive.indexCount, (const device uint*)(indexBufferInfo.addr + primitive.firstIndex * sizeof(uint32_t)), 1, primitive.baseVertex, primitiveID);
}
