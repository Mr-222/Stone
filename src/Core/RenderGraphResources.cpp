#include "RenderGraphResources.h"

#include "Utility/Logger.h"

RenderGraphResources::RenderGraphResources(const uint32_t frameSlotCount)
    : m_frameSlotCount(frameSlotCount)
{
    LOG_ERROR_IF(frameSlotCount == 0, "Render graph requires at least one frame slot.");
}

void RenderGraphResources::Clear() {
    m_resources.clear();
    m_resourcesLUT.clear();
}

void RenderGraphResources::SetCurrentFrameSlot(const uint32_t frameSlot) {
    LOG_ERROR_IF(frameSlot >= m_frameSlotCount,
        "Render graph frame slot {} exceeds the configured frame slot count {}.",
        frameSlot,
        m_frameSlotCount);
    m_currentFrameSlot = frameSlot;
}

RenderGraphResourceHandle RenderGraphResources::Declare(std::string name, const RenderGraphResourceType type) {
    if (m_resourcesLUT.contains(name)) {
        LOG_ERROR_IF(m_resources[m_resourcesLUT[name].index].type != type,
            "Render graph resource '{}' was redeclared with a different type.", name);
        return m_resourcesLUT[name];
    }

    m_resources.emplace_back(std::move(name), type);

    RenderGraphResourceHandle handle { static_cast<uint32_t>(m_resources.size() - 1) };
    m_resourcesLUT[m_resources.back().name] = handle;
    return handle;
}

RenderGraphPhysicalResource& RenderGraphResources::PreparePhysicalResource(
    const RenderGraphResourceHandle handle,
    const RenderGraphResourceScope scope,
    const uint32_t frameSlot)
{
    RenderGraphResource& resource = m_resources[handle.index];
    LOG_ERROR_IF(resource.scope != RenderGraphResourceScope::Unspecified && resource.scope != scope,
        "Render graph resource '{}' cannot be both persistent and frame-local.",
        resource.name);

    if (resource.scope == RenderGraphResourceScope::Unspecified) {
        resource.scope = scope;
        const size_t physicalResourceCount = scope == RenderGraphResourceScope::FrameLocal ? m_frameSlotCount : 1;
        resource.physicalResources.resize(physicalResourceCount);
    }

    const uint32_t physicalResourceIndex = scope == RenderGraphResourceScope::FrameLocal ? frameSlot : 0;
    LOG_ERROR_IF(physicalResourceIndex >= resource.physicalResources.size(),
        "Frame slot {} for render graph resource '{}' exceeds the configured frame slot count {}.",
        frameSlot,
        resource.name,
        m_frameSlotCount);
    return resource.physicalResources[physicalResourceIndex];
}

const RenderGraphPhysicalResource* RenderGraphResources::GetCurrentPhysicalResource(const RenderGraphResourceHandle handle) const
{
    if (!handle.IsValid() || handle.index >= m_resources.size())
        return nullptr;

    const RenderGraphResource& resource = m_resources[handle.index];
    if (resource.scope == RenderGraphResourceScope::Unspecified)
        return nullptr;

    const uint32_t physicalResourceIndex = resource.scope == RenderGraphResourceScope::FrameLocal ? m_currentFrameSlot : 0;
    if (physicalResourceIndex >= resource.physicalResources.size())
        return nullptr;

    return &resource.physicalResources[physicalResourceIndex];
}

bool RenderGraphResources::IsHandleDeclared(
    const RenderGraphResourceHandle handle,
    const RenderGraphResourceType type) const
{
    return handle.IsValid() &&
           handle.index < m_resources.size() &&
           m_resources[handle.index].type == type;
}

RenderGraphResourceHandle RenderGraphResources::DeclareTexture(std::string name) {
    return Declare(std::move(name), RenderGraphResourceType::Texture);
}

RenderGraphResourceHandle RenderGraphResources::RegisterTexture(std::string name, const Texture& texture) {
    MTL::Texture* nativeTexture = texture.GetNative();
    LOG_ERROR_IF(!nativeTexture, "Cannot register null texture '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareTexture(name);
    PreparePhysicalResource(handle, RenderGraphResourceScope::Persistent, 0).texture = nativeTexture;

    return handle;
}

RenderGraphResourceHandle RenderGraphResources::RegisterFrameLocalTexture(
    std::string name,
    const uint32_t frameSlot,
    const Texture& texture)
{
    MTL::Texture* nativeTexture = texture.GetNative();
    LOG_ERROR_IF(!nativeTexture, "Cannot register null frame-local texture '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareTexture(name);
    PreparePhysicalResource(handle, RenderGraphResourceScope::FrameLocal, frameSlot).texture = nativeTexture;

    return handle;
}

MTL::Texture* RenderGraphResources::GetTexture(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!IsTextureHandleValid(handle), "Invalid render graph texture handle {}.", handle.index);
    return GetCurrentPhysicalResource(handle)->texture;
}

RenderGraphResourceHandle RenderGraphResources::GetTextureHandle(const std::string& name) const {
    const auto it = m_resourcesLUT.find(name);
    LOG_ERROR_IF(it == m_resourcesLUT.end(), "Texture '{}' does not exist.", name);
    assert(m_resources[it->second.index].type == RenderGraphResourceType::Texture);
    return it->second;
}

RenderGraphResourceHandle RenderGraphResources::DeclareBuffer(std::string name) {
    return Declare(std::move(name), RenderGraphResourceType::Buffer);
}

RenderGraphResourceHandle RenderGraphResources::RegisterBuffer(std::string name, const Buffer& buffer) {
    MTL::Buffer* nativeBuffer = buffer.GetNative();
    LOG_ERROR_IF(!nativeBuffer, "Cannot register null buffer '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareBuffer(name);
    PreparePhysicalResource(handle, RenderGraphResourceScope::Persistent, 0).buffer = nativeBuffer;

    return handle;
}

RenderGraphResourceHandle RenderGraphResources::RegisterFrameLocalBuffer(
    std::string name,
    const uint32_t frameSlot,
    const Buffer& buffer)
{
    MTL::Buffer* nativeBuffer = buffer.GetNative();
    LOG_ERROR_IF(!nativeBuffer, "Cannot register null frame-local buffer '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareBuffer(name);
    PreparePhysicalResource(handle, RenderGraphResourceScope::FrameLocal, frameSlot).buffer = nativeBuffer;

    return handle;
}

MTL::Buffer *RenderGraphResources::GetBuffer(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!IsBufferHandleValid(handle), "Invalid render graph buffer handle {}.", handle.index);
    return GetCurrentPhysicalResource(handle)->buffer;
}

RenderGraphResourceHandle RenderGraphResources::GetBufferHandle(const std::string &name) const {
    const auto it = m_resourcesLUT.find(name);
    LOG_ERROR_IF(it == m_resourcesLUT.end(), "Buffer '{}' does not exist.", name);
    assert(m_resources[it->second.index].type == RenderGraphResourceType::Buffer);
    return it->second;
}

RenderGraphResourceHandle RenderGraphResources::DeclareIndirectCommandBuffer(std::string name) {
    return Declare(std::move(name), RenderGraphResourceType::IndirectCommandBuffer);
}

RenderGraphResourceHandle RenderGraphResources::RegisterIndirectCommandBuffer(std::string name, const IndirectCommandBuffer& indirectCB) {
    MTL::IndirectCommandBuffer* nativeICB = indirectCB.GetNative();
    LOG_ERROR_IF(!nativeICB, "Cannot register null indirect command buffer '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareIndirectCommandBuffer(name);
    PreparePhysicalResource(handle, RenderGraphResourceScope::Persistent, 0).indirectCB = nativeICB;

    return handle;
}

RenderGraphResourceHandle RenderGraphResources::RegisterFrameLocalIndirectCommandBuffer(
    std::string name,
    const uint32_t frameSlot,
    const IndirectCommandBuffer& indirectCB)
{
    MTL::IndirectCommandBuffer* nativeICB = indirectCB.GetNative();
    LOG_ERROR_IF(!nativeICB, "Cannot register null frame-local indirect command buffer '{}' into render graph.", name);

    RenderGraphResourceHandle handle = DeclareIndirectCommandBuffer(name);
    PreparePhysicalResource(handle, RenderGraphResourceScope::FrameLocal, frameSlot).indirectCB = nativeICB;

    return handle;
}

MTL::IndirectCommandBuffer* RenderGraphResources::GetIndirectCommandBuffer(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!IsIndirectCommandBufferHandleValid(handle), "Invalid render graph indirect command buffer handle {}.", handle.index);
    return GetCurrentPhysicalResource(handle)->indirectCB;
}

RenderGraphResourceHandle RenderGraphResources::GetIndirectCommandBufferHandle(const std::string& name) const {
    const auto it = m_resourcesLUT.find(name);
    LOG_ERROR_IF(it == m_resourcesLUT.end(), "IndirectCommandBuffer '{}' does not exist.", name);
    assert(m_resources[it->second.index].type == RenderGraphResourceType::IndirectCommandBuffer);
    return it->second;
}

bool RenderGraphResources::IsTextureHandleValid(RenderGraphResourceHandle handle) const {
    const RenderGraphPhysicalResource* physicalResource = GetCurrentPhysicalResource(handle);
    return IsTextureHandleDeclared(handle) && physicalResource && physicalResource->texture;
}

bool RenderGraphResources::IsTextureHandleDeclared(RenderGraphResourceHandle handle) const {
    return IsHandleDeclared(handle, RenderGraphResourceType::Texture);
}

bool RenderGraphResources::IsBufferHandleValid(RenderGraphResourceHandle handle) const {
    const RenderGraphPhysicalResource* physicalResource = GetCurrentPhysicalResource(handle);
    return IsBufferHandleDeclared(handle) && physicalResource && physicalResource->buffer;
}

bool RenderGraphResources::IsBufferHandleDeclared(RenderGraphResourceHandle handle) const {
    return IsHandleDeclared(handle, RenderGraphResourceType::Buffer);
}

bool RenderGraphResources::IsIndirectCommandBufferHandleValid(RenderGraphResourceHandle handle) const {
    const RenderGraphPhysicalResource* physicalResource = GetCurrentPhysicalResource(handle);
    return IsIndirectCommandBufferHandleDeclared(handle) && physicalResource && physicalResource->indirectCB;
}

bool RenderGraphResources::IsIndirectCommandBufferHandleDeclared(RenderGraphResourceHandle handle) const {
    return IsHandleDeclared(handle, RenderGraphResourceType::IndirectCommandBuffer);
}

const std::string& RenderGraphResources::GetName(RenderGraphResourceHandle handle) const {
    LOG_ERROR_IF(!handle.IsValid() || handle.index >= m_resources.size(),
        "Invalid render graph resource handle {}.",
        handle.index);
    return m_resources[handle.index].name;
}
