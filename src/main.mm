#include <Cocoa/Cocoa.h>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

#include <iostream>

#include <objc/message.h>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    int width = 800;
    int height = 600;
    GLFWwindow* window = glfwCreateWindow(width, height, "My Metal Renderer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to find a compatible Metal device!" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    MTL::CommandQueue* commandQueue = device->newCommandQueue();

    CA::MetalLayer* metalLayer = CA::MetalLayer::layer();
    metalLayer->setDevice(device);
    metalLayer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);

    // Get the native Cocoa window from GLFW
    NSWindow* cocoaWindow = id(glfwGetCocoaWindow(window));

    NSView* contentView = [cocoaWindow contentView];

    [contentView setWantsLayer:YES];

    [contentView setLayer:(__bridge CALayer*)metalLayer];

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        CA::MetalDrawable* drawable = metalLayer->nextDrawable();
        if (drawable) {

            MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

            MTL::RenderPassDescriptor* passDescriptor = MTL::RenderPassDescriptor::renderPassDescriptor();
            MTL::RenderPassColorAttachmentDescriptor* colorAttachment = passDescriptor->colorAttachments()->object(0);

            colorAttachment->setTexture(drawable->texture());
            colorAttachment->setLoadAction(MTL::LoadActionClear);
            colorAttachment->setClearColor(MTL::ClearColor::Make(0.1, 0.2, 0.3, 1.0));
            colorAttachment->setStoreAction(MTL::StoreActionStore);

            MTL::RenderCommandEncoder* commandEncoder = commandBuffer->renderCommandEncoder(passDescriptor);
            commandEncoder->endEncoding();

            commandBuffer->presentDrawable(drawable);
            commandBuffer->commit();
        }

        pool->release();
    }

    commandQueue->release();
    device->release();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}