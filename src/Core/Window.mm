#include "Core/Window.h"

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <Cocoa/Cocoa.h>
#include <Metal/Metal.hpp>

#include "Utility/Logger.h"

Window::Window(int width, int height): m_width(width), m_height(height) {
    bool success = glfwInit();
    LOG_ERROR_IF(!success, "Failed to initialize GLFW3!");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_window = glfwCreateWindow(m_width, m_height, "Stone", nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        LOG_ERROR("Failed to create GLFW window!");
    }

    m_metalLayer = NS::RetainPtr(CA::MetalLayer::layer());

    NSWindow* cocoaWindow = (NSWindow*)glfwGetCocoaWindow(m_window);
    NSView* contentView = [cocoaWindow contentView];
    [contentView setWantsLayer:YES];
    [contentView setLayer:(__bridge CALayer*)m_metalLayer.get()];
}

Window::~Window() {
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

bool Window::ShouldClose() {
    return glfwWindowShouldClose(m_window);
}

void Window::PollEvents() {
    glfwPollEvents();
}

GLFWwindow* Window::GetGLFWWindow() {
    return m_window;
}

CA::MetalLayer* Window::GetCAMetalLayer() {
    return m_metalLayer.get();
}
