#pragma once

#include <QuartzCore/QuartzCore.hpp>

inline constexpr uint MAX_FRAMES_IN_FLIGHT = 3;

struct GLFWwindow;

class Window {
public:
    explicit Window(int width, int height);
    ~Window();

    bool ShouldClose();
    void PollEvents();
    GLFWwindow* GetGLFWWindow();
    CA::MetalLayer* GetCAMetalLayer();

private:
    int m_width, m_height;
    GLFWwindow* m_window;
    NS::SharedPtr<CA::MetalLayer> m_metalLayer;
};
