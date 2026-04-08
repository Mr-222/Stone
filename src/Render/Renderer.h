#pragma once

#include <memory>

class MetalContext;
class Window;

class Renderer {
public:
    Renderer();
    ~Renderer();

    void Init();
    void Run();

private:
    std::unique_ptr<MetalContext> m_metalContext;
    std::unique_ptr<Window> m_window;
};
