#include "Render/Renderer.h"

int main() {
#ifndef NDEBUG
    setenv("MTL_DEBUG_LAYER", "1", 0);
    setenv("MTL_SHADER_VALIDATION", "1", 0);
#endif

    Renderer renderer;
    renderer.Run();

    return 0;
}
