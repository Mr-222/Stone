#include "Render/Renderer.h"

int main() {
#ifndef NDEBUG
    // Enable Metal API and shader validation before the device is created.
    setenv("MTL_DEBUG_LAYER", "1", 0);
    setenv("MTL_SHADER_VALIDATION", "1", 0);
#endif

    Renderer renderer;
    renderer.Init();
    renderer.Run();

    return 0;
}
