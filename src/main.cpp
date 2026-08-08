#include "Render/Renderer.h"

int main() {
#ifndef NDEBUG
    setenv("MTL_DEBUG_LAYER", "1", 0);
    // The current Shader Validation layer misclassifies 2D textures reached
    // through an inherited argument buffer in a GPU-generated ICB. Direct
    // draws and compute access validate the same resource and binding as 2D.
    // Keep API validation and validation on the culling pipeline enabled.
    setenv("MTL_SHADER_VALIDATION", "0", 0);
#endif

    Renderer renderer;
    renderer.Run();

    return 0;
}
