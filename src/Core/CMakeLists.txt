add_library(Core STATIC
        MetalInit.cpp
        MetalContext.mm
        Window.mm
        CommandBufferPool.cpp
        CommandBuffer.cpp
        CommandBuffer.h
)

target_include_directories(Core PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/External/metal-cpp
)

target_link_libraries(Core PRIVATE
        Utility
        glfw
)

target_link_libraries(Core PUBLIC
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework AppKit"
        "-framework Cocoa"
)