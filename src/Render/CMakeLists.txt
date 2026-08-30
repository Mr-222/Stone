add_library(Render STATIC
        Renderer.cpp
        Scene.cpp
        Camera.cpp
        ObjectCullingPass.cpp
        OpaqueDirectLightingPass.cpp
        TransparentObjectCullingPass.cpp
        TransparentDirectLightingPass.cpp
        TransparentCompositePass.cpp
)

target_include_directories(Render PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(Render PUBLIC
        Core
)

target_link_libraries(Render PRIVATE
        Utility
        fastgltf::fastgltf
        glm::glm
)
