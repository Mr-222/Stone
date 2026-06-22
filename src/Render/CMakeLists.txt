add_library(Render STATIC
        Renderer.cpp
        TrianglePass.cpp
        Scene.cpp
        Camera.cpp
        ObjectCullingPass.cpp
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
