add_library(Render STATIC
        Renderer.cpp
        Scene.cpp
)

target_include_directories(Render PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(Render PRIVATE
        Core
        Utility
        fastgltf::fastgltf
        glm::glm
)
