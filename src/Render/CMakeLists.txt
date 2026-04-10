add_library(Render STATIC
        Renderer.cpp
)

target_include_directories(Render PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(Render PRIVATE
        Core
)

target_link_libraries(Render PUBLIC
    glm::glm
)