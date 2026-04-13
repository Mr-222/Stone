find_package(fmt 12.0 CONFIG REQUIRED)
find_package(glfw3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(Utility)
add_subdirectory(Core)
add_subdirectory(Render)
add_subdirectory(Shader)

add_executable(${PROJECT_NAME}
        main.cpp
)

add_dependencies(${PROJECT_NAME} StoneShaders)

target_compile_definitions(Render PRIVATE
        STONE_SHADER_DIR="$<TARGET_FILE_DIR:${PROJECT_NAME}>/Shader"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Render
)
