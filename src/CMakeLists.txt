include(FetchContent)

FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.1.0
        FIND_PACKAGE_ARGS 12.0 CONFIG
)

FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        FIND_PACKAGE_ARGS
)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.17.0
        FIND_PACKAGE_ARGS CONFIG
)

FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.3
        FIND_PACKAGE_ARGS CONFIG
)

FetchContent_MakeAvailable(fmt glfw spdlog glm)

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
