find_package(glfw3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(Utility)
add_subdirectory(Core)
add_subdirectory(Render)

add_executable(${PROJECT_NAME}
        main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Render
)
