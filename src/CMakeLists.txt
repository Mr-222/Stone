add_subdirectory(Utility)

set(SOURCES
        main.mm
        MetalInit.cpp
)

find_package(glfw3 REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Utility
        glfw
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
        "-framework AppKit"
        "-framework Cocoa"
)