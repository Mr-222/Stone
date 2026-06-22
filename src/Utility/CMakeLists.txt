add_library(Utility STATIC
    Logger.cpp
    ShaderLibrary.cpp
)

target_include_directories(Utility PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/External/metal-cpp
)

target_link_libraries(Utility PUBLIC
    spdlog::spdlog
    "-framework Metal"
    "-framework Foundation"
)
