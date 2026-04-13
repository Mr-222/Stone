add_library(Utility STATIC
    Logger.cpp
)

target_include_directories(Utility PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(Utility PUBLIC spdlog::spdlog)