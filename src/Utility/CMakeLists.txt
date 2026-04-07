add_library(Utility STATIC
    Logger.cpp
)

target_include_directories(Utility PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)