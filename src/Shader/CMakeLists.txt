find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

# Find Metal Tools using xcrun (Cached for speed)
if(NOT METAL_EXECUTABLE OR NOT METALLIB_EXECUTABLE)
    message(STATUS "Searching for Metal toolchain via xcrun...")

    # Find the metal compiler
    execute_process(
            COMMAND "${XCRUN_EXECUTABLE}" --toolchain Metal --find metal
            RESULT_VARIABLE _METAL_RESULT
            OUTPUT_VARIABLE _METAL_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    # Find the metallib linker
    execute_process(
            COMMAND "${XCRUN_EXECUTABLE}" --toolchain Metal --find metallib
            RESULT_VARIABLE _METALLIB_RESULT
            OUTPUT_VARIABLE _METALLIB_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    # Cache the variables so this block doesn't run again unless cleared
    if(_METAL_RESULT EQUAL 0 AND _METALLIB_RESULT EQUAL 0)
        set(METAL_EXECUTABLE "${_METAL_PATH}" CACHE FILEPATH "Path to the Apple Metal compiler")
        set(METALLIB_EXECUTABLE "${_METALLIB_PATH}" CACHE FILEPATH "Path to the Apple Metal linker")
        message(STATUS "Found Metal Compiler: ${METAL_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Metal shader tools were not found via xcrun. Install them with: xcodebuild -downloadComponent MetalToolchain")
    endif()
endif()

# Shader Compilation Targets
file(GLOB METAL_SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.metal")

set(STONE_SHADER_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Shader")
set(STONE_SHADER_MODULE_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ModuleCache")
set(STONE_METALLIB_OUTPUTS)

# Create directories during CMake configuration
file(MAKE_DIRECTORY "${STONE_SHADER_OUTPUT_DIR}" "${STONE_SHADER_MODULE_CACHE_DIR}")

foreach(SHADER_SOURCE IN LISTS METAL_SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER_SOURCE}" NAME_WE)

    set(AIR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.air")
    set(METALLIB_OUTPUT "${STONE_SHADER_OUTPUT_DIR}/${SHADER_NAME}.metallib")

    add_custom_command(
            OUTPUT "${METALLIB_OUTPUT}" "${AIR_OUTPUT}"
            COMMAND "${METAL_EXECUTABLE}" -fmodules-cache-path="${STONE_SHADER_MODULE_CACHE_DIR}" -c "${SHADER_SOURCE}" -o "${AIR_OUTPUT}"
            COMMAND "${METALLIB_EXECUTABLE}" "${AIR_OUTPUT}" -o "${METALLIB_OUTPUT}"
            DEPENDS "${SHADER_SOURCE}"
            COMMENT "Compiling Metal shader ${SHADER_NAME}.metal"
            VERBATIM
    )

    list(APPEND STONE_METALLIB_OUTPUTS "${METALLIB_OUTPUT}")
endforeach()

add_custom_target(StoneShaders ALL
        DEPENDS ${STONE_METALLIB_OUTPUTS}
)