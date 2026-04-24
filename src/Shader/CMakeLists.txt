find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

function(stone_resolve_metal_tool out_var tool_name)
    if(DEFINED ${out_var} AND EXISTS "${${out_var}}")
        set(${out_var} "${${out_var}}" PARENT_SCOPE)
        return()
    endif()

    execute_process(
            COMMAND "${XCRUN_EXECUTABLE}" --toolchain Metal --find "${tool_name}"
            RESULT_VARIABLE _STONE_TOOL_RESULT
            OUTPUT_VARIABLE _STONE_TOOL_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
    )

    if(_STONE_TOOL_RESULT EQUAL 0 AND EXISTS "${_STONE_TOOL_PATH}")
        set(${out_var} "${_STONE_TOOL_PATH}" CACHE FILEPATH "Path to the Apple ${tool_name} tool" FORCE)
        set(${out_var} "${_STONE_TOOL_PATH}" PARENT_SCOPE)
    else()
        unset(${out_var} CACHE)
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

message(STATUS "Searching for Metal toolchain via xcrun...")
stone_resolve_metal_tool(METAL_EXECUTABLE metal)
stone_resolve_metal_tool(METALLIB_EXECUTABLE metallib)

if(NOT METALLIB_EXECUTABLE AND METAL_EXECUTABLE)
    get_filename_component(_STONE_METAL_BIN_DIR "${METAL_EXECUTABLE}" DIRECTORY)
    set(_STONE_METALLIB_FALLBACK "${_STONE_METAL_BIN_DIR}/metallib")
    if(EXISTS "${_STONE_METALLIB_FALLBACK}")
        set(METALLIB_EXECUTABLE "${_STONE_METALLIB_FALLBACK}")
        set(METALLIB_EXECUTABLE "${_STONE_METALLIB_FALLBACK}" CACHE FILEPATH "Path to the Apple metallib tool" FORCE)
    endif()
endif()

if(NOT EXISTS "${METAL_EXECUTABLE}" OR NOT EXISTS "${METALLIB_EXECUTABLE}")
    message(FATAL_ERROR "Metal shader tools were not found via xcrun. Install them with: xcodebuild -downloadComponent MetalToolchain")
endif()

message(STATUS "Found Metal Compiler: ${METAL_EXECUTABLE}")
message(STATUS "Found Metal Library Tool: ${METALLIB_EXECUTABLE}")

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
