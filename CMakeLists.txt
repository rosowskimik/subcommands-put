cmake_minimum_required(VERSION 3.29 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(commands LANGUAGES C CXX)

include(StandardProjectSettings)
include(Dependencies)

file(GLOB_RECURSE headers CONFIGURE_DEPEDNS
     "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(commands ${sources} ${headers})
set_target_properties(commands PROPERTIES CXX_STANDARD 20 CXX_EXTENSIONS OFF)
target_link_libraries(commands PRIVATE commands_options)
target_include_directories(commands
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(
  commands
  PRIVATE executorch extension_module_static extension_tensor
          optimized_native_cpu_ops_lib xnnpack_backend cxxopts::cxxopts)
