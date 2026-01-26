if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS "Setting build type to 'RelWithDebInfo' as none aws specified.")
  set(CMAKE_BUILD_TYPE
      RelWithDebInfo
      CACHE STRING "Choose the type of build" FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# Auto-create a .gitignore in the specified "build" directory.
if(NOT PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  file(
    GENERATE
    OUTPUT .gitignore
    CONTENT "*")
endif()

# Interface target for compile options
add_library(commands_options INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  target_compile_options(commands_options INTERFACE -fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(commands_options INTERFACE -fdiagnostics-color=always)
endif()

# Generate compile_commands.json for clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
