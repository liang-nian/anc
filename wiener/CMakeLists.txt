cmake_minimum_required(VERSION 3.20)
project(wiener C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-extra-semi)

add_library(wiener SHARED wiener.c)
target_include_directories(wiener PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wiener PUBLIC m)
