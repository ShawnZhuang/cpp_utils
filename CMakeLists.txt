cmake_minimum_required(VERSION 3.10)
project(project-name)

enable_testing()

add_subdirectory(combine)
add_subdirectory(tests/cpp)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)