add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../combine ${CMAKE_CURRENT_BINARY_DIR}/combine)

enable_testing()

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(shift_stack_test shift_stack_test.cc)
target_link_libraries(shift_stack_test GTest::GTest GTest::Main)

add_executable(dp_test dp_test.cc)
target_link_libraries(dp_test GTest::GTest GTest::Main)

add_test(NAME shift_stack_test COMMAND shift_stack_test)
add_test(NAME dp_test COMMAND dp_test)