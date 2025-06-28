cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME tensor-contractions)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -march=native")

add_compile_options(-Wno-sign-compare)

find_package(TBB REQUIRED)

link_libraries(${CMAKE_THREAD_LIBS_INIT})
link_libraries(TBB::tbb)

include_directories(include)

add_executable(playground "${PROJECT_SOURCE_DIR}/src/playground.cpp")
target_compile_options(playground PRIVATE -Wall -Wextra -pedantic)

add_executable(ravel_index "${PROJECT_SOURCE_DIR}/src/ravel_index.cpp")
target_compile_options(ravel_index PRIVATE -Wall -Wextra -pedantic)

option(BUILD_TESTS "Build the tests" OFF)

if(BUILD_TESTS)

    enable_testing()
    find_package(GTest REQUIRED)
    include(GoogleTest)

    set(PROJECT_TEST_NAME test_${PROJECT_NAME})

    file(GLOB_RECURSE TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
    add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})
    target_include_directories(${PROJECT_TEST_NAME} PRIVATE tests/include)
    target_link_libraries(${PROJECT_TEST_NAME} GTest::gtest_main)

    gtest_discover_tests(${PROJECT_TEST_NAME})

endif()
