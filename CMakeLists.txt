FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(tenkai ${SRC_FILES})
add_executable(main main.cpp)
include_directories(src)
target_link_libraries(main tenkai ${CMAKE_DL_LIBS})

# bench
add_executable(bench test/bench.cpp)
target_link_libraries(bench tenkai ${CMAKE_DL_LIBS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
