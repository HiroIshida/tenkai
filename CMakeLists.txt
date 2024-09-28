FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
add_library(cgunroll ${SRC_FILES})
add_executable(main main.cpp)
include_directories(src)
target_link_libraries(main cgunroll ${CMAKE_DL_LIBS})

# bench
add_executable(bench test/bench.cpp)
target_link_libraries(bench cgunroll ${CMAKE_DL_LIBS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
