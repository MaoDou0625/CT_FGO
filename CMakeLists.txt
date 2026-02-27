cmake_minimum_required(VERSION 3.10)
project(ob_gins)
enable_testing()
# 针对 Windows MSVC 编译器开启 /bigobj 支持
if(MSVC)
    add_compile_options(/bigobj)
endif()
# Use C++20 to support designated initializers on MSVC
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    # Avoid passing GCC/Clang flags like -O3/-Wall to MSVC
    # Force UTF-8 source file encoding to eliminate comment garbling on MSVC
    add_compile_options(/permissive- /utf-8)
else ()
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        add_compile_options(-Wall)
    else ()
        add_compile_options(-Wall -O3)
    endif ()
endif ()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR})

# Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# yaml-cpp
find_package(yaml-cpp REQUIRED)

# abseil
set(ABSL_PROPAGATE_CXX_STD true)
add_subdirectory(src/thirdparty/abseil-cpp abseil-cpp)

# Ceres
find_package(Ceres REQUIRED)

# glog & gflags
find_package(glog REQUIRED)
find_package(gflags REQUIRED)

# GTest
find_package(GTest REQUIRED)



# Continuous Time OB_GINS (Batch Optimization)
add_executable(ob_gins_ct 
    src/ob_gins_ct.cc
    src/core/imu_processor.cc
    src/core/data_buffer.cc
    src/common/debug.cc
    src/common/interpolation.cc
    src/fileio/fileloader.cc
    src/fileio/filesaver.cc
    src/factors/pose_manifold.cc
)
target_link_libraries(ob_gins_ct 
    Ceres::ceres 
    yaml-cpp::yaml-cpp 
    Sophus::Sophus 
    glog::glog
    gflags
    absl::strings
    absl::str_format
    absl::time
)
