cmake_minimum_required(VERSION 3.19)
project(anomalib_openvino_cpp)

# 设置C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 允许用户指定OpenCV和OpenVINO路径
set(OpenCV_DIR "" CACHE PATH "OpenCV installation directory")
set(OpenVINO_DIR "" CACHE PATH "OpenVINO installation directory")

# 如果未指定路径，尝试在系统中查找
if(NOT OpenCV_DIR AND NOT OpenVINO_DIR)
    message(STATUS "Searching for OpenCV and OpenVINO in system...")
endif()

# RapidJSON包含目录
include_directories(include)

# 查找OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found:")
    message(STATUS "  Version: ${OpenCV_VERSION}")
    message(STATUS "  Include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR or install OpenCV.")
endif()

# 查找OpenVINO
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
if(OpenVINO_FOUND)
    message(STATUS "OpenVINO found:")
    message(STATUS "  Version: ${OpenVINO_VERSION}")
else()
    message(FATAL_ERROR "OpenVINO not found. Please set OpenVINO_DIR or install OpenVINO.")
endif()

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件
set(SOURCES
    patchcore_detector.cpp
    utils.cpp
    opencv_utils.cpp
)

set(HEADERS
    patchcore_detector.h
    utils.h
    opencv_utils.h
    inference.hpp
)

# 创建库
add_library(anomalib_core STATIC ${SOURCES} ${HEADERS})
target_link_libraries(anomalib_core ${OpenCV_LIBS} openvino::runtime)

# 主程序
add_executable(main main.cpp)
target_link_libraries(main anomalib_core ${OpenCV_LIBS} openvino::runtime)

# 安装规则
install(TARGETS main DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)
