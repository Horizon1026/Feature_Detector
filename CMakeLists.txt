# cmake工程最低版本为3.1
cmake_minimum_required( VERSION 3.10 )

# cmake工程名
project( featureDetector )

# 设置编译类型为发布
set(CMAKE_BUILD_TYPE "Release")

# 添加c++ 11标准的支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# 寻找OpenCV库并添加其路径
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 添加路径
include_directories( ${PROJECT_SOURCE_DIR}/include )

# 添加可执行文件
add_executable( featureDetector
    src/main.cpp
    include/FASTFeatureDetector.cpp
)

# 链接库到可执行文件
target_link_libraries( featureDetector
    ${OpenCV_LIBS}
)
