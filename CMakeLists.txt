cmake_minimum_required( VERSION 3.10 )

project( feature_detector )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )

set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_slam_utility )
endif()

include_directories( ${PROJECT_SOURCE_DIR}/src )
aux_source_directory( ${PROJECT_SOURCE_DIR}/src FEATURE_DETECTOR )

add_executable( test_feature_detector
    ${FEATURE_DETECTOR}
    test/test_feature_detector.cpp
)

target_link_libraries( test_feature_detector
    ${OpenCV_LIBS}
    lib_slam_utility
)
