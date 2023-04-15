cmake_minimum_required( VERSION 3.10 )

project( feature_detector )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

find_package( OpenCV REQUIRED )

# Add feature detector.
if ( NOT TARGET lib_feature_detector )
add_subdirectory( src ${PROJECT_SOURCE_DIR}/build/lib_feature_detector )
endif()

# Add slam utility log.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

add_executable( test_feature_detector
    test/test_feature_detector.cpp
)

target_link_libraries( test_feature_detector
    ${OpenCV_LIBS}
    lib_feature_detector
    lib_slam_utility_log
)
