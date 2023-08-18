cmake_minimum_required( VERSION 3.10 )

project( feature_detector )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -Wall -lpthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 17 )

# Add feature point/line detector and feature descriptor.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector/src )
if ( NOT TARGET lib_feature_point_detector )
add_subdirectory( ${FEATURE_DETECTOR_PATH}/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()
if ( NOT TARGET lib_feature_line_detector )
add_subdirectory( ${FEATURE_DETECTOR_PATH}/feature_line_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_line_detector )
endif()
if ( NOT TARGET lib_feature_descriptor )
add_subdirectory( ${FEATURE_DETECTOR_PATH}/feature_descriptor ${PROJECT_SOURCE_DIR}/build/lib_feature_descriptor )
endif()

# Add slam utility log.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add visualizor.
set( VISUALIZOR_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if ( NOT TARGET lib_slam_visualize )
    add_subdirectory( ${VISUALIZOR_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_slam_visualize )
endif()

# Test feature point detector.
add_executable( test_feature_point_detector
    test/test_feature_point_detector.cpp
)
target_link_libraries( test_feature_point_detector
    lib_feature_point_detector
    lib_slam_utility_log
    lib_slam_visualize
)

# Test feature descriptor.
add_executable( test_feature_descriptor
    test/test_feature_descriptor.cpp
)
target_link_libraries( test_feature_descriptor
    lib_feature_point_detector
    lib_feature_descriptor
    lib_slam_utility_log
    lib_slam_visualize
)
