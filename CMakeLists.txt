cmake_minimum_required( VERSION 3.10 )

project( feature_detector )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -Wall -pthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 20 )

# Add feature point/line detector and feature descriptor.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector/src )
if( NOT TARGET lib_feature_point_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()
if( NOT TARGET lib_feature_line_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/feature_line_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_line_detector )
endif()
if( NOT TARGET lib_feature_descriptor )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/feature_descriptor ${PROJECT_SOURCE_DIR}/build/lib_feature_descriptor )
endif()

# Add slam utility log.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add image painter.
set( IMAGE_PAINTER_PATH ${PROJECT_SOURCE_DIR}/../Image_Painter )
if( NOT TARGET lib_image_painter )
    add_subdirectory( ${IMAGE_PAINTER_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_image_painter )
endif()

# Add visualizor.
set( VISUALIZOR_PATH ${PROJECT_SOURCE_DIR}/../Visualizor2D )
if( NOT TARGET lib_2d_visualizor )
    add_subdirectory( ${VISUALIZOR_PATH}/src ${PROJECT_SOURCE_DIR}/build/lib_2d_visualizor )
endif()

# Add stack backward.
if( NOT TARGET lib_stack_backward )
    add_subdirectory( ${SLAM_UTILITY_PATH}/3rd_libraries/backward ${PROJECT_SOURCE_DIR}/build/lib_stack_backward )
endif()

# Test feature point detector.
add_executable( test_feature_point_detector
    test/test_feature_point_detector.cpp
)
target_link_libraries( test_feature_point_detector
    lib_feature_point_detector
    lib_slam_utility_log
    lib_image_painter
    lib_2d_visualizor
)

# Test feature descriptor.
add_executable( test_feature_descriptor
    test/test_feature_descriptor.cpp
)
target_link_libraries( test_feature_descriptor
    lib_feature_point_detector
    lib_feature_descriptor
    lib_slam_utility_log
    lib_image_painter
    lib_2d_visualizor
)

# Test feature line detector.
add_executable( test_feature_line_detector
    test/test_feature_line_detector.cpp
)
target_link_libraries( test_feature_line_detector
    lib_feature_line_detector
    lib_slam_utility_log
    lib_image_painter
    lib_2d_visualizor
)
