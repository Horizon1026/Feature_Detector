aux_source_directory( . AUX_SRC_FEATURE_POINT_DETECTOR )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

add_library( lib_feature_line_detector ${AUX_SRC_FEATURE_POINT_DETECTOR} )
target_include_directories( lib_feature_line_detector PUBLIC . )
target_link_libraries( lib_feature_line_detector
    lib_slam_utility_datatype
    lib_slam_utility_operate
)
