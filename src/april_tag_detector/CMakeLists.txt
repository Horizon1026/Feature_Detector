aux_source_directory( . AUX_SRC_APRIL_TAG_DETECTOR )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()
if( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()

# Add all relative components of slam utility data type.
if( NOT TARGET lib_image )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/image ${PROJECT_SOURCE_DIR}/build/lib_image )
endif()
if( NOT TARGET lib_line_segment )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/line_segment ${PROJECT_SOURCE_DIR}/build/lib_line_segment )
endif()

# Add all relative components of slam utility data struct.
if( NOT TARGET lib_kd_tree )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_struct/kd_tree ${PROJECT_SOURCE_DIR}/build/lib_kd_tree )
endif()

# Add feature point/line detector and feature descriptor.
set( FEATURE_DETECTOR_PATH ${PROJECT_SOURCE_DIR}/../Feature_Detector )
if( NOT TARGET lib_feature_point_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_point_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_point_detector )
endif()
if( NOT TARGET lib_feature_line_detector )
    add_subdirectory( ${FEATURE_DETECTOR_PATH}/src/feature_line_detector ${PROJECT_SOURCE_DIR}/build/lib_feature_line_detector )
endif()

add_library( lib_april_tag_detector ${AUX_SRC_APRIL_TAG_DETECTOR} )
target_include_directories( lib_april_tag_detector PUBLIC . )
target_link_libraries( lib_april_tag_detector
    lib_slam_utility_basic_type
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_tick_tock
    lib_slam_utility_math

    lib_image
    lib_line_segment

    lib_kd_tree

    lib_feature_point_detector
    lib_feature_line_detector
)
