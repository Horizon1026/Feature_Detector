aux_source_directory( . AUX_SRC_FEATURE_POINT_DETECTOR )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()

# Add all relative components of slam utility data type.
if ( NOT TARGET lib_image )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/image ${PROJECT_SOURCE_DIR}/build/lib_image )
endif()

# Add circular buffer of slam utility data struct.
if ( NOT TARGET lib_circular_buffer )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_struct/circular_buffer ${PROJECT_SOURCE_DIR}/build/lib_circular_buffer )
endif()

add_library( lib_feature_line_detector ${AUX_SRC_FEATURE_POINT_DETECTOR} )
target_include_directories( lib_feature_line_detector PUBLIC . )
target_link_libraries( lib_feature_line_detector
    lib_slam_utility_basic_type
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_tick_tock
    lib_slam_utility_math

    lib_image
    lib_circular_buffer
)
