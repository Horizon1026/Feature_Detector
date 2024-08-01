aux_source_directory( . AUX_SRC_FEATURE_POINT_DETECTOR )
aux_source_directory( feature_point AUX_SRC_FEATURE_POINT )

# Add all components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility/src )
if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( ${SLAM_UTILITY_PATH}/datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()

add_library( lib_feature_point_detector
    ${AUX_SRC_FEATURE_POINT_DETECTOR}
    ${AUX_SRC_FEATURE_POINT}
)
target_include_directories( lib_feature_point_detector
    PUBLIC .
           feature_point
)
target_link_libraries( lib_feature_point_detector
    lib_slam_utility_datatype
    lib_slam_utility_operate
    lib_slam_utility_log
    lib_slam_utility_tick_tock
)
