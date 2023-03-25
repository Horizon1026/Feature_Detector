aux_source_directory( . AUX_SRC_FEATURE_DETECTOR )

if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()

if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

add_library( lib_feature_detector ${AUX_SRC_FEATURE_DETECTOR} )
target_include_directories( lib_feature_detector PUBLIC . )
target_link_libraries( lib_feature_detector
    lib_slam_utility_datatype
    lib_slam_utility_operate
)
