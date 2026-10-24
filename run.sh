cd build/
./test_feature_point_detector ../examples/calibration_board_1.png
./test_feature_descriptor ../examples/calibration_board_1.png
./test_feature_line_detector ../examples/calibration_board_1.png
./test_nn_feature_point_detector ../examples/calibration_board_1.png

./test_april_tag_detector ../examples/calibration_board_1.png tag_36h11 2
./test_april_tag_detector ../examples/calibration_board_2.png aruco_mip_36h12 1
./test_april_tag_detector ../examples/calibration_board_3.png aruco_4x4_100 1
./test_april_tag_detector ../examples/calibration_board_4.png aruco_4x4_100 1
./test_april_tag_detector ../examples/calibration_board_5.png aruco_mip_36h12 1
./test_april_tag_detector ../examples/calibration_board_6.png tag_36h11 2
./test_april_tag_detector ../examples/calibration_board_7.png tag_36h11 2
cd ..
