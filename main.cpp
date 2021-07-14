#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2\objdetect\objdetect_c.h> 

#include <opencv2\imgproc\types_c.h>
#include "utils.h"

using namespace cv;
using namespace std;

const bool AUTO = false;

//相机内参矩阵，左相机
const Mat CAMERA_MARTRIX_LC = (Mat_<float>(3, 3) << 889.919440818368, 0, 644.372835716297, 0, 869.209302050750, 327.156627141872, 0, 0, 1.000);
//相机畸变矩阵，左相机
const Mat DIST_COEFFS_LC = (Mat_<float>(1, 5) << -0.400427519434363, 0.184052702091282, 0, 0, 0);
//相机旋转矩阵，左相机
const Mat ROTATION_LC = (Mat_<float>(3, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
//相机平移矩阵，左相机
const Mat TRANSLATION_LC = (Mat_<float>(3, 1) << 0.0, 0.0, 0.0);

//相机内参矩阵，右相机
const Mat CAMERA_MARTRIX_RC = (Mat_<float>(3, 3) << 884.450254898406, 0, 643.469632880406, 0, 864.450430947805, 325.481385900049, 0, 0, 1.000);
//相机畸变矩阵，右相机
const Mat DIST_COEFFS_RC = (Mat_<float>(1, 5) << -0.404293670845458, 0.223765597230932, 0, 0, 0);
//相机旋转矩阵，右相机
//const Mat ROTATION_RC = (Mat_<float>(3, 3) << 0.999621995981635, 0.00449026121635850, 0.0271238401392682, -0.00438565285588986, 0.999982719446515, -0.00391495304967741, -0.0271409505861408, 0.00379451743472343, 0.999624414687195);
const Mat ROTATION_RC = (Mat_<float>(3, 3) << 0.9997, - 0.0006  ,  0.0234,0.0006 ,   1.0000 ,- 0.0006,- 0.0234  ,  0.0006  ,  0.9997);
//相机平移矩阵，右相机
//const Mat TRANSLATION_RC = (Mat_<float>(3, 1) << 31.6178586403377, 1.14206108048751, 2.24521331246957);
const Mat TRANSLATION_RC = (Mat_<float>(3, 1) << 33.0271,0.6603,3.8414);


//const Mat RE = (Mat_<float>(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);

int main(int argc, char *argv[]) {

	vector<Point2f> real_marker;
	real_marker.push_back(Point2f(0, 0));
	real_marker.push_back(Point2f(MARKER_LENGTH, 0));
	real_marker.push_back(Point2f(MARKER_LENGTH, MARKER_LENGTH));
	real_marker.push_back(Point2f(0, MARKER_LENGTH));

	vector<Point3f> real_world_marker;
	real_world_marker.clear();
	real_world_marker.push_back(Point3f(0, 0, 0));
	real_world_marker.push_back(Point3f(MARKER_LENGTH, 0, 0));
	real_world_marker.push_back(Point3f(MARKER_LENGTH, MARKER_LENGTH, 0));
	real_world_marker.push_back(Point3f(0, MARKER_LENGTH, 0));

	
	
	//string file_path_lc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_16_38_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_16_41_Pro.jpg"; 
	//string file_path_lc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_18_01_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_18_06_Pro.jpg";
	//string file_path_lc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_17_38_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_17_42_Pro.jpg";
	//string file_path_lc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_08_53_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_08_57_Pro.jpg"; 
	//string file_path_lc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_01_17_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Desktop\\箱子正放\\箱子正放左右图像\\WIN_20190501_16_01_20_Pro.jpg";
	//string file_path_lc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190515_07_40_33_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190515_07_40_39_Pro.jpg";
	//string file_path_lc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190514_21_44_41_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190514_21_44_43_Pro.jpg";
	//string file_path_lc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190515_08_41_28_Pro.jpg";
	//string file_path_rc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190515_08_41_31_Pro.jpg";

	string file_path_lc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190603_18_51_28_Pro.jpg";
	string file_path_rc = "C:\\Users\\武士老师\\Pictures\\Camera Roll\\WIN_20190603_18_51_31_Pro.jpg";
	//VideoCapture cap_lc, cap_rc;
	//string file_path_lc = "0";
	//string file_path_rc = "1";


	if (argc > 2) {
		file_path_lc = argv[argc - 2];
		file_path_rc = argv[argc - 1];
	}

	loadImg cap(file_path_lc, file_path_rc);
	unsigned long int img_index = 0;
	while (1) {
		img_index++;
		Mat img_lc_origin, img_rc_origin;
		cap.load(img_lc_origin, img_rc_origin);
		if (img_lc_origin.empty()) break;
		if (img_rc_origin.empty()) break;

		Mat img_lc, img_rc;
		bool is_undistort = false;			//true 则开启去畸变
		if (is_undistort) {
			undistort(img_lc_origin, img_lc, CAMERA_MARTRIX_LC, DIST_COEFFS_LC);
			undistort(img_rc_origin, img_rc, CAMERA_MARTRIX_RC, DIST_COEFFS_RC);
		}
		else
		{
			img_lc = img_lc_origin;
			img_rc = img_rc_origin;
		}

		//
		//imshow("testl", img_rc);
		//cout << img_lc.size();
		//waitKey(0);
		//获取标记
		vector<Marker> good_markers_lc, good_markers_rc;
		bool is_success = getGoodMarker(img_lc, real_marker, good_markers_lc, "LC");
		if (!is_success) {
			cout << "[" << to_string(img_index) << "]左相机不能检测到AR标记！" << endl;
			continue;
		}
		is_success = getGoodMarker(img_rc, real_marker, good_markers_rc, "RC");
		if (!is_success) {
			cout << "[" << to_string(img_index) << "]右相机不能检测到AR标记！" << endl;
			continue;
		}
		if (good_markers_lc.size() != good_markers_rc.size()) continue;

		Mat R, T;
		if (AUTO) {
			//计算相机相对位姿，用ORB特征匹配计算
			vector<KeyPoint> kp_lc, kp_rc;
			vector<DMatch> matches;
			matchPoints(img_lc, img_rc, kp_lc, kp_rc, matches, 1.5, 10.0);
			getC2CRT(kp_lc, kp_rc, matches, CAMERA_MARTRIX_LC, CAMERA_MARTRIX_RC, R, T);
		}
		else
		{
			ROTATION_RC.copyTo(R);
			TRANSLATION_RC.copyTo(T);
		}
		//invert(R, R);
		//T = -R * T;
		cout << "右相机相对左相机平移：" << T << endl;
		cout << "右相机相对左相机旋转：" << R << endl;

		for (int i = 0; i < good_markers_lc.size(); i++) {
			vector <Point2f> ps_lc, ps_rc;
			is_undistort = false;
			if (is_undistort) {
				undistort(good_markers_lc[i].points, ps_lc, CAMERA_MARTRIX_LC, DIST_COEFFS_LC);
				undistort(good_markers_rc[i].points, ps_rc, CAMERA_MARTRIX_RC, DIST_COEFFS_RC);
			}
			else
			{
				ps_lc = good_markers_lc[i].points;
				ps_rc = good_markers_rc[i].points;
			}
			//cout << ps_lc[0] << good_markers_lc[i].points[0];

			Point3f world;
			Point2f p_lc = ps_lc[0];
			Point2f p_rc = ps_rc[0];

			//由相机坐标系的点计算空间坐标系坐标
			uv2xyz(p_lc, p_rc, world,
				ROTATION_LC,
				TRANSLATION_LC,
				CAMERA_MARTRIX_LC,
				R,
				T,
				CAMERA_MARTRIX_RC);
			//Mat RW, Q;


		
				Point3f v0 = world;
				cout << "v0：" << v0 << endl;
				//swap(v0.y, v0.z);
				//v0.z = -v0.z;
				//v0 = v0 - BIAS;
				Point3f world_temp;
				Point2f p_lc_temp = ps_lc[1];
				Point2f p_rc_temp = ps_rc[1];
				//由相机坐标系的点计算空间坐标系坐标
				uv2xyz(p_lc_temp, p_rc_temp, world_temp,
					ROTATION_LC,
					TRANSLATION_LC,
					CAMERA_MARTRIX_LC,
					R,
					T,
					CAMERA_MARTRIX_RC);
				Point3f v1 = world_temp;
				//swap(v1.y, v1.z);
				//v1.z = -v1.z;
				//v1 = v1 - BIAS;
				cout << "v1：" << v1 << endl;
				p_lc_temp = ps_lc[2];
				p_rc_temp = ps_rc[2];
				//由相机坐标系的点计算空间坐标系坐标
				uv2xyz(p_lc_temp, p_rc_temp, world_temp,
					ROTATION_LC,
					TRANSLATION_LC,
					CAMERA_MARTRIX_LC,
					R,
					T,
					CAMERA_MARTRIX_RC);
				Point3f v2 = world_temp;
				cout << "v2：" << v2 << endl;
				//swap(v2.y, v2.z);
				//v2.z = -v2.z;
				//v2 = v2 - BIAS;
				p_lc_temp = ps_lc[3];
				p_rc_temp = ps_rc[3];
				//由相机坐标系的点计算空间坐标系坐标
				uv2xyz(p_lc_temp, p_rc_temp, world_temp,
					ROTATION_LC,
					TRANSLATION_LC,
					CAMERA_MARTRIX_LC,
					R,
					T,
					CAMERA_MARTRIX_RC);
				Point3f v3 = world_temp;
				cout << "v3：" << v3 << endl;
				//swap(v3.y, v3.z);
				//v3.z = -v3.z;
				//v3 = v3 - BIAS;



			//Mat R1_;
			//eulerAnglesToRotationMatrix(angle, R1_);
			//Mat R2_ = (Mat_<float>(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
			//Mat Q = R2_ * R1_; 
				float MARKER_LENGT = 154;
				vector<Point3f> real_worldmarker;
				//real_worldmarker.push_back(Point3f(0, 0, 0));
				//real_worldmarker.push_back(Point3f(MARKER_LENGT, 0, 0));
				//real_worldmarker.push_back(Point3f(MARKER_LENGT, MARKER_LENGT, 0));
				//real_worldmarker.push_back(Point3f(0, MARKER_LENGT, 0));
				real_worldmarker.push_back(Point3f(0, 0, 0));
				real_worldmarker.push_back(Point3f(154, 0, 0));
				real_worldmarker.push_back(Point3f(154, 154, 0));
				real_worldmarker.push_back(Point3f(0,154, 0));
				vector<Point3f> camera_world_marker;

				//camera_world_marker.push_back(Point3f(121.009, 132.026, 344.248));
				//camera_world_marker.push_back(Point3f(138.729, 132.299, 326.305));
				//camera_world_marker.push_back(Point3f(66.3089, 132.311, 331.145));
				//camera_world_marker.push_back(Point3f(55.4888, 132.157, 343.398));
				camera_world_marker.push_back(v0);
				camera_world_marker.push_back(v1);
				camera_world_marker.push_back(v2);
				camera_world_marker.push_back(v3);
				
				
				
				Mat R_est, t_est;
				FindTransformMat(real_worldmarker, camera_world_marker, R_est, t_est);

				
				cout << "R_est：" << R_est << endl;
				
				cout << "t_est：" << t_est << endl;
				Mat R2_ = (Mat_<float>(3, 3) << -0.05824374, -0.010102862, 0.99825126,
				0.99769229, 0.034365308, 0.058558922,
				-0.034896825, 0.99935824, 0.0080779828);
				Mat Q = R2_ * R_est;



				Vec3f R_euler = rotationMatrixToEulerAngles(Q);	//将旋转矩阵转换成欧拉角
				cout << "欧拉角:" << R_euler << endl;
				
				Mat T2 = (Mat_<float>(3, 1) << 0,
				0,
				0);
				Mat worldm = Mat(v0);
				worldm = R2_*worldm + T2;
				worldm.at<float>(0, 0) = worldm.at<float>(0, 0);
				worldm.at<float>(1, 0) = worldm.at<float>(1, 0);
				worldm.at<float>(2, 0) = -worldm.at<float>(2, 0);
				cout << "目标位置：" << worldm << endl;


#if defined _DEBUG
				Point2f origin_p_lc = p_lc;
				Point2f recover_pl_lc;
				Mat K1;
				invert(CAMERA_MARTRIX_LC, K1);
				Mat R2;
				invert(R, R2);
				Mat pointr2 = (Mat_<float>(3, 1) << p_lc.x, p_lc.y, 1);
				//Mat pointl2 = CAMERA_MARTRIX_LC * R2 * (K2 * pointr2 - T);
				Mat wp = (Mat_<float>(3, 1) << world.x, world.y, world.z);
				Mat pointl2 = CAMERA_MARTRIX_RC * (R * wp + T);
				recover_pl_lc.x = pointl2.at<float>(0, 0) / pointl2.at<float>(2, 0);
				recover_pl_lc.y = pointl2.at<float>(1, 0) / pointl2.at<float>(2, 0);
				//cout << recover_pl_lc;
				//circle(img_lc, p_rc, 4, Scalar(0, 255, 0), 2);
				//circle(img_lc, recover_pl_lc, 8, Scalar(0, 0, 255), 4);
				//line(img_lc, Point(0, int(CAMERA_MARTRIX_LC.at<float>(1, 2))), Point(img_lc.cols, int(CAMERA_MARTRIX_LC.at<float>(1, 2))), Scalar(255, 0, 0), 2);
				//line(img_lc, Point(int(CAMERA_MARTRIX_LC.at<float>(0, 2)), 0), Point(int(CAMERA_MARTRIX_LC.at<float>(0, 2)), img_lc.rows), Scalar(255, 0, 0), 2);
				//putText(img_lc, "X", Point(img_lc.cols - 20, int(CAMERA_MARTRIX_LC.at<float>(1, 2)) - 5), 1, 2, Scalar(0, 255, 255), 2);
				//putText(img_lc, "Z", Point(int(CAMERA_MARTRIX_LC.at<float>(0, 2)) + 5, 25), 1, 2, Scalar(0, 255, 255), 2);
				//putText(img_lc, "Y", Point(int(CAMERA_MARTRIX_LC.at<float>(0, 2)) + 5, int(CAMERA_MARTRIX_LC.at<float>(1, 2)) - 5), 1, 2, Scalar(0, 255, 255), 2);
				//putText(img_lc, "x", Point(int(CAMERA_MARTRIX_LC.at<float>(0, 2)) - 5, int(CAMERA_MARTRIX_LC.at<float>(1, 2)) + 5), 1, 1, Scalar(0, 255, 255), 1);
				string angle = "Ang: ";
				string position = "Pos: ";
				for (int j = 0; j < 3; j++)
				{
					angle += to_string(int(R_euler.val[j])) + " ";
					position += to_string(int(worldm.at<float>(j, 0))) + " ";
				}
				rectangle(img_lc, Rect(0, 20 * (i), 200, 60), Scalar(255, 0, 0));
				putText(img_lc, angle, Point(10, 20 * (i + 1)), 1, 1, Scalar(0, 255, 255));
				putText(img_lc, position, Point(10, 20 * (i + 2)), 1, 1, Scalar(0, 255, 255));
				imshow("lc", img_lc);
				waitKey(10);
#endif
		}
				

			
			//waitKey(30);
		}
		waitKey(0);
			}

