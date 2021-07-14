#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

//显示效果，注释则不显示
#define _DEBUG

using namespace cv;
using namespace std;

const float MARKER_CELL_LENGTH = 22.0;
const int MARKER_SCALE = 7;
const float MARKER_LENGTH = MARKER_CELL_LENGTH * MARKER_SCALE;

//待识别的Marker，1表示白色区域，0表示黑色区域，最外层不参与构成Marker
const int TARGET_SIZE = 1;	//双目最好视野中只有一个目标，多目标复杂
const bool TARGET[TARGET_SIZE][MARKER_SCALE - 2][MARKER_SCALE - 2] = {
	{
		{ 1, 1, 1, 0, 1 },
		{ 1, 1, 1, 0, 1 },
		{ 1, 1, 1, 0, 1 },
		{ 0, 1, 1, 1, 0 },
		{ 0, 0, 0, 0, 1 }
	}
};

typedef struct Marker{
	vector<Point2f> points;
	bool id[MARKER_SCALE - 2][MARKER_SCALE - 2];
	bool is_good = false;
	int index = -1;
} Marker;

/**
* @brief 读取图片或相机或视频
* @demo:
	loadImg img("1.jpg", "2.jpg");
	loadImg video("1.mp4", "");
	loadImg camera("0", "1");
*/
class loadImg{
public:
	loadImg(){}
	loadImg(string img_lc_path, string img_rc_path){
		if (!img_lc_path.empty() && !img_rc_path.empty()){
			if ((img_lc_path.size() < 2) && (img_rc_path.size() < 2)){
				//cout << "opening cameras" << int(atoi(img_lc_path.c_str())) << endl;
				cap_lc.open(int(atoi(img_lc_path.c_str())));
				cap_rc.open(int(atoi(img_rc_path.c_str())));
				is_video = false;
				is_cap = true;
			}
			else{
				cap_lc.open(img_lc_path);
				cap_rc.open(img_rc_path);
				is_video = false;
				is_cap = false;
			}
		}
		else{
			string video_path = img_lc_path.empty() ? img_rc_path : img_lc_path;
			cap.open(video_path);
			is_video = true;
		}
	}
	~loadImg(){}

	void load(Mat &img_lc, Mat &img_rc){
		if (!is_video){
			cap_lc >> img_lc;
			//cout << img_lc.size();
			cap_rc >> img_rc;
			if (is_cap) {
				resize(img_lc, img_lc, Size(1080, 720));
				resize(img_rc, img_rc, Size(1080, 720));
			}
		}
		else{
			Mat img;
			cap >> img;
			Mat(img(Rect(0, 0, img.rows, img.cols / 2))).copyTo(img_lc);
			Mat(img(Rect(img.cols / 2, 0, img.rows, img.cols / 2))).copyTo(img_rc);
		}
	}
private:
	VideoCapture cap_lc, cap_rc, cap;
	bool is_video;
	bool is_cap;
};

/**
* @brief 提取轮廓并对轮廓分析，进一步获取车牌区域
* @param bin二值图
* @param contours输出轮廓，会剔除过小的轮廓
* @return false表示没有找到轮廓
*/
bool getGoodContours(Mat& bin, vector<vector<Point>>& contours, int min_threshold);

/**
* @brief 求多边形周长
* @param a输入的多边形，用顶点表示
* @return 多边形的周长，浮点型
*/
float perimeter(const vector<Point2f> &a);

/**
* @brief 顺时针旋转矩阵
* @param m输入的矩阵【M x M】，值会直接改变
*/
template <class T, int M>
void rotateMatrix(T(&m)[M][M]){
	T n[M][M];
	for (int i = 0; i < M; i++){
		for (int j = 0; j < M; j++){
			n[i][j] = m[j][M - 1 - i];
		}
	}
	for (int i = 0; i < M; i++){
		for (int j = 0; j < M; j++){
			m[i][j] = n[i][j];
		}
	}
}

/**
* @brief 判断两个矩阵是否相同
* @param m、n待比较的矩阵
* @return 矩阵相同则输出true，否则false
*/
template <class T, int M>
bool isSameMatrix(T(&m)[M][M], const T(&n)[M][M]){
	for (int i = 0; i < M; i++){
		for (int j = 0; j < M; j++){
			if (m[i][j] != n[i][j])
				return false;
		}
	}
	return true;
}

/**
* @brief 旋转矩形，矩形用顶点表示
* @param points待旋转矩形的顶点
*/
void rotateRectangle(vector<Point2f> &points);

/**
* @brief 将旋转矩阵转换为欧拉角表示
* @param R旋转矩阵
* @return 欧拉角
*/
Vec3f rotationMatrixToEulerAngles(Mat &R);

/**
* @brief 剔除一些不符合要求的目标
* @param possible_markers用于存储可能的目标
* @param good_markers用于存储好的目标
* @param min_threshold 阈值，越小要求越高
*/
void choiceGoodMarkers(vector <Marker> &possible_markers, vector <Marker> &good_markers, int min_threshold);

/**
* @brief 获取视野中所有可能的目标
* @param contours轮廓
* @param possible_markers用于存储可能的目标
* @param min_threshold 阈值，越小要求越高
*/
void getAllMarkers(vector<vector<Point>> &contours, vector <Marker> &possible_markers, int min_threshold);

/**
* @brief 获取Marker图像的id
* @param mat_bin_marker Marker二值图
* @param id_marker Markerid的数组
*/
void getMarkerId(Mat &mat_bin_marker, bool(*id_marker)[MARKER_SCALE]);

/**
* @brief PnP解算，并转换成相机坐标系
* @param real_world_marker世界坐标系
* @param camera_points图像坐标系
* @param R\T世界坐标系到相机坐标系的旋转矩阵和平移矩阵
* @param c2w世界坐标系原点在相机坐标系下的坐标
*/
void getC2WPosition(vector<Point3f> &real_world_marker, vector<Point2f> &camera_points, Mat &R, Mat &T, Mat &c2w, const Mat &CAMERA_MARTRIX, const Mat &DIST_COEFFS);

/**
* @brief 用于判断矩阵是不是可以构成一个Marker
* @param id_marker待判断的矩阵
* @return 返回true则构成Marker，否则不构成Marker
*/
bool isMarker(bool id_marker[MARKER_SCALE][MARKER_SCALE]);

/**
* @brief 欧拉角计算对应的旋转矩阵
* @param theta 欧拉角
* @param id_marker Markerid的数组
*/
void eulerAnglesToRotationMatrix(Vec3f &theta, Mat &R);

/**
* @brief 获取maker全流程
* @img 输入图像
* @real_marker 真实marker
* @dst_markers 输出marker
* @name 显示图像的名字
*/
bool getGoodMarker(Mat &img, vector<Point2f> &real_marker, vector<Marker> &dst_markers, string name = "img");

/**
* @brief 双目，由相机坐标得到世界坐标
* @param uv_lc、uv_rc 左右相机中需要计算的坐标，是匹配的一对点
* @param world 输出的世界坐标
* @param R_LC T_LC I_LC R_RC T_RC I_RC左相机旋转矩阵 左相机平移矩阵 左相机内参矩阵 右相机旋转矩阵 右相机平移矩阵 右相机内参矩阵
*/
void uv2xyz(Point2f &uv_lc, Point2f &uv_rc, Point3f &world, const Mat R_LC, const Mat T_LC, const Mat I_LC, const Mat R_RC, const Mat T_RC, const Mat I_RC);

/**
* @brief 双目，特征匹配
* @param img_lc、img_rc 左右相机图像
* @param kp_lc、kp_rc 左右图像特征点
* @param matches 匹配
* @param max_times、min_threshold匹配精度阈值
*/
void matchPoints(Mat &img_lc, Mat &img_rc, vector<KeyPoint> &kp_lc, vector<KeyPoint> &kp_rc, vector<DMatch> &matches, const double max_times, const double min_threshold);

/**
* @brief 双目，两个函数一起，用于计算相机相对位姿
* @param 省略...
*/
bool testEtR(Point2f &p1, Point2f &p2, Mat &test_t, Mat &test_R, const Mat K1, const Mat K2, Mat &R, Mat &t);
void getC2CRT(vector<KeyPoint> &kp_lc, vector<KeyPoint> &kp_rc, vector<DMatch> &matches, const Mat CAMERA_MARTRIX_LC, const Mat CAMERA_MARTRIX_RC, Mat &R, Mat &T);
void FindTransformMat(vector<Point3f> real_world_marker, vector<Point3f> camera_world_marker, Mat &R, Mat &t);