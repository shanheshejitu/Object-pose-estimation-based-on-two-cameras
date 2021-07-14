#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

//��ʾЧ����ע������ʾ
#define _DEBUG

using namespace cv;
using namespace std;

const float MARKER_CELL_LENGTH = 22.0;
const int MARKER_SCALE = 7;
const float MARKER_LENGTH = MARKER_CELL_LENGTH * MARKER_SCALE;

//��ʶ���Marker��1��ʾ��ɫ����0��ʾ��ɫ��������㲻���빹��Marker
const int TARGET_SIZE = 1;	//˫Ŀ�����Ұ��ֻ��һ��Ŀ�꣬��Ŀ�긴��
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
* @brief ��ȡͼƬ���������Ƶ
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
* @brief ��ȡ��������������������һ����ȡ��������
* @param bin��ֵͼ
* @param contours������������޳���С������
* @return false��ʾû���ҵ�����
*/
bool getGoodContours(Mat& bin, vector<vector<Point>>& contours, int min_threshold);

/**
* @brief �������ܳ�
* @param a����Ķ���Σ��ö����ʾ
* @return ����ε��ܳ���������
*/
float perimeter(const vector<Point2f> &a);

/**
* @brief ˳ʱ����ת����
* @param m����ľ���M x M����ֵ��ֱ�Ӹı�
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
* @brief �ж����������Ƿ���ͬ
* @param m��n���Ƚϵľ���
* @return ������ͬ�����true������false
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
* @brief ��ת���Σ������ö����ʾ
* @param points����ת���εĶ���
*/
void rotateRectangle(vector<Point2f> &points);

/**
* @brief ����ת����ת��Ϊŷ���Ǳ�ʾ
* @param R��ת����
* @return ŷ����
*/
Vec3f rotationMatrixToEulerAngles(Mat &R);

/**
* @brief �޳�һЩ������Ҫ���Ŀ��
* @param possible_markers���ڴ洢���ܵ�Ŀ��
* @param good_markers���ڴ洢�õ�Ŀ��
* @param min_threshold ��ֵ��ԽСҪ��Խ��
*/
void choiceGoodMarkers(vector <Marker> &possible_markers, vector <Marker> &good_markers, int min_threshold);

/**
* @brief ��ȡ��Ұ�����п��ܵ�Ŀ��
* @param contours����
* @param possible_markers���ڴ洢���ܵ�Ŀ��
* @param min_threshold ��ֵ��ԽСҪ��Խ��
*/
void getAllMarkers(vector<vector<Point>> &contours, vector <Marker> &possible_markers, int min_threshold);

/**
* @brief ��ȡMarkerͼ���id
* @param mat_bin_marker Marker��ֵͼ
* @param id_marker Markerid������
*/
void getMarkerId(Mat &mat_bin_marker, bool(*id_marker)[MARKER_SCALE]);

/**
* @brief PnP���㣬��ת�����������ϵ
* @param real_world_marker��������ϵ
* @param camera_pointsͼ������ϵ
* @param R\T��������ϵ���������ϵ����ת�����ƽ�ƾ���
* @param c2w��������ϵԭ�����������ϵ�µ�����
*/
void getC2WPosition(vector<Point3f> &real_world_marker, vector<Point2f> &camera_points, Mat &R, Mat &T, Mat &c2w, const Mat &CAMERA_MARTRIX, const Mat &DIST_COEFFS);

/**
* @brief �����жϾ����ǲ��ǿ��Թ���һ��Marker
* @param id_marker���жϵľ���
* @return ����true�򹹳�Marker�����򲻹���Marker
*/
bool isMarker(bool id_marker[MARKER_SCALE][MARKER_SCALE]);

/**
* @brief ŷ���Ǽ����Ӧ����ת����
* @param theta ŷ����
* @param id_marker Markerid������
*/
void eulerAnglesToRotationMatrix(Vec3f &theta, Mat &R);

/**
* @brief ��ȡmakerȫ����
* @img ����ͼ��
* @real_marker ��ʵmarker
* @dst_markers ���marker
* @name ��ʾͼ�������
*/
bool getGoodMarker(Mat &img, vector<Point2f> &real_marker, vector<Marker> &dst_markers, string name = "img");

/**
* @brief ˫Ŀ�����������õ���������
* @param uv_lc��uv_rc �����������Ҫ��������꣬��ƥ���һ�Ե�
* @param world �������������
* @param R_LC T_LC I_LC R_RC T_RC I_RC�������ת���� �����ƽ�ƾ��� ������ڲξ��� �������ת���� �����ƽ�ƾ��� ������ڲξ���
*/
void uv2xyz(Point2f &uv_lc, Point2f &uv_rc, Point3f &world, const Mat R_LC, const Mat T_LC, const Mat I_LC, const Mat R_RC, const Mat T_RC, const Mat I_RC);

/**
* @brief ˫Ŀ������ƥ��
* @param img_lc��img_rc �������ͼ��
* @param kp_lc��kp_rc ����ͼ��������
* @param matches ƥ��
* @param max_times��min_thresholdƥ�侫����ֵ
*/
void matchPoints(Mat &img_lc, Mat &img_rc, vector<KeyPoint> &kp_lc, vector<KeyPoint> &kp_rc, vector<DMatch> &matches, const double max_times, const double min_threshold);

/**
* @brief ˫Ŀ����������һ�����ڼ���������λ��
* @param ʡ��...
*/
bool testEtR(Point2f &p1, Point2f &p2, Mat &test_t, Mat &test_R, const Mat K1, const Mat K2, Mat &R, Mat &t);
void getC2CRT(vector<KeyPoint> &kp_lc, vector<KeyPoint> &kp_rc, vector<DMatch> &matches, const Mat CAMERA_MARTRIX_LC, const Mat CAMERA_MARTRIX_RC, Mat &R, Mat &T);
void FindTransformMat(vector<Point3f> real_world_marker, vector<Point3f> camera_world_marker, Mat &R, Mat &t);