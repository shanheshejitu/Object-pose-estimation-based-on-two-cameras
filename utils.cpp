#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2\objdetect\objdetect_c.h> 
#include "utils.h"
#include <opencv2\imgproc\types_c.h>

/**
* @brief 提取轮廓并对轮廓分析，进一步获取车牌区域
* @param bin二值图
* @param contours输出轮廓，会剔除过小的轮廓
* @return false表示没有找到轮廓
*/
bool getGoodContours(Mat& bin, vector<vector<Point>>& contours, int min_threshold){
	vector<vector<Point>> all_contours;	//all_contours表示所有轮廓，contours表示期望的轮廓
	vector<Vec4i> hierarchy;
	findContours(bin, all_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

	//通过轮廓的长度（像素数目）筛选轮廓，剔除过短的轮廓
	if (all_contours.size() <= 0)
		return false;
	for (int i = 0; i < all_contours.size(); i++){
		if (all_contours[i].size() < min_threshold)
			continue;
		contours.push_back(all_contours[i]);
	}
	return true;
}

/**
* @brief 求多边形周长
* @param a输入的多边形，用顶点表示
* @return 多边形的周长，浮点型
*/
float perimeter(const vector<Point2f> &a){
	float sum = 0, dx, dy;
	for (size_t i = 0; i<a.size(); i++)
	{
		size_t i2 = (i + 1) % a.size();

		dx = a[i].x - a[i2].x;
		dy = a[i].y - a[i2].y;

		sum += sqrt(dx*dx + dy*dy);
	}

	return sum;
}

/**
* @brief 旋转矩形，矩形用顶点表示
* @param points待旋转矩形的顶点
*/
void rotateRectangle(vector<Point2f> &points){
	if (points.size() != 4) return;
	swap(points[0], points[3]);
	swap(points[0], points[2]);
	swap(points[0], points[1]);
}

/**
* @brief 将旋转矩阵转换为欧拉角表示
* @param R旋转矩阵
* @return 欧拉角
*/
Vec3f rotationMatrixToEulerAngles(Mat &R){

	assert(isRotationMatrix(R));

	float sy = sqrt(R.at<float>(0, 0) * R.at<float>(0, 0) + R.at<float>(1, 0) * R.at<float>(1, 0));

	bool singular = sy < 1e-6; // If

	float x, y, z;
	if (!singular)
	{
		x = atan2(R.at<float>(2, 1), R.at<float>(2, 2));
		y = atan2(-R.at<float>(2, 0), sy);
		z = atan2(R.at<float>(1, 0), R.at<float>(0, 0));
	}
	else
	{
		x = atan2(-R.at<float>(1, 2), R.at<float>(1, 1));
		y = atan2(-R.at<float>(2, 0), sy);
		z = 0;
	}
	return Vec3f(x / CV_PI * 180, y / CV_PI * 180, z / CV_PI * 180);
}

/**
* @brief 获取视野中所有可能的目标
* @param contours轮廓
* @param possible_markers用于存储可能的目标
* @param min_threshold 阈值，越小要求越高
*/
void getAllMarkers(vector<vector<Point>> &contours, vector <Marker> &possible_markers, int min_threshold){
	//返回结果为多边形，用点集表示相似形状
	vector <Point> approxCurve;
	for (int i = 0; i < contours.size(); i++){
		//近似一个多边形逼近，为了减少轮廓的像素
		float eps = contours[i].size() * 0.05;
		//输入图像的轮廓，输出结果，估计精度，是否闭合。输出多边形的顶点组成的点集
		//使多边形边缘平滑，得到近似的多边形 
		approxPolyDP(contours[i], approxCurve, eps, true);
		//我们感兴趣的多边形只有四个顶点
		if (approxCurve.size() != 4)
			continue;
		//检查轮廓是否是凸边形
		if (!isContourConvex(approxCurve))
			continue;

		//确保连续点之间的距离是足够大的。
		float minDist = 1e10;

		//求当前四边形各顶点之间的最短距离，用来剔除太小的目标
		for (int i = 0; i<4; i++)
		{
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];		//这里应该是2维的相减
			float squaredSideLength = side.dot(side);					//求2维向量的点积，就是XxY
			minDist = min(minDist, squaredSideLength);					//找出最小的距离
		}

		//检查距离是不是特别小，小的话就退出本次循环，开始下一次循环
		if (minDist < min_threshold)
			continue;

		Marker marker;
		for (int i = 0; i<4; i++)
			marker.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));

		//将点按顺序储存，这里使用逆时针储存
		Point v1 = marker.points[1] - marker.points[0];
		Point v2 = marker.points[2] - marker.points[0];
		float s = (v1.x * v2.y) - (v1.y * v2.x);
		//如果第三个点在左边，那么交换第一个点和第三个点，逆时针保存
		if (s < 0.0)
			swap(marker.points[1], marker.points[3]);
		//储存可能的目标，后面还需要筛选
		possible_markers.push_back(marker);
	}

}

/**
* @brief 剔除一些不符合要求的目标
* @param possible_markers用于存储可能的目标
* @param good_markers用于存储好的目标
* @param min_threshold 阈值，越小要求越高
*/
void choiceGoodMarkers(vector <Marker> &possible_markers, vector <Marker> &good_markers, int min_threshold){
	//移除那些角点互相离的太近的四边形
	vector< pair<int, int> > near_candidates;
	for (size_t i = 0; i<possible_markers.size(); i++)
	{
		const Marker& m_1 = possible_markers[i];

		//计算两个marker四边形之间的距离，四组点之间距离和的平均值，若平均值较小，则认为两个maker很相近
		for (size_t j = i + 1; j<possible_markers.size(); j++)
		{
			const Marker& m_2 = possible_markers[j];
			float distSquared = 0;
			for (int c = 0; c<4; c++)
			{
				Point v = m_1.points[c] - m_2.points[c];
				distSquared += v.dot(v);
			}
			distSquared /= 4;
			//距离很近，剔除目标，这里先标记目标，之后再剔除
			if (distSquared < min_threshold)
			{
				near_candidates.push_back(pair<int, int>(i, j));
			}
		}
	}

	//开始剔除目标
	vector<bool> removal_mask(possible_markers.size(), false);
	for (size_t i = 0; i<near_candidates.size(); i++)
	{
		//求这一对相邻四边形的周长
		float p1 = perimeter(possible_markers[near_candidates[i].first].points);
		float p2 = perimeter(possible_markers[near_candidates[i].second].points);

		//谁周长小，移除谁
		size_t removal_index;
		if (p1 > p2)
			removal_index = near_candidates[i].second;
		else
			removal_index = near_candidates[i].first;

		removal_mask[removal_index] = true;
	}

	for (size_t i = 0; i < possible_markers.size(); i++)
	{
		if (!removal_mask[i])
			good_markers.push_back(possible_markers[i]);
	}
}

/**
* @brief 获取Marker图像的id
* @param mat_bin_marker Marker二值图
* @param id_marker Markerid的数组
*/
void getMarkerId(Mat &mat_bin_marker, bool(*id_marker)[MARKER_SCALE]){
	//将目标AR Marker转换成7x7的矩阵，并且最外围应该是0，这样可用用于判断分割是不是好的
	for (int j = 0; j < MARKER_SCALE; j++){
		for (int k = 0; k < MARKER_SCALE; k++){
			int back_pixels = 0;
			int white_pixels = 0;
			for (int l = 0; l < MARKER_CELL_LENGTH; l++){
				//只检查对角线的像素值
				int check_x = k * MARKER_CELL_LENGTH + l;
				int check_y = j * MARKER_CELL_LENGTH + l;
				if (mat_bin_marker.at<uchar>(check_y, check_x) > 125){
					white_pixels++;
				}
				else
				{
					back_pixels++;
				}
				//如果对角线上黑色像素多于白色像素，则这个格子为0，否则为1
				if (back_pixels >= white_pixels){
					id_marker[j][k] = 0;
				}
				else
				{
					id_marker[j][k] = 1;
				}
				//mat_bin_marker.at<uchar>(check_y, check_x) = 255;
			}
		}
	}
}

/**
* @brief 欧拉角计算对应的旋转矩阵
* @param theta 欧拉角
* @param R 旋转矩阵
*/
void eulerAnglesToRotationMatrix(Vec3f &theta, Mat &R){
	// 计算旋转矩阵的X分量
	Mat R_x = (Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cos(theta[0]), -sin(theta[0]),
		0, sin(theta[0]), cos(theta[0])
		);

	// 计算旋转矩阵的Y分量
	Mat R_y = (Mat_<float>(3, 3) <<
		cos(theta[1]), 0, sin(theta[1]),
		0, 1, 0,
		-sin(theta[1]), 0, cos(theta[1])
		);

	// 计算旋转矩阵的Z分量
	Mat R_z = (Mat_<float>(3, 3) <<
		cos(theta[2]), -sin(theta[2]), 0,
		sin(theta[2]), cos(theta[2]), 0,
		0, 0, 1);

	// 合并 
	R = R_z * R_y * R_x;
}

/**
* @brief PnP解算，并转换成相机坐标系
* @param real_world_marker世界坐标系
* @param camera_points图像坐标系
* @param R\T世界坐标系到相机坐标系的旋转矩阵和平移矩阵
* @param c2w世界坐标系点在相机坐标系下的坐标
*/
void getC2WPosition(vector<Point3f> &real_world_marker, vector<Point2f> &camera_points, Mat &R, Mat &T, Mat &c2w, const Mat &CAMERA_MARTRIX, const Mat &DIST_COEFFS){
	//通过PNP解算位姿，PNP解算需要三个点，但是三个点可以得出四种可能结果，在需要一个点计算最优结果
	//计算结果与世界坐标系的定义有关
	//此处将世界坐标系原点定义为AR Marker的左上角一点...
	Mat r;
	solvePnP(real_world_marker, camera_points, CAMERA_MARTRIX, DIST_COEFFS, r, T);
	Rodrigues(r, R);

	//通过RT矩阵，将世界坐标系转换到相机坐标系Tc = [TR] Tw
	Mat rt(4, 4, CV_64F);
	Mat t(4, 1, CV_64F);
	for (int j = 0; j < 3; j++){
		for (int k = 0; k < 3; k++){
			rt.at<float>(j, k) = R.at<float>(j, k);
		}
	}
	for (int j = 0; j < 3; j++){
		rt.at<float>(j, 3) = T.at<float>(j, 0);
		rt.at<float>(3, j) = 0;
	}
	t.at<float>(0, 0) = MARKER_CELL_LENGTH * MARKER_SCALE / 2;		//待转换的点，这里取标记中心
	t.at<float>(1, 0) = MARKER_CELL_LENGTH * MARKER_SCALE / 2;		//待转换的点，这里取标记中心
	t.at<float>(2, 0) = 0;		//待转换的点，这里取标记中心
	rt.at<float>(3, 3) = 1;
	t.at<float>(3, 0) = 1;
	c2w = rt * t;
}

/**
* @brief 用于判断矩阵是不是可以构成一个Marker
* @param id_marker待判断的矩阵
* @return 返回true则构成Marker，否则不构成Marker
*/
bool isMarker(bool id_marker[MARKER_SCALE][MARKER_SCALE]){
	int zero_x[2] = { 0, MARKER_SCALE - 1 };	//用于判断目标的外围是不是全是0
	int zero_y[2] = { 0, MARKER_SCALE - 1 };

	//开始检测AR Marker的外围是不是全是0
	bool is_border_zero = true;
	//检测第一和最后一列
	for (int j = 0; j < 2; j++){
		for (int k = 0; k < MARKER_SCALE; k++){
			if (id_marker[zero_y[j]][k] != 0){
				is_border_zero = false;
			}
		}
	}
	//检测第一和最后一行
	for (int j = 0; j < 2; j++){
		for (int k = 0; k < MARKER_SCALE; k++){
			if (id_marker[k][zero_x[j]] != 0){
				is_border_zero = false;
			}
		}
	}
	return is_border_zero;
}


/**
* @brief 获取maker全流程
* @param img 输入图像
* @param real_marker 真实marker
* @param dst_markers 输出marker
* @param name 显示图像的名字
* @return 找到返回true
*/
bool getGoodMarker(Mat &img, vector<Point2f> &real_marker, vector<Marker> &dst_markers, string name){
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	Mat bin;
	adaptiveThreshold(gray, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);

	vector<vector<Point>> contours;
	bool is_success = getGoodContours(bin, contours, 100);
	if (!is_success) return false;

	//开始提取一些可能的Marker
	vector <Marker> possible_markers;
	getAllMarkers(contours, possible_markers, 100);
	vector<Marker> good_markers;
	choiceGoodMarkers(possible_markers, good_markers, 100);

	//透视变换，用于提取目标
	vector<Mat> mat_markers;
	vector<Mat> transforms;
	for (size_t i = 0; i < good_markers.size(); i++)
	{
		Marker& marker = good_markers[i];

		Size marker_size = Size(real_marker[3].y - real_marker[0].y, real_marker[1].x - real_marker[0].x);
		Mat markerTransform = getPerspectiveTransform(marker.points, real_marker);	//计算变换矩阵
		transforms.push_back(markerTransform);

		Mat mat_marker;
		warpPerspective(gray, mat_marker, markerTransform, marker_size);
		mat_markers.push_back(mat_marker);
	}

	//提取并分析目标，用于计算AR Marker的值是否与需求相等
	for (int i = 0; i < mat_markers.size(); i++){
		bool id_marker[MARKER_SCALE][MARKER_SCALE];
		Mat mat_bin_marker;
		//转化为二值图方便将Marker转换成矩阵
		threshold(mat_markers[i], mat_bin_marker, 125, 255, THRESH_BINARY | THRESH_OTSU);
		getMarkerId(mat_bin_marker, id_marker);
		if (!isMarker(id_marker)) continue;

		//将Marker的实际内容分离，即最外层是无效信息，只能用来判断分割是不是好的
		bool good_id_marker[MARKER_SCALE - 2][MARKER_SCALE - 2];
		for (int j = 0; j < MARKER_SCALE - 2; j++){
			for (int k = 0; k < MARKER_SCALE - 2; k++){
				good_markers[i].id[j][k] = id_marker[j + 1][k + 1];
			}
		}

		//将Marker的内容与目标比较，比较时需要考虑旋转，因此比对不上时，旋转Marker矩阵再次比对
		bool is_good = false;
		for (int j = 0; j < 4; j++){
			for (int k = 0; k < TARGET_SIZE; k++){
				//判断是否相同
				is_good = isSameMatrix(good_markers[i].id, TARGET[k]);
				if (is_good) {
					good_markers[i].index = k;
					break;
				}
			}
			if (is_good) break;

			//旋转矩阵
			rotateMatrix(good_markers[i].id);
			//旋转Marker坐标，用于后续PNP解算
			rotateRectangle(good_markers[i].points);

#if defined _DEBUG
			Point center(mat_bin_marker.cols / 2, mat_bin_marker.rows / 2); //旋转中心
			double angle = 90.0;											//角度
			double scale = 1.0;												//缩放系数
			Mat rotMat = getRotationMatrix2D(center, angle, scale);
			warpAffine(mat_bin_marker, mat_bin_marker, rotMat, Size(mat_bin_marker.rows, mat_bin_marker.cols));
#endif
		}
		good_markers[i].is_good = is_good;

#if defined _DEBUG
		//开始输出Marker矩阵，视野中可能有多个符合要求的Marker
		if (!is_good) continue;
		cout << name << "_AR标记[" << i << "]：" << endl;
		for (int j = 0; j < MARKER_SCALE - 2; j++){
			for (int k = 0; k < MARKER_SCALE - 2; k++)
			{
				cout << good_markers[i].id[j][k] << " ";
			}
			cout << endl;
		}
		imshow(name + "_bin", mat_bin_marker);

		//显示地面
		Mat test;
		warpPerspective(img, test, transforms[i], img.size());
		imshow(name, test);
#endif
	}

	for (int i = 0; i < good_markers.size(); i++){
		if (good_markers[i].is_good){
			dst_markers.push_back(good_markers[i]);
		}
	}
	return (dst_markers.size() > 0);
}

/**
* @brief 双目，由相机坐标得到世界坐标
* @param uv_lc、uv_rc 左右相机中需要计算的坐标，是匹配的一对点
* @param world 输出的世界坐标
* @param R_LC T_LC I_LC R_RC T_RC I_RC左相机旋转矩阵 左相机平移矩阵 左相机内参矩阵 右相机旋转矩阵 右相机平移矩阵 右相机内参矩阵
*/
void uv2xyz(Point2f &uv_lc, Point2f &uv_rc, Point3f &world, const Mat R_LC, const Mat T_LC, const Mat I_LC, const Mat R_RC, const Mat T_RC, const Mat I_RC){
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	Mat rt_l = Mat(3, 4, CV_32F);//左相机M矩阵
	hconcat(R_LC, T_LC, rt_l);
	Mat m_l = I_LC * rt_l;
	//cout << "左相机M矩阵 = " << endl << m_l << endl;

	Mat rt_r = Mat(3, 4, CV_32F);//右相机M矩阵
	hconcat(R_RC, T_RC, rt_r);
	Mat m_r = I_RC * rt_r;
	//cout << "右相机M矩阵 = " << endl << m_r << endl;

	//最小二乘法A矩阵
	Mat A = Mat(4, 3, CV_32F);
	A.at<float>(0, 0) = uv_lc.x * m_l.at<float>(2, 0) - m_l.at<float>(0, 0);
	A.at<float>(0, 1) = uv_lc.x * m_l.at<float>(2, 1) - m_l.at<float>(0, 1);
	A.at<float>(0, 2) = uv_lc.x * m_l.at<float>(2, 2) - m_l.at<float>(0, 2);

	A.at<float>(1, 0) = uv_lc.y * m_l.at<float>(2, 0) - m_l.at<float>(1, 0);
	A.at<float>(1, 1) = uv_lc.y * m_l.at<float>(2, 1) - m_l.at<float>(1, 1);
	A.at<float>(1, 2) = uv_lc.y * m_l.at<float>(2, 2) - m_l.at<float>(1, 2);

	A.at<float>(2, 0) = uv_rc.x * m_r.at<float>(2, 0) - m_r.at<float>(0, 0);
	A.at<float>(2, 1) = uv_rc.x * m_r.at<float>(2, 1) - m_r.at<float>(0, 1);
	A.at<float>(2, 2) = uv_rc.x * m_r.at<float>(2, 2) - m_r.at<float>(0, 2);

	A.at<float>(3, 0) = uv_rc.y * m_r.at<float>(2, 0) - m_r.at<float>(1, 0);
	A.at<float>(3, 1) = uv_rc.y * m_r.at<float>(2, 1) - m_r.at<float>(1, 1);
	A.at<float>(3, 2) = uv_rc.y * m_r.at<float>(2, 2) - m_r.at<float>(1, 2);

	//最小二乘法B矩阵
	Mat B = Mat(4, 1, CV_32F);
	B.at<float>(0, 0) = m_l.at<float>(0, 3) - uv_lc.x * m_l.at<float>(2, 3);
	B.at<float>(1, 0) = m_l.at<float>(1, 3) - uv_lc.y * m_l.at<float>(2, 3);
	B.at<float>(2, 0) = m_r.at<float>(0, 3) - uv_rc.x * m_r.at<float>(2, 3);
	B.at<float>(3, 0) = m_r.at<float>(1, 3) - uv_rc.y * m_r.at<float>(2, 3);

	Mat xyz = Mat(3, 1, CV_32F);
	//采用SVD最小二乘法求解XYZ
	solve(A, B, xyz, DECOMP_SVD);

	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;

	//世界坐标系中坐标
	world.x = xyz.at<float>(0, 0);
	world.y = xyz.at<float>(1, 0);
	world.z = xyz.at<float>(2, 0);
}

/**
* @brief 双目，特征匹配
* @param img_lc、img_rc 左右相机图像
* @param kp_lc、kp_rc 左右图像特征点
* @param matches 匹配
* @param max_times、min_threshold匹配精度阈值
*/
void matchPoints(Mat &img_lc, Mat &img_rc, vector<KeyPoint> &kp_lc, vector<KeyPoint> &kp_rc, vector<DMatch> &matches, const double max_times, const double min_threshold){
	Ptr<ORB> orb = ORB::create();
	
	Mat dp_lc, dp_rc;

	orb->detect(img_lc, kp_lc);
	orb->detect(img_rc, kp_rc);
	orb->compute(img_lc, kp_lc, dp_lc);
	orb->compute(img_rc, kp_rc, dp_rc);
	//orb(img_lc, Mat(), kp_lc, dp_lc);
	//orb(img_rc, Mat(), kp_rc, dp_rc);

	vector<DMatch> all_matches;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(dp_lc, dp_rc, all_matches);

	double min_dist = DBL_MAX, max_dist = -1;
	for (int i = 0; i < dp_lc.rows; i++){
		double dist = all_matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	for (int i = 0; i < dp_lc.rows; i++){
		if (all_matches[i].distance <= max(max_times * min_dist, min_threshold)){
			matches.push_back(all_matches[i]);
		}
	}

#if defined _DEBUG
	Mat img;
	drawMatches(img_lc, kp_lc, img_rc, kp_rc, matches, img);
	resize(img, img, Size(img.cols / 2, img.rows / 2));
	imshow("matches", img);
#endif

}

/**
* @brief 双目，两个函数一起，用于计算相机相对位姿

*/
bool testEtR(Point2f &p1, Point2f &p2, Mat &test_t, Mat &test_R, const Mat K1, const Mat K2, Mat &R, Mat &t){
	Mat R_invert, K1_invert, K2_invert;
	invert(test_R, R_invert);
	invert(K1, K1_invert);
	invert(K2, K2_invert);

	Mat p1_mat = (Mat_<float>(3, 1) << p1.x, p1.y, 1.0);
	Mat p2_mat = (Mat_<float>(3, 1) << p2.x, p2.y, 1.0);
	Mat P1 = K1_invert * p1_mat;
	Mat P2 = R_invert * (K2_invert * p2_mat - test_t);

	if ((P1.at<float>(2, 0) > 0) && (P2.at<float>(2, 0) > 0))
	{
		test_t.copyTo(t);
		test_R.copyTo(R);
		return true;
	}
	return false;
}
void getC2CRT(vector<KeyPoint> &kp_lc, vector<KeyPoint> &kp_rc, vector<DMatch> &matches, const Mat CAMERA_MARTRIX_LC, const Mat CAMERA_MARTRIX_RC, Mat &R, Mat &T){
	vector<Point2f> p_lc, p_rc;
	for (int i = 0; i < matches.size(); i++){
		p_lc.push_back(kp_lc[matches[i].queryIdx].pt);
		p_rc.push_back(kp_rc[matches[i].trainIdx].pt);
	}
	//cout << p_lc.size() << endl;

	Mat F = findFundamentalMat(p_lc, p_rc, FM_8POINT);
	F.convertTo(F, CV_32F);
	Mat K1 = CAMERA_MARTRIX_LC;
	Mat K2T = CAMERA_MARTRIX_RC.t(); 
	Mat E = K2T * F * K1;

	SVD svd;
	Mat W, VT, U;
	svd.compute(E, W, U, VT, SVD::FULL_UV);
	W.convertTo(W, CV_32F);
	U.convertTo(U, CV_32F);
	VT.convertTo(VT, CV_32F);
	Mat full_W = (Mat_<float>(3, 3) << W.at<float>(0, 0), .0, .0, .0, W.at<float>(1, 0), .0, .0, .0, W.at<float>(2, 0));
	Mat tR = (Mat_<float>(3, 3) << .0, 1.0, .0, -1.0, .0, .0, .0, .0, -1.0);
	Mat nR = (Mat_<float>(3, 3) << .0, -1.0, .0, 1.0, .0, .0, .0, .0, -1.0);

	Mat t1 = U * tR * full_W * U.t();
	Mat t2 = U * nR * full_W * U.t();
	cout << "t1:" << t1 << endl;
	cout << "t2:" << t2 << endl;
	Mat tT1(3, 1, CV_32F), tT2(3, 1, CV_32F);
	//tT1.at<float>(0, 2) = t1.at<float>(2, 1);
	//tT1.at<float>(0, 1) = t1.at<float>(0, 2);
	//tT1.at<float>(0, 0) = t1.at<float>(1, 0);
	tT1.at<float>(0,0) = t1.at<float>(1, 0);
	tT1.at<float>(0, 1) = t1.at<float>(0, 2);
	tT1.at<float>(0, 2) = t1.at<float>(2, 1);
	//tT2.at<float>(0, 2) = t2.at<float>(2, 1);
	//tT2.at<float>(0, 1) = t2.at<float>(0, 2);
	//tT2.at<float>(0, 0) = t2.at<float>(1, 0);
	tT2.at<float>(0, 0) = t2.at<float>(1, 0);
	tT2.at<float>(0, 1) = t2.at<float>(0, 2);
	tT2.at<float>(0, 2) = t2.at<float>(2, 1);
	Mat R1 = U * tR.t() * VT;
	Mat R2 = U * nR.t() * VT;
	cout << "R1:" << R1 << endl;
	cout << "R2:" << R2 << endl;
	//Mat testR1, testR2, testT1, testT2;
	//invert(t1, testT1);
	//invert(t2, testT2);
	//testR1 = testT1 * E;
	//testR2 = testT2 * E;
	////cout << rotationMatrixToEulerAngles(testR1) << rotationMatrixToEulerAngles(testR2);
	//cout << t2 * R2 << endl << E;
	if (testEtR(p_lc[0], p_rc[0], tT1, R1, CAMERA_MARTRIX_LC, CAMERA_MARTRIX_RC, R, T)) { return; }
	if (testEtR(p_lc[0], p_rc[0], tT1, R2, CAMERA_MARTRIX_LC, CAMERA_MARTRIX_RC, R, T)) { return; }
	if (testEtR(p_lc[0], p_rc[0], tT2, R1, CAMERA_MARTRIX_LC, CAMERA_MARTRIX_RC, R, T)) { return; }
	if (testEtR(p_lc[0], p_rc[0], tT2, R2, CAMERA_MARTRIX_LC, CAMERA_MARTRIX_RC, R, T)) { return; }
}
void FindTransformMat(vector<Point3f> real_world_marker, vector<Point3f> camera_world_marker, Mat &R, Mat &t) {


	int pointsNum = real_world_marker.size();
	Point3f real_mean(0, 0, 0);
	Point3f camera_mean(0, 0, 0);
	for (int i = 0; i < pointsNum; ++i) {
		real_mean += real_world_marker[i];
		camera_mean += camera_world_marker[i];
	}
	real_mean = real_mean / pointsNum;
	camera_mean = camera_mean / pointsNum;

	cv::Mat srcMat(3, pointsNum, CV_32FC1);
	cv::Mat dstMat(3, pointsNum, CV_32FC1);

	for (int i = 0; i < pointsNum; ++i)
	{
		srcMat.at<float>(0, i) = real_world_marker[i].x - real_mean.x;
		srcMat.at<float>(1, i) = real_world_marker[i].y - real_mean.y;
		srcMat.at<float>(2, i) = real_world_marker[i].z - real_mean.z;

		dstMat.at<float>(0, i) = camera_world_marker[i].x - camera_mean.x;
		dstMat.at<float>(1, i) = camera_world_marker[i].y - camera_mean.y;
		dstMat.at<float>(2, i) = camera_world_marker[i].z - camera_mean.z;
	}

	cv::Mat matS = srcMat * dstMat.t();

	cv::Mat matU, matW, matV;
	cv::SVDecomp(matS, matW, matU, matV);

	cv::Mat matTemp = matU * matV;

	double det = cv::determinant(matTemp);
	double datM[] = { 1, 0, 0, 0, 1, 0, 0, 0, det };
	cv::Mat matM = Mat::eye(3, 3, CV_32F);
	matM.at<float>(2, 2) = det;

	R = Mat::zeros(3, 3, CV_32F);
	t = Mat::zeros(3, 1, CV_32F);
	R = matV.t() * matM * matU.t();


	Matx33f rr = R;
	Point3f tt;
	tt = camera_mean - rr*real_mean;
	t.at<float>(0, 0) = tt.x;
	t.at<float>(1, 0) = tt.y;
	t.at<float>(2, 0) = tt.z;


	{
		//        计算误差
		double error = 0.0;
		for (int i = 0; i < pointsNum; ++i) {
			Point3f e_pt = camera_world_marker[i] - rr*real_world_marker[i] - tt;
			error += sqrt(e_pt.ddot(e_pt));
		}
		error /= pointsNum;
		//cout << "平均误差:" << error << endl;
	}
}