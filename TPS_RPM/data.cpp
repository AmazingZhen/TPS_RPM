// This file is for data generating.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "data.h"

#include <iostream>
#include <fstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

string data_visualize::res_dir = "res_rpm";
bool data_visualize::save_intermediate_result = true;

MatrixXd data_generate::generate_random_points(const int point_num, const double range_min, const double range_max) {
	std::default_random_engine gen;
	std::uniform_real_distribution<double> dist(range_min, range_max);
	auto random = bind(dist, gen);

	MatrixXd X(point_num, rpm::D);
#pragma omp parallel for
	for (int i = 0; i < point_num; i++) {
		for (int d = 0; d < rpm::D; d++) {
			X.row(i)[d] = random();
		}
	}

	return X;
}

MatrixXd data_generate::add_gaussian_noise(const MatrixXd& X, const double mu, const double sigma) {
	std::default_random_engine gen;
	std::normal_distribution<double> dist(mu, sigma);
	auto random = bind(dist, gen);

	const int point_num = X.rows();
	MatrixXd Y = X;
#pragma omp parallel for
	for (int i = 0; i < point_num; i++) {
		for (int d = 0; d < rpm::D; d++) {
			Y.row(i)[d] += random();
		}
	}

	return Y;
}

bool data_generate::load(MatrixXd& X, const string &filename)
{
	try {
		std::ifstream f(filename);
		if (!f.is_open()) {
			throw std::runtime_error("can not open file : " + filename);
		}
		cout << "Read : " << filename << endl;

		vector<Vector2d> points;
		while (!f.eof()) {
			Vector2d p;
			f >> p.x() >> p.y();
			points.push_back(p);
		}
		f.close();
		//cout << points.size() << endl;

		X = MatrixXd(points.size(), 2);
#pragma omp parallel for
		for (int i = 0; i < points.size(); i++) {
			X.row(i) = points[i];
		}

		return true;
	}
	catch (std::exception& e) {
		cout << e.what() << endl;
		return false;
	}
}

void data_generate::save(const MatrixXd& X, const string& filename)
{
	std::ofstream f(filename);
	if (!f.is_open()) {
		throw std::runtime_error("can not open file : " + filename);
	}
	cout << "Save : " << filename << endl;

	for (int i = 0; i < X.rows(); i++) {
		const Vector2d& p = X.row(i);
		f << p.x() << " " << p.y();
		if (i != X.rows() - 1) {
			f << endl;
		}
	}
	f.close();
}

void data_generate::add_outlier(MatrixXd& X, const int num)
{
	if (X.cols() != rpm::D) {
		return;
	}

	double min_x = X.col(0).minCoeff();
	double max_x = X.col(0).maxCoeff();
	double min_y = X.col(1).minCoeff();
	double max_y = X.col(1).maxCoeff();

	std::random_device rd;
	std::default_random_engine gen_x(rd()), gen_y(rd());
	std::uniform_real_distribution<double> dist_x(min_x, max_x), dist_y(min_y, max_y);
	auto random_x = bind(dist_x, gen_x), random_y = bind(dist_y, gen_y);

	MatrixXd X_(X.rows() + num, X.cols());
	for (int x_i = 0; x_i < X.rows(); x_i++) {
		X_.row(x_i) = X.row(x_i);
	}

	for (int x_i = 0; x_i < num; x_i++) {
		Vector2d random_point(random_x(), random_y());
		X_.row(X.rows() + x_i) = random_point;
		//cout << x_i << endl;
		//cout << random_point << endl;
	}

	X = X_;
}

Mat data_visualize::visualize(const MatrixXd& X_, const MatrixXd& Y_, const bool draw_line)
{
	if (X_.cols() != rpm::D && X_.cols() != rpm::D + 1 && Y_.cols() != rpm::D && Y_.cols() != rpm::D + 1) {
		throw std::invalid_argument("Only support 2d points now!");
	}

	MatrixXd X = X_;
	MatrixXd Y = Y_;

	data_process::hnorm(X);
	data_process::hnorm(Y);

	const double min_x = std::min(X.col(0).minCoeff(), Y.col(0).minCoeff());
	const double max_x = std::max(X.col(0).maxCoeff(), Y.col(0).maxCoeff());
	const double min_y = std::min(X.col(1).minCoeff(), Y.col(1).minCoeff());
	const double max_y = std::max(X.col(1).maxCoeff(), Y.col(1).maxCoeff());

	const int padding = 20;

	const int image_height = ceil(max_y - min_y + padding * 2), image_width = ceil(max_x - min_x + padding * 2);

	const int radius_x = 5, radius_y = 1;
	const int thickness_x = 2, thickness_y = -1;
	const int lineType = 8;
	const cv::Scalar color_x(255, 0, 0), color_y(0, 0, 255);

	//std::cout << "X" << std::endl;
	//std::cout << X << std::endl;
	//std::cout << "Y" << std::endl;
	//std::cout << Y << std::endl;

	Mat img(image_height, image_width, CV_8UC3);
	img = cv::Scalar(255, 255, 255);

	if (draw_line) {
		for (int i = 0; i < min(X.rows(), Y.rows()); i++) {
			const Vector2d& x = X.row(i), &y = Y.row(i);

			cv::line(img,
				cv::Point2f(x.x() - min_x + padding, image_height - 1 - (x.y() - min_y) + padding),
				cv::Point2f(y.x() - min_x + padding, image_height - 1 - (y.y() - min_y) + padding),
				cv::Scalar(255, 255, 255));
		}
	}

	for (int i = 0; i < X.rows(); i++) {
		const Vector2d& x = X.row(i);
		circle(img,
			cv::Point2f(x.x() - min_x + padding, image_height - 1 - (x.y() - min_y) + padding),
			radius_x,
			color_x,
			thickness_x,
			lineType);
	}

	for (int i = 0; i < Y.rows(); i++) {
		const Vector2d& y = Y.row(i);
		circle(img,
			cv::Point2f(y.x() - min_x + padding, image_height - 1 - (y.y() - min_y) + padding),
			radius_y,
			color_y,
			thickness_y,
			lineType);
	}

	return img;
}

void data_visualize::visualize(const string& file_name, const MatrixXd& X, const MatrixXd& Y, const bool draw_line)
{
	static char file_buf[256];

	Mat image = data_visualize::visualize(X, Y, draw_line);
	sprintf_s(file_buf, "%s/%s", res_dir.c_str(), file_name.c_str());
	imwrite(file_buf, image);

	printf("Saved : %s\n", file_buf);
}

void data_visualize::visualize_origin(const const string & file_name, const MatrixXd & X, const MatrixXd & Y,
	const MatrixXd X_outlier, const MatrixXd& Y_outlier, const int grid_step)
{
	const cv::Scalar color_background(200, 200, 200);
	const cv::Scalar color_grid_point(120, 120, 120);

	const int radius_x = 7, radius_y = 11;
	const int thickness_x = -1, thickness_y = -1;
	const cv::Scalar color_x(72, 71, 235), color_y(176, 137, 35);

	const int radius = 4, radius_grid = 3;
	const int thickness = -1;
	const cv::Scalar color(0, 0, 0);
	
	const double min_x = std::min(X_outlier.col(0).minCoeff(), Y_outlier.col(0).minCoeff());
	const double max_x = std::max(X_outlier.col(0).maxCoeff(), Y_outlier.col(0).maxCoeff());
	const double min_y = std::min(X_outlier.col(1).minCoeff(), Y_outlier.col(1).minCoeff());
	const double max_y = std::max(X_outlier.col(1).maxCoeff(), Y_outlier.col(1).maxCoeff());

	const int height = ceil((max_y - min_y) / grid_step + 2) * grid_step, width = ceil((max_x - min_x) / grid_step + 2) * grid_step;
	Mat img_origin(height, width, CV_8UC3);
	img_origin = color_background;

	// Draw grid points

	for (int y = grid_step; y < height; y += grid_step) {
		for (int x = grid_step; x < width; x += grid_step) {
			Vector2d coord(x, y);

			circle(img_origin,
				cv::Point2f(coord.x(), coord.y()),
				radius_grid,
				color_grid_point,
				thickness);
		}
	}

	// Draw source points

	for (int i = 0; i < X_outlier.rows(); i++) {
		const Vector2d& x = X_outlier.row(i);
		circle(img_origin,
			cv::Point2f(x.x() - min_x + grid_step, x.y() - min_y + grid_step),
			radius_x,
			color_x,
			thickness_x);
	}

	// Draw target points

	for (int i = 0; i < Y_outlier.rows(); i++) {
		const Vector2d& y = Y_outlier.row(i);
		circle(img_origin,
			cv::Point2f(y.x() - min_x + grid_step, y.y() - min_y + grid_step),
			radius_y,
			color_y,
			thickness_y);
	}

	// Draw lines

	for (int i = 0; i < min(X.rows(), Y.rows()); i++) {
		const Vector2d& x = X.row(i), &y = Y.row(i);

		cv::line(img_origin,
			cv::Point2f(x.x() - min_x + grid_step, x.y() - min_y + grid_step),
			cv::Point2f(y.x() - min_x + grid_step, y.y() - min_y + grid_step),
			cv::Scalar(0, 0, 0),
			3);
	}

	imwrite(file_name, img_origin);
}

void data_visualize::visualize_result(const string & file_name, const MatrixXd X_outlier, const MatrixXd & Y_outlier, const rpm::ThinPlateSplineParams& params, const int grid_step)
{
	const cv::Scalar color_background(200, 200, 200);
	const cv::Scalar color_grid_point(120, 120, 120);

	const int radius_x = 7, radius_y = 11;
	const int thickness_x = -1, thickness_y = -1;
	const cv::Scalar color_x(72, 71, 235), color_y(176, 137, 35);

	const int radius = 4, radius_grid = 3;
	const int thickness = -1;
	const cv::Scalar color(0, 0, 0);

	const double min_x = std::min(X_outlier.col(0).minCoeff(), Y_outlier.col(0).minCoeff());
	const double max_x = std::max(X_outlier.col(0).maxCoeff(), Y_outlier.col(0).maxCoeff());
	const double min_y = std::min(X_outlier.col(1).minCoeff(), Y_outlier.col(1).minCoeff());
	const double max_y = std::max(X_outlier.col(1).maxCoeff(), Y_outlier.col(1).maxCoeff());

	const int height = ceil((max_y - min_y) / grid_step + 2) * grid_step, width = ceil((max_x - min_x) / grid_step + 2) * grid_step;
	Mat img_result(height, width, CV_8UC3);
	img_result = color_background;

	MatrixXd X_norm = X_outlier, Y_norm = Y_outlier;
	Matrix3d preprocess_trans = data_process::preprocess(X_norm, Y_norm);
	Matrix3d preprocess_trans_inv = preprocess_trans.inverse();

	// Draw grid points

	for (int y = grid_step; y < height; y += grid_step) {
		for (int x = grid_step; x < width; x += grid_step) {
			Vector2d coord(x + min_x - grid_step, y + min_y - grid_step);

			// Apply preprocess transform.
			data_process::apply_transform(coord, preprocess_trans);
			// Apply tps transform.
			Vector2d target_coord = params.applyTransform(coord, true);
			// Inverse preprocess transform.
			data_process::apply_transform(target_coord, preprocess_trans_inv);
			//cout << target_coord << endl;

			target_coord.x() = target_coord.x() - min_x + grid_step;
			target_coord.y() = target_coord.y() - min_y + grid_step;

			if (target_coord.x() < 0 || target_coord.x() >= width || target_coord.y() < 0 || target_coord.y() >= height) {
				continue;
			}

			circle(img_result,
				cv::Point2f(target_coord.x(), target_coord.y()),
				radius_grid,
				color_grid_point,
				thickness);
		}
	}

	// Draw target points

	for (int i = 0; i < Y_outlier.rows(); i++) {
		const Vector2d& y = Y_outlier.row(i);

		circle(img_result,
			cv::Point2f(y.x() - min_x + grid_step, y.y() - min_y + grid_step),
			radius_y,
			color_y,
			thickness_y);
	}

	// Draw source points after transform

	for (int i = 0; i < X_outlier.rows(); i++) {
		const Vector2d& x = X_outlier.row(i);

		Vector2d x_ = x;
		data_process::apply_transform(x_, preprocess_trans);
		// Apply tps transform.
		Vector2d target_coord = params.applyTransform(x_, true);
		// Inverse preprocess transform.
		data_process::apply_transform(target_coord, preprocess_trans_inv);

		target_coord.x() = target_coord.x() - min_x + grid_step;
		target_coord.y() = target_coord.y() - min_y + grid_step;

		if (target_coord.x() < 0 || target_coord.x() >= width || target_coord.y() < 0 || target_coord.y() >= height) {
			continue;
		}

		circle(img_result,
			cv::Point2f(target_coord.x(), target_coord.y()),
			radius_x,
			color_x,
			thickness_x);
	}

	imwrite(file_name, img_result);
}

void data_visualize::create_directory()
{
	fs::create_directory(res_dir);
}

void data_visualize::clean_directory()
{
	fs::path p(res_dir);
	if (fs::exists(p) && fs::is_directory(p))
	{
		fs::directory_iterator end;
		for (fs::directory_iterator it(p); it != end; ++it)
		{
			try
			{
				if (fs::is_regular_file(it->status()))
				{
					fs::remove(it->path());
				}
			}
			catch (const std::exception &ex)
			{
				ex;
			}
		}
	}
}

void data_process::sample(MatrixXd &X, int sample_num)
{
	if (X.rows() < sample_num) {
		return;
	}

	int interval = ceil(X.rows() / (double)sample_num);
	MatrixXd X_(sample_num, X.cols());

	int count = 0;
	for (int x_i = 0; x_i < X.rows(); x_i += interval) {
		X_.row(count) = X.row(x_i);
		count++;
	}
	X_.conservativeResize(count, X_.cols());
	X = X_;
}

void data_process::remove_rows(MatrixXd& X, int start, int end)
{
	if (start < 0 || end >= X.rows()) {
		return;
	}

	MatrixXd X_ = X;
	int count = 0;
	for (int i = 0; i < start; i++) {
		X_.row(count) = X.row(i);
		count++;
	}
	for (int i = end + 1; i < X.rows(); i++) {
		X_.row(count) = X.row(i);
		count++;
	}
	X_.conservativeResize(count, X_.cols());

	X = X_;
}

void data_process::homo(MatrixXd & X)
{
	if (X.cols() != rpm::D && X.cols() != rpm::D + 1) {
		throw invalid_argument("Can not convert 2d points to 3d homogeneous points.");
	}

	if (X.cols() == rpm::D + 1) {
		return;
	}

	X.conservativeResize(X.rows(), rpm::D + 1);
	X.col(rpm::D).setConstant(1);
}

void data_process::hnorm(MatrixXd & X)
{
	if (X.cols() != rpm::D && X.cols() != rpm::D + 1) {
		throw invalid_argument("Can not convert 2d points to 3d homogeneous points.");
	}

	if (X.cols() == rpm::D) {
		return;
	}

	MatrixXd X_ = X.rowwise().hnormalized();
	X = X_;
}

Matrix3d data_process::preprocess(MatrixXd& X, MatrixXd& Y)
{
	if (X.cols() != rpm::D || Y.cols() != rpm::D) {
		throw invalid_argument("data_process::preprocess only support 2d points!");
	}

	double min_x = std::min(X.col(0).minCoeff(), Y.col(0).minCoeff());
	double max_x = std::max(X.col(0).maxCoeff(), Y.col(0).maxCoeff());
	double min_y = std::min(X.col(1).minCoeff(), Y.col(1).minCoeff());
	double max_y = std::max(X.col(1).maxCoeff(), Y.col(1).maxCoeff());

	double max_len = max((max_x - min_x), (max_y - min_y));

	Matrix3d translate = Matrix3d::Identity();
	translate.col(2) = Vector3d(-min_x, -min_y, 1);
	Matrix3d scale = Matrix3d::Identity();
	scale(0, 0) = scale(1, 1) = 1.0 / max_len;

	Matrix3d transform = scale * translate;

	apply_transform(X, transform);
	apply_transform(Y, transform);

	return transform;
}

void data_process::apply_transform(MatrixXd& m, const Matrix3d & trans)
{
	if (m.cols() != rpm::D) {
		throw invalid_argument("data_process::apply_transform() only support 2d points!");
	}

	homo(m);
	m = m * trans.transpose();
	hnorm(m);
}

void data_process::apply_transform(Vector2d & X, const Matrix3d & trans)
{
	Vector3d X_ = X.homogeneous();
	X = (trans * X_).hnormalized();
}
