// This file is for data generating.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "data.h"
#include "rpm.h"

#include <iostream>
#include <fstream>

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

MatrixXd data_generate::read_from_file(const string & filename)
{
	std::ifstream f(filename);
	if (!f.is_open()) {
		throw std::runtime_error("can not open file : " + filename);
	}
	cout << "Opened : " << filename << endl;

	vector<Vector2d> points;
	while (!f.eof()) {
		Vector2d p;
		f >> p.x() >> p.y();
		points.push_back(p);
	}
	f.close();
	//cout << points.size() << endl;

	MatrixXd X(points.size(), 2);
#pragma omp parallel for
	for (int i = 0; i < points.size(); i++) {
		X.row(i) = points[i];
	}

	//cout << X << endl;

	return X;
}

bool data_generate::preprocess(MatrixXd& X, MatrixXd& Y, const double scale)
{
	double min_x = std::min(X.col(0).minCoeff(), X.col(0).minCoeff());
	double max_x = std::max(X.col(0).maxCoeff(), X.col(0).maxCoeff());
	double min_y = std::min(X.col(1).minCoeff(), Y.col(1).minCoeff());
	double max_y = std::max(X.col(1).maxCoeff(), Y.col(1).maxCoeff());

	auto normalize_mat = [](MatrixXd& m, double min_x, double max_x, double min_y, double max_y) {
		MatrixXd t = m;
		t.col(0).setConstant(min_x);
		t.col(1).setConstant(min_y);

		m -= t;
		m.col(0) *= (1.0 / (max_x - min_x));
		m.col(1) *= (1.0 / (max_y - min_y));
	};

	normalize_mat(X, min_x, max_x, min_y, max_y);
	normalize_mat(Y, min_x, max_x, min_y, max_y);

	X *= scale;
	Y *= scale;

	return true;
}

Mat data_visualize::visualize(const MatrixXd& X, const MatrixXd& Y, const bool draw_line)
{
	if (X.cols() != Y.cols() || X.cols() != rpm::D) {
		throw std::invalid_argument("Only support 2d points now!");
	}

	const double min_x = std::min(X.col(0).minCoeff(), Y.col(0).minCoeff());
	const double max_x = std::max(X.col(0).maxCoeff(), Y.col(0).maxCoeff());
	const double min_y = std::min(X.col(1).minCoeff(), Y.col(1).minCoeff());
	const double max_y = std::max(X.col(1).maxCoeff(), Y.col(1).maxCoeff());

	const int image_height = ceil(max_y - min_y), image_width = ceil(max_x - min_x);

	const int radius_x = 5, radius_y = 1;
	const int thickness_x = 2, thickness_y = -1;
	const int lineType = 8;
	const cv::Scalar color_x(255, 0, 0), color_y(0, 0, 255);

	//std::cout << "X" << std::endl;
	//std::cout << X << std::endl;
	//std::cout << "Y" << std::endl;
	//std::cout << Y << std::endl;

	Mat img(image_height, image_width, CV_8UC3);
	img = cv::Scalar(0, 0, 0);

	if (draw_line && X.rows() == Y.rows()) {
		for (int i = 0; i < X.rows(); i++) {
			const Vector2d& x = X.row(i), &y = Y.row(i);

			cv::line(img,
				cv::Point2f(x.x() - min_x, image_height - 1 - (x.y() - min_y)),
				cv::Point2f(y.x() - min_x, image_height - 1 - (y.y() - min_y)),
				cv::Scalar(255, 255, 255));
		}
	}

	for (int i = 0; i < X.rows(); i++) {
		const Vector2d& x = X.row(i);
		circle(img,
			cv::Point2f(x.x() - min_x, image_height - 1 - (x.y() - min_y)),
			radius_x,
			color_x,
			thickness_x,
			lineType);
	}

	for (int i = 0; i < Y.rows(); i++) {
		const Vector2d& y = Y.row(i);
		circle(img,
			cv::Point2f(y.x() - min_x, image_height - 1 - (y.y() - min_y)),
			radius_y,
			color_y,
			thickness_y,
			lineType);
	}

	return img;
}
