// This file is for data generating.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "data.h"

#include <iostream>

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

Mat data_visualize::visualize(const MatrixXd& X, const MatrixXd& Y)
{
	if (X.rows() != Y.rows() || X.cols() != Y.cols()) {
		throw std::invalid_argument("X size not same as Y size!");
	}

	if (X.cols() != rpm::D) {
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
	int point_num = X.rows();
	for (int i = 0; i < point_num; i++) {
		const Vector2d& x = X.row(i), &y = Y.row(i);

		circle(img,
			cv::Point2f(x.x() - min_x, image_height - 1 - (x.y() - min_y)),
			radius_x,
			color_x,
			thickness_x,
			lineType);

		circle(img,
			cv::Point2f(y.x() - min_x, image_height - 1 - (y.y() - min_y)),
			radius_y,
			color_y,
			thickness_y,
			lineType);
	}

	return img;
}
