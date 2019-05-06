// This file is for data generating.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <iostream>
#include <random>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace Eigen;
using namespace cv;

using std::cin;
using std::cout;
using std::endl;
using std::string;

namespace data_generate {
	MatrixXd generate_random_points(const int point_num, const double range_min, const double range_max);
	MatrixXd add_gaussian_noise(const MatrixXd& X, const double mu, const double sigma);
	MatrixXd read_from_file(const string& filename);
	void add_outlier(MatrixXd& X, const int num);

	// Normalize x and y range to [0, 1] then scale it.
	bool preprocess(MatrixXd& X, MatrixXd& Y);
}

namespace data_visualize {
	Mat visualize(const MatrixXd& X, const MatrixXd& Y, const double scale, const bool draw_line = true);
}