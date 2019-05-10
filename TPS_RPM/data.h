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

namespace data_process {
	void sample(MatrixXd& X, int sample_num);

	// (x,y) -> (x,y,1)
	void homo(MatrixXd& X);
	// (x,y,w) -> (x/w, y/w)
	void hnorm(MatrixXd& X);

	// Normalize X and Y to range [0, 1].
	void preprocess(MatrixXd& X, MatrixXd& Y);
}

namespace data_generate {
	extern string res_dir;
	extern bool save_intermediate_result;

	MatrixXd generate_random_points(const int point_num, const double range_min, const double range_max);
	MatrixXd add_gaussian_noise(const MatrixXd& X, const double mu, const double sigma);
	bool load(MatrixXd& X, const string &filename);
	void save(const MatrixXd& X, const string& filename);
	void add_outlier(MatrixXd& X, const double factor = 0.3);
}

namespace data_visualize {
	Mat visualize(const MatrixXd& X, const MatrixXd& Y, const double scale, const bool draw_line = false);
}