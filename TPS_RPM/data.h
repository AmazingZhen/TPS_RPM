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

#include "rpm.h"

using namespace Eigen;
using namespace cv;
using namespace rpm;

using std::cin;
using std::cout;
using std::endl;
using std::string;

namespace data_process {
	void sample(MatrixXd& X, int sample_num);
	void remove_rows(MatrixXd& X, int start_row, int end_row);

	// (x,y) -> (x,y,1)
	void homo(MatrixXd& X);
	// (x,y,w) -> (x/w, y/w)
	void hnorm(MatrixXd& X);

	// Normalize X and Y to range [0, 1].
	// Return a 3*3 matrix represent the transform.
	Matrix3d preprocess(MatrixXd& X, MatrixXd& Y);

	void apply_transform(MatrixXd& X, const Matrix3d& trans);

	void apply_transform(Vector2d& X, const Matrix3d& trans);
}

namespace data_generate {
	MatrixXd generate_random_points(const int point_num, const double range_min, const double range_max);
	MatrixXd add_gaussian_noise(const MatrixXd& X, const double mu, const double sigma);
	bool load(MatrixXd& X, const string &filename);
	void save(const MatrixXd& X, const string& filename);
	void add_outlier(MatrixXd& X, const int num);
}

namespace data_visualize {
	extern string res_dir;
	extern bool save_intermediate_result;

	Mat visualize(const MatrixXd& X, const MatrixXd& Y, const bool draw_line = false);
	void visualize(const string& file_name, const MatrixXd& X, const MatrixXd& Y, const bool draw_line = false);
	void visualize_origin(const const string& file_name, const MatrixXd& X, const MatrixXd& Y,
		const MatrixXd X_outlier, const MatrixXd& Y_outlier, const int grid_step = 20);

	void visualize_result(const string& file_name, const MatrixXd X_outlier, const MatrixXd& Y_outlier, const rpm::ThinPlateSplineParams& params, const int grid_step = 20);

	// create_directory(data_visualize::res_dir); 
	void create_directory();
	// clean_directory(data_visualize::res_dir); 
	void clean_directory();
}