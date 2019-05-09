// This file is for the thin-plate spline 2d point matching, a specific form of non-rigid transformation.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "rpm.h"

#include <iostream>
#include <chrono>

#include "data.h"

using std::cout;
using std::endl;
using namespace rpm;

// Annealing params
double rpm::T_start = 1;
double rpm::T_end = T_start * 1e-4;
double rpm::r = 0.93, rpm::I0 = 5, rpm::epsilon0 = 1e-2;
double rpm::alpha = 0.0; // 5 * 5
// Softassign params
double rpm::I1 = 10, rpm::epsilon1 = 1e-4;
// Thin-plate spline params
double rpm::lambda_start = T_start;

double rpm::scale = 300;

//#define USE_SVD_SOLVER

namespace {
	inline bool _matrices_equal(
		const MatrixXd &m1,
		const MatrixXd &m2,
		const double tol = 1e-3)
	{
		if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
			return false;
		}

		return ((m1 - m2).cwiseAbs().maxCoeff() <= tol);
	}

	inline void _soft_assign(
		MatrixXd& assignment_matrix,
		const int max_iteration = 30)
	{
		int iter = 0;
		while (iter++ < max_iteration){
			// normalizing across all rows
#pragma omp parallel for
			for (int r = 0; r < assignment_matrix.rows() - 1; r++) {
				double row_sum = assignment_matrix.row(r).sum();
				if (row_sum < epsilon1) {
					continue;
				}
				assignment_matrix.row(r) /= row_sum;
			}

			// normalizing across all cols
#pragma omp parallel for
			for (int c = 0; c < assignment_matrix.cols() - 1; c++) {
				double col_sum = assignment_matrix.col(c).sum();
				if (col_sum < epsilon1) {
					continue;
				}
				assignment_matrix.col(c) /= col_sum;
			}
		}
	}

	inline double _distance(const MatrixXd &Y_, const MatrixXd& M, const rpm::ThinPlateSplineParams& params) {
		MatrixXd Y = rpm::apply_correspondence(Y_, M);
		MatrixXd XT = params.applyTransform(true);

		if (XT.rows() != Y.rows() || XT.cols() != Y.cols()) {
			throw std::invalid_argument("X size not same as Y in _distance!");
		}

		MatrixXd diff = (Y - XT).cwiseAbs();
		return diff.maxCoeff();
	}

	// Normalize X and Y to range [0, 1].
	inline void _preprocess(MatrixXd& X, MatrixXd& Y) {
		double min_x = std::min(X.col(0).minCoeff(), Y.col(0).minCoeff());
		double max_x = std::max(X.col(0).maxCoeff(), Y.col(0).maxCoeff());
		double min_y = std::min(X.col(1).minCoeff(), Y.col(1).minCoeff());
		double max_y = std::max(X.col(1).maxCoeff(), Y.col(1).maxCoeff());

		double max_len = max((max_x - min_x), (max_y - min_y));

		auto normalize_mat = [](MatrixXd& m, double min_x, double min_y, double max_len) {
			MatrixXd t = m;
			t.col(0).setConstant(min_x);
			t.col(1).setConstant(min_y);

			m -= t;
			m /= max_len;
		};

		normalize_mat(X, min_x, min_y, max_len);
		normalize_mat(Y, min_x, min_y, max_len);

		return;
	}
}

void rpm::set_T_start(double T)
{
	T_start = T;
	T_end = T * 1e-4;
	lambda_start = T * 10;

	cout << "Set T_start : " << T_start << endl;
	//getchar();
}

bool rpm::estimate(
	const MatrixXd& X_,
	const MatrixXd& Y_,
	MatrixXd& M,
	ThinPlateSplineParams& params)
{
	auto t1 = std::chrono::high_resolution_clock::now();

	try {
		if (X_.cols() != rpm::D || Y_.cols() != rpm::D) {
			throw std::invalid_argument("rpm::estimate() only support 2d points!");
		}

		MatrixXd X = X_, Y = Y_;
		_preprocess(X, Y);

		params = ThinPlateSplineParams(X);

		double max_dist = 0, average_dist = 0;
		int K = X.rows(), N = Y.rows();
		for (int k = 0; k < K; k++) {
			const VectorXd& x = X.row(k);
			for (int n = 0; n < N; n++) {
				const VectorXd& y = Y.row(n);
				double dist = (y - x).squaredNorm();

				max_dist = max(max_dist, dist);
				average_dist += dist;
			}
		}
		average_dist /= (K * N);
		cout << "max_dist : " << max_dist << endl;
		cout << "average_dist : " << average_dist << endl;
		set_T_start(max_dist);

		//double T_end = T_start * 1e-5;

		double T_cur = T_start;
		double lambda = lambda_start;

		if (!init_params(X, Y, T_start, M, params)) {
			throw std::runtime_error("init params failed!");
		}

		char file[256];
		if (data_generate::save_intermediate_result) {
			sprintf_s(file, "%s/data_%.3f.png", data_generate::res_dir.c_str(), T_cur);
			Mat result_image = data_visualize::visualize(params.applyTransform(false), Y, scale);
			imwrite(file, result_image);
		}

		while (T_cur >= T_end) {

			//printf("T : %.2f\n\n", T_cur);
			//printf("lambda : %.2f\n\n", lambda);

			int iter = 0;

			while (iter++ < I0) {
				//printf("	Annealing iter : %d\n", iter);
				MatrixXd M_prev = M;
				ThinPlateSplineParams params_prev = params;
				if (!estimate_correspondence(X, Y, params, T_cur, T_start, M)) {
					throw std::runtime_error("estimate correspondence failed!");
				}

				if (!estimate_transform(X, Y, M, lambda, params)) {
					throw std::runtime_error("estimate transform failed!");
				}

				//if (_matrices_equal(M_prev, M, epsilon0)) {  // hack!!!
				//	//M = M_prev;
				//	//params = params_prev;
				//	break;
				//}
			}

			T_cur *= r;
			lambda *= r;

			if (data_generate::save_intermediate_result) {
				sprintf_s(file, "%s/data_%.3f.png", data_generate::res_dir.c_str(), T_cur);
				Mat result_image = data_visualize::visualize(params.applyTransform(false), Y, scale);
				imwrite(file, result_image);
			}
		}

		// Re-estimate real ThinPlateSplineParams on unnormalized data.

		//MatrixXd M_binary = MatrixXd::Zero(K, N);
		//for (int k = 0; k < K; k++) {
		//	Eigen::Index n;
		//	double max_coeff = M.row(k).maxCoeff(&n);
		//	if (max_coeff > 1.0 / N) {
		//		M_binary(k, n) = 1;
		//	}
		//}
		//M = M_binary;

		params = ThinPlateSplineParams(X_);
		estimate_transform(X_, Y_, M, lambda, params);
	}
	catch (const std::exception& e){
		std::cout << e.what();
		return false;
	}

	auto t2 = std::chrono::high_resolution_clock::now();

	auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	std::cout << "TPS-RPM estimate time: " << timespan.count() << " seconds.\n";

	return true;
}

bool rpm::init_params(
	const MatrixXd& X,
	const MatrixXd& Y,
	const double T,
	MatrixXd& M,
	ThinPlateSplineParams& params)
{
	const int K = X.rows(), N = Y.rows();

	//estimate_transform(X, X, MatrixXd::Identity(K + 1, K + 1), T, lambda, params);

	//estimate_correspondence(X, Y, params, T, T, M);

	//M = Eigen::MatrixXd::Identity(K + 1, N + 1);

//	const double beta = 1.0 / T;
//	M = Eigen::MatrixXd::Zero(K + 1, N + 1);
//#pragma omp parallel for
//	for (int k = 0; k < K; k++) {
//		const VectorXd& x = X.row(k);
//		for (int n = 0; n < N; n++) {
//			const VectorXd& y = Y.row(n);
//
//			//assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
//			double dist = ((y - x).squaredNorm());
//
//			//assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
//			M(k, n) = std::exp(beta * -dist);
//		}
//	};
//
//	_soft_assign(M);
	//cout << "M" << endl;
	//cout << M << endl;

	return true;
}

bool rpm::estimate_correspondence(
	const MatrixXd& X,
	const MatrixXd& Y,
	const ThinPlateSplineParams& params,
	const double T,
	const double T0,
	MatrixXd& M)
{
	const int K = X.rows(), N = Y.rows();
	const double beta = 1.0 / T;

	M = MatrixXd::Zero(K + 1, N + 1);

	MatrixXd XT = params.applyTransform(false);

#pragma omp parallel for
	for (int k = 0; k < K; k++) {
		const Vector2d& x = XT.row(k);
		for (int n = 0; n < N; n++) {
			const Vector2d& y = Y.row(n);

			//assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
			double dist = ((y - x).squaredNorm());

			//assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
			M(k, n) = beta * std::exp(beta *  -dist);
		}
	};

//	Vector2d center_x(XT.col(0).mean(), XT.col(1).mean()), center_y(Y.col(0).mean(), Y.col(1).mean());
//	const double beta_start = 1.0 / T0;
//#pragma omp parallel for
//	for (int k = 0; k < K; k++) {
//		const Vector2d& x = XT.row(k);
//		double dist = ((center_y - x).squaredNorm());
//		M(k, N) = beta_start * std::exp(beta_start * -dist);
//	}
//
//#pragma omp parallel for
//	for (int n = 0; n < N; n++) {
//		const Vector2d& y = Y.row(n);
//		double dist = ((y - center_x).squaredNorm());
//		M(K, n) = beta_start * std::exp(beta_start * -dist);
//	}

	M.row(K).setConstant(1.0 / (N + 1));
	M.col(N).setConstant(1.0 / (K + 1));
	
	_soft_assign(M);

	M.conservativeResize(K, N);

	return true;
}

bool rpm::estimate_transform(
	const MatrixXd& X_,
	const MatrixXd& Y_,
	const MatrixXd& M_,
	const double lambda,
	ThinPlateSplineParams& params)
{
	//auto t1 = std::chrono::high_resolution_clock::now();

	try {
		if (X_.cols() != D || Y_.cols() != D) {
			throw std::invalid_argument("Current only support 2d point set !");
		}

		const int K = X_.rows(), N = Y_.rows();
		if (M_.rows() != K || M_.cols() != N) {
			throw std::invalid_argument("Matrix M size not same as X and Y!");
		}

		MatrixXd M = M_.block(0, 0, K, N);

		int dim = D + 1;
		MatrixXd X(K, dim), Y = apply_correspondence(Y_, M);
		for (int k = 0; k < K; k++) {
			X.row(k) = X_.row(k).homogeneous();
		}
		
		const MatrixXd& phi = params.get_phi();
		const MatrixXd& Q = params.get_Q();
		const MatrixXd& R_ = params.get_R();

		MatrixXd Q1 = Q.block(0, 0, K, dim), Q2 = Q.block(0, dim, K, K - dim);
		MatrixXd R = R_.block(0, 0, dim, dim);

#ifdef RPM_USE_BOTHSIDE_OUTLIER_REJECTION
		MatrixXd W = MatrixXd::Zero(K, K);
		for (int k = 0; k < K; k++) {
			W(k, k) = 1.0 / std::max(M.row(k).sum(), epsilon1);
		}

		MatrixXd T = phi + N * lambda * W;

		LDLT<MatrixXd> solver;
		MatrixXd L_mat = Q2.transpose() * T * Q2;

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param w ldlt decomposition failed!");
		}

		MatrixXd b_mat = Q2.transpose() * Y;
		MatrixXd gamma = solver.solve(L_mat.transpose() * b_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param w ldlt solve failed!");
		}

		params.w = Q2 * gamma;


#ifdef RPM_REGULARIZE_AFFINE_PARAM  // Add regular term lambdaI * d = lambdaI * I
		double lambda_d = N * lambda * 0.01;

		L_mat = MatrixXd(R.rows() * 2, R.cols());
		L_mat << R,
			MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
		L_mat = R;
#endif // RPM_REGULARIZE_AFFINE_PARAM

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param d ldlt decomposition failed!");
		}

#ifdef RPM_REGULARIZE_AFFINE_PARAM
		b_mat = MatrixXd(R.rows() * 2, R.cols());
		b_mat << Q1.transpose() * (Y - T * params.w),
			MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
		b_mat = Q1.transpose() * (Y - K * params.w);
#endif // RPM_REGULARIZE_AFFINE_PARAM

		params.d = solver.solve(L_mat.transpose() * b_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param d ldlt solve failed!");
		}
#else
		LDLT<MatrixXd> solver;
		MatrixXd L_mat = (Q2.transpose() * phi * Q2 + (MatrixXd::Identity(K - dim, K - dim) * K * lambda));

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param w ldlt decomposition failed!");
		}

		MatrixXd b_mat = Q2.transpose() * Y;
		MatrixXd gamma = solver.solve(L_mat.transpose() * b_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param w ldlt solve failed!");
		}

		params.w = Q2 * gamma;


#ifdef RPM_REGULARIZE_AFFINE_PARAM  // Add regular term lambdaI * d = lambdaI * I
		double lambda_d = K * lambda * 0.01;

		L_mat = MatrixXd(R.rows() * 2, R.cols());
		L_mat << R,
			MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
		L_mat = R;
#endif // RPM_REGULARIZE_AFFINE_PARAM

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param d ldlt decomposition failed!");
		}

#ifdef RPM_REGULARIZE_AFFINE_PARAM
		b_mat = MatrixXd(R.rows() * 2, R.cols());
		b_mat << Q1.transpose() * (Y - phi * params.w),
			MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
		b_mat = Q1.transpose() * (Y - phi * params.w);
#endif // RPM_REGULARIZE_AFFINE_PARAM

		params.d = solver.solve(L_mat.transpose() * b_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Param d ldlt solve failed!");
		}

		// Another form of regularize d.
		//MatrixXd A = (R.transpose() * R + 0.01 * lambda * MatrixXd::Identity(dim, dim)).inverse()
		//	* (R.transpose() * ((Q1.transpose() * (Y - phi * params.w)) - R));
		//params.d = A + MatrixXd::Identity(dim, dim);

#endif // RPM_USE_BOTHSIDE_OUTLIER_REJECTION
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;

		return false;
	}

	//auto t2 = std::chrono::high_resolution_clock::now();

	//auto span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
	//std::cout << "Thin-plate spline params estimating time: " << span.count() << " seconds.\n";

	return true;
}

MatrixXd rpm::apply_correspondence(const MatrixXd& Y_, const MatrixXd& M)
{
	if (Y_.cols() != rpm::D) {
		throw std::invalid_argument("input must be 2d!");
	}

	MatrixXd Y(Y_.rows(), rpm::D + 1);
	Y.leftCols(rpm::D) = Y_;
	Y.rightCols(1).setConstant(1);
	
	MatrixXd MY = M * Y;
#ifdef RPM_USE_BOTHSIDE_OUTLIER_REJECTION
	for (int k = 0; k < M.rows(); k++) {
		MY.row(k) /= std::max(M.row(k).sum(), epsilon1);
	}
#endif // RPM_USE_BOTHSIDE_OUTLIER_REJECTION

	return MY;
}

rpm::ThinPlateSplineParams::ThinPlateSplineParams(const MatrixXd & X)
{
	const int K = X.rows();

	phi = MatrixXd::Zero(K, K);  // phi(a, b) = || Xb - Xa || ^ 2 * log(|| Xb - Xa ||);
#pragma omp parallel for
	for (int a_i = 0; a_i < K; a_i++) {
		VectorXd a = X.row(a_i);

		for (int b_i = 0; b_i < K; b_i++) {
			if (b_i == a_i) {
				continue;
			}

			VectorXd b = X.row(b_i);

			phi(b_i, a_i) = ((b - a).squaredNorm() * log((b - a).norm()));
		}
	}

	this->X = MatrixXd(K, D + 1);
	for (int k = 0; k < K; k++) {
		this->X.row(k) = X.row(k).homogeneous();
	}

	HouseholderQR<MatrixXd> qr;
	qr.compute(this->X);

	Q = qr.householderQ();
	R = qr.matrixQR().triangularView<Upper>();

	w = MatrixXd::Zero(X.rows(), rpm::D + 1);
	d = MatrixXd::Identity(rpm::D + 1, rpm::D + 1);
}

rpm::ThinPlateSplineParams::ThinPlateSplineParams(const ThinPlateSplineParams& other)
{
	d = other.d;
	w = other.w;
	X = other.X;
	phi = other.phi;
	Q = other.Q;
	R = other.R;
}

MatrixXd rpm::ThinPlateSplineParams::applyTransform(bool homo) const
{
	MatrixXd XT_ = X * d + phi * w;
	if (homo) {
		return XT_;
	}

	MatrixXd XT(XT_.rows(), XT_.cols() - 1);
	for (int k = 0; k < XT_.rows(); k++) {
		XT.row(k) = XT_.row(k).hnormalized();
	}
	return XT;
}

VectorXd rpm::ThinPlateSplineParams::applyTransform(int x_i) const
{
	VectorXd xt = X.row(x_i) * d + phi.row(x_i) * w;

	return xt.hnormalized();
}
