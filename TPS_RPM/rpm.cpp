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

namespace {
	bool _matrices_equal(
		const MatrixXd &m1,
		const MatrixXd &m2,
		const double tol = 0.005)
	{
		if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
			return false;
		}

		bool res = ((m1 - m2).cwiseAbs().maxCoeff() <= tol);
		//if (res) {
		//	printf("equal\n");
		//}

		return res;
	}

	void _soft_assign(
		MatrixXd& assignment_matrix,
		const int max_iteration = 30,
		const double epsilon = 1e-20)
	{
		MatrixXd assignment_matrix_old;

		int iter = 0;
		do {
			//printf("	Softassign iter : %d\n", iter);

			assignment_matrix_old = assignment_matrix;

			// normalizing across all rows
#pragma omp parallel for
			for (int r = 0; r < assignment_matrix.rows(); r++) {
				double row_sum = assignment_matrix.row(r).sum();
				if (row_sum < epsilon) {
					continue;
				}
				for (int c = 0; c < assignment_matrix.cols(); c++) {
					assignment_matrix(r, c) /= row_sum;
				}
			}

			// normalizing across all cols
#pragma omp parallel for
			for (int c = 0; c < assignment_matrix.cols(); c++) {
				double col_sum = assignment_matrix.col(c).sum();
				if (col_sum < epsilon) {
					continue;
				}
				for (int r = 0; r < assignment_matrix.rows(); r++) {
					assignment_matrix(r, c) /= col_sum;
				}
			}
		} while (!_matrices_equal(assignment_matrix_old, assignment_matrix, epsilon) && iter++ < max_iteration);

		//printf("	Softassign iter : %d\n", iter);
	}
}

bool rpm::estimate(
	const MatrixXd& X,
	const MatrixXd& Y,
	MatrixXd& M,
	ThinPLateSplineParams& params)
{
	auto t1 = std::chrono::high_resolution_clock::now();

	try {
		// Annealing params
		const double T_start = 1.0 / 0.00091, T_end = 1.0 / 0.2, r = 0.95, I0 = 5, epsilon0 = 0.05;
		// Softassign params
		const double I1 = 30, epsilon1 = 0.005;
		// Thin-plate spline params
		const double lambda = 0.01;

		double T_cur = T_start;

		MatrixXd M_prev;

		if (!init_params(X, Y, T_start, M, params)) {
			throw std::runtime_error("init params failed!");
		}

		while (T_cur >= T_end) {

			//printf("T : %.2f\n\n", T_cur);

			int iter = 0;
			MatrixXd M_prev;

			do {
				//printf("	Annealing iter : %d\n", iter);

				if (!estimate_correspondence(X, Y, params, T_cur, M)) {
					throw std::runtime_error("estimate correspondence failed!");
				}
				//getchar();

				if (!estimate_transform(X, Y, M, lambda, params)) {
					throw std::runtime_error("estimate transform failed!");
				}
				//getchar();

			} while (!_matrices_equal(M_prev, M, epsilon0) && iter++ < I0);

			T_cur *= r;

			//char file[256];
			//sprintf_s(file, "res/data_%.2f.png", T_cur);
			//Mat result_image = data_visualize::visualize(params.applyTransform(), Y);
			//imwrite(file, result_image);
			//cout << endl << endl;
			//getchar();
		}
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
	ThinPLateSplineParams& params)
{
	const int K = X.rows(), N = Y.rows();

	const double beta = 1.0 / T;
	M = Eigen::MatrixXd::Zero(K, N);
#pragma omp parallel for
	for (int k = 0; k < K; k++) {
		const VectorXd& x = X.row(k);
		for (int n = 0; n < N; n++) {
			const VectorXd& y = Y.row(n);

			//assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
			double dist = ((y - x).squaredNorm());

			//assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
			M(k, n) = std::exp(beta * -dist);
		}
	};

	_soft_assign(M);
	//cout << "M" << endl;
	//cout << M << endl;

	return estimate_transform(X, Y, M, 0.01, params);
}

bool rpm::estimate_correspondence(
	const MatrixXd& X,
	const MatrixXd& Y,
	const ThinPLateSplineParams& params,
	const double T,
	MatrixXd& M)
{
	const int K = X.rows(), N = Y.rows();
	const double beta = 1.0 / T;

	M = MatrixXd::Zero(K, N);

	MatrixXd XT = params.applyTransform();
	//MatrixXd dist = MatrixXd::Zero(K, N);
	//for (int k = 0; k < K; k++) {
	//	const VectorXd& x = XT.row(k);
	//	for (int n = 0; n < N; n++) {
	//		const VectorXd& y = Y.row(n);
	//		
	//		dist(k, n) = ((y - x).squaredNorm());
	//	}
	//}
	//cout << "dist" << endl;
	//cout << dist << endl;

#pragma omp parallel for
	for (int k = 0; k < K; k++) {
		const VectorXd& x = XT.row(k);
		for (int n = 0; n < N; n++) {
			const VectorXd& y = Y.row(n);

			//assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
			double dist = ((y - x).squaredNorm());

			//assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
			M(k, n) = std::exp(beta * -dist);
		}
	};
	//cout << "M before soft assign" << endl;
	//cout << M << endl;
	_soft_assign(M);
	//cout << "M" << endl;
	//cout << M << endl;

	return true;
}

bool rpm::estimate_transform(
	const MatrixXd& X_,
	const MatrixXd& Y_,
	const MatrixXd& M,
	const double lambda,
	ThinPLateSplineParams& params)
{
	//auto t1 = std::chrono::high_resolution_clock::now();

	try {
		if (X_.cols() != D || Y_.cols() != D) {
			throw std::invalid_argument("Current only support 2d point set !");
		}

		const int K = X_.rows(), N = Y_.rows();
		if (M.rows() != K || M.cols() != N) {
			throw std::invalid_argument("Matrix M size not same as X and Y!");
		}

		int dim = D;
		MatrixXd X, Y;
		if (USE_HOMO) {
			dim = D + 1;

			X = MatrixXd(K, dim);
			Y = MatrixXd(K, dim);
			for (int k = 0; k < K; k++) {
				X.row(k) = X_.row(k).homogeneous();
				Y.row(k) = (M.row(k) * Y_).homogeneous();
			}
		}
		else {
			X = X_;
			Y = M * Y_;
		}

		MatrixXd phi = MatrixXd::Zero(K, K);  // phi(a, b) = || Xb - Xa || ^ 2 * log(|| Xb - Xa ||);
#pragma omp parallel for
		for (int a_i = 0; a_i < K; a_i++) {
			VectorXd a = X.row(a_i);

			for (int b_i = 0; b_i < K; b_i++) {
				if (b_i == a_i) {
					continue;
				}

				VectorXd b = X.row(b_i);

				phi(a_i, b_i) = ((b - a).squaredNorm() * log((b - a).norm()));
			}
		}

		//std::cout << "phi size: " << phi.rows() << ", " << phi.cols() << std::endl;
		//std::cout << phi << std::endl;

		HouseholderQR<MatrixXd> qr;
		qr.compute(X);
		//if (qr.info() != Eigen::Success) {
		//	throw std::runtime_error("QR decomposition failed!");
		//}

		MatrixXd Q = qr.householderQ();
		MatrixXd R_ = qr.matrixQR().triangularView<Upper>();

		//std::cout << "QR" << std::endl;
		//std::cout << Q * R_ << std::endl;
		//std::cout << "X" << std::endl;
		//std::cout << X << std::endl;
		//getchar();

		MatrixXd Q1 = Q.block(0, 0, K, dim), Q2 = Q.block(0, dim, K, K - dim);
		MatrixXd R = R_.block(0, 0, dim, dim);

		//std::cout << "Q1 size: " << Q1.rows() << ", " << Q1.cols() << std::endl;
		//std::cout << Q1 << std::endl;
		//std::cout << "Q2 size: " << Q2.rows() << ", " << Q2.cols() << std::endl;
		//std::cout << Q2 << std::endl;
		//std::cout << "R size: " << R.rows() << ", " << R.cols() << std::endl;
		//std::cout << R << std::endl;

		LDLT<MatrixXd> solver;
		MatrixXd L_mat = (Q2.transpose() * phi * Q2 + lambda * MatrixXd::Identity(K - dim, K - dim));

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt decomposition failed!");
		}

		MatrixXd b_mat = Q2.transpose() * Y;
		MatrixXd gamma = solver.solve(L_mat.transpose() * b_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt solve failed!");
		}

		params.w = Q2 * gamma;
		//std::cout << "w" << std::endl;
		//std::cout << params.w << std::endl;
		//getchar();

		L_mat = R;
		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt decomposition failed!");
		}

		b_mat = Q1.transpose() * (Y - phi * params.w);
		params.d = solver.solve(L_mat.transpose() * b_mat);

		//std::cout << "d" << std::endl;
		//std::cout << params.d << std::endl;
		//getchar();

		//std::cout << "X" << std::endl;
		//std::cout << X << std::endl;
		//std::cout << "Y" << std::endl;
		//std::cout << Y << std::endl;

		MatrixXd XT = params.applyTransform();
		//std::cout << "XT" << std::endl;
		//std::cout << XT << std::endl;
		//std::cout << "(M * Y_)" << std::endl;
		//std::cout << (M * Y_) << std::endl;

		//MatrixXd diff = ((M * Y_) - XT).cwiseAbs();

		//std::cout << "diff" << std::endl;
		//std::cout << diff << std::endl;

		//std::cout << "diff.size() : " << diff.rows() << ", " << diff.cols() << std::endl;
		//std::cout << "diff" << std::endl;
		//std::cout << diff << std::endl;
		//std::cout << "diff.maxCoeff() : " << diff.maxCoeff() << std::endl;

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

rpm::ThinPLateSplineParams::ThinPLateSplineParams(const MatrixXd & X)
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

			phi(a_i, b_i) = ((b - a).squaredNorm() * log((b - a).norm()));
		}
	}

	if (USE_HOMO) {
		const int K = X.rows();
		this->X = MatrixXd(K, D + 1);
		for (int k = 0; k < K; k++) {
			this->X.row(k) = X.row(k).homogeneous();
		}
	}
	else {
		this->X = X;
	}
}

MatrixXd rpm::ThinPLateSplineParams::applyTransform() const
{
	MatrixXd XT_ = X * d + phi * w;
	if (USE_HOMO) {
		MatrixXd XT(XT_.rows(), XT_.cols() - 1);
		
		for (int k = 0; k < XT_.rows(); k++) {
			XT.row(k) = XT_.row(k).hnormalized();
		}
		return XT;
	}

	return XT_;
}

VectorXd rpm::ThinPLateSplineParams::applyTransform(int x_i) const
{
	VectorXd xt = X.row(x_i) * d + phi.row(x_i) * w;

	return (USE_HOMO ? xt.hnormalized() : xt);
}
