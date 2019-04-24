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

bool rpm::estimate_transform(
	const MatrixXd& X_,
	const MatrixXd& Y_,
	const MatrixXd& M,
	const double lambda,
	ThinPLateSplineParams& params)
{
	auto t1 = std::chrono::high_resolution_clock::now();

	try {
		if (X_.cols() != D || Y_.cols() != D) {
			throw std::invalid_argument("Current only support 2d point set !");
		}

		const int K = X_.rows(), N = Y_.rows();
		if (M.rows() != K || M.cols() != N) {
			throw std::invalid_argument("Matrix M size not same as X and Y!");
		}

		MatrixXd X = MatrixXd::Zero(K, D + 1);
#pragma omp parallel for
		for (int x_i = 0; x_i < K; x_i++) {
			Vector3d x(1.0, X_.row(x_i).x(), X_.row(x_i).y());
			X.row(x_i) = x;
		}

		MatrixXd Y = MatrixXd::Zero(N, D + 1);
#pragma omp parallel for
		for (int y_i = 0; y_i < N; y_i++) {
			Vector3d y(1.0, Y_.row(y_i).x(), Y_.row(y_i).y());
			Y.row(y_i) = y;
		}

		MatrixXd X_target = MatrixXd::Zero(K, D + 1);
#pragma omp parallel for
		for (int x_i = 0; x_i < K; x_i++) {
			Vector3d x_target = Vector3d::Zero();
			for (int y_i = 0; y_i < N; y_i++) {
				x_target += M(x_i, y_i) * Y.row(y_i);
			}
			X_target.row(x_i) = x_target;
		}

		Y = X_target;

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

		/*std::cout << "phi size: " << phi.rows() << ", " << phi.cols() << std::endl;
		std::cout << phi << std::endl;*/

		ColPivHouseholderQR<MatrixXd> qr;
		qr.compute(X);
		if (qr.info() != Eigen::Success) {
			throw std::runtime_error("QR decomposition failed!");
		}

		MatrixXd Q = qr.householderQ();
		MatrixXd R_ = qr.matrixQR().triangularView<Upper>();

		/*std::cout << "Q size: " << Q.rows() << ", " << Q.cols() << std::endl;
		std::cout << Q << std::endl;

		std::cout << "R_ size: " << R_.rows() << ", " << R_.cols() << std::endl;
		std::cout << R_ << std::endl;*/

		MatrixXd Q1 = Q.block(0, 0, K, D + 1), Q2 = Q.block(0, D + 1, K, K - (D + 1));
		MatrixXd R = R_.block(0, 0, D + 1, D + 1);

		/*std::cout << "Q1 size: " << Q1.rows() << ", " << Q1.cols() << std::endl;
		std::cout << Q1 << std::endl;
		std::cout << "Q2 size: " << Q2.rows() << ", " << Q2.cols() << std::endl;
		std::cout << Q2 << std::endl;
		std::cout << "R size: " << R.rows() << ", " << R.cols() << std::endl;
		std::cout << R << std::endl;*/

		LDLT<MatrixXd> solver;
		MatrixXd L_mat = (Q2.transpose() * phi * Q2 + lambda * MatrixXd::Identity(K - (D + 1), K - (D + 1)));

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt decomposition failed!");
		}

		MatrixXd gamma(K - (D + 1), (D + 1));
#pragma omp parallel for
		for (int i = 0; i < D + 1; i++) {
			VectorXd b_vec = Q2.transpose() * Y.col((i + 1) % (D + 1));
			VectorXd res = solver.solve(L_mat.transpose() * b_vec);
			if (solver.info() != Eigen::Success) {
				throw std::runtime_error("ldlt solve failed!");
			}
			gamma.col(i) = res;
		}

		params.w = Q2 * gamma;
		/*std::cout << "w" << std::endl;
		std::cout << params.w << std::endl;

		getchar();*/

		L_mat = R;
		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt decomposition failed!");
		}

		params.d = MatrixXd::Zero(D + 1, D + 1);
#pragma omp parallel for
		for (int i = 0; i < D + 1; i++) {
			VectorXd b_vec = Q1.transpose() * (Y - phi * params.w).col((i + 1) % (D + 1));
			solver.compute(L_mat.transpose() * L_mat);
			if (solver.info() != Eigen::Success) {
				throw std::runtime_error("ldlt decomposition failed!");
			}
			params.d.col(i) = solver.solve(L_mat.transpose() * b_vec);
			if (solver.info() != Eigen::Success) {
				throw std::runtime_error("ldlt solve failed!");
			}
		}

		/*std::cout << "d" << std::endl;
		std::cout << params.d << std::endl;

		getchar();

		std::cout << "X" << std::endl;
		std::cout << X << std::endl;
		std::cout << "Y" << std::endl;
		std::cout << Y << std::endl;

		MatrixXd T = X * params.d + phi * params.w;
		std::cout << "T" << std::endl;
		std::cout << T << std::endl;*/

		MatrixXd diff = (Y - X * params.d + phi * params.w).cwiseAbs();

		/*std::cout << "diff" << std::endl;
		std::cout << diff << std::endl;*/

		//std::cout << "diff.size() : " << diff.rows() << ", " << diff.cols() << std::endl;
		std::cout << "diff.maxCoeff() : " << diff.maxCoeff() << std::endl;

	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;

		return false;
	}

	auto t2 = std::chrono::high_resolution_clock::now();

	auto span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
	std::cout << "Thin-plate spline params estimating time: " << span.count() << " seconds.\n";

	return true;
}
