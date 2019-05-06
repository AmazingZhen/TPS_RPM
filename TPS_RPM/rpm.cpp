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
double rpm::T_start = 1e5;
double rpm::T_end = T_start * 1e-4;
double rpm::r = 0.93, rpm::I0 = 5, rpm::epsilon0 = 1e-2;
double rpm::alpha = 0.0; // 5 * 5
// Softassign params
double rpm::I1 = 30, rpm::epsilon1 = 1e-3;
// Thin-plate spline params
double rpm::lambda_start = T_start;

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

		bool res = ((m1 - m2).cwiseAbs().maxCoeff() <= tol);
		//if (res) {
		//	printf("equal\n");
		//}

		return res;
	}

	inline void _soft_assign(
		MatrixXd& assignment_matrix,
		const int max_iteration = 30,
		const double epsilon = 1e-3)
	{
		MatrixXd assignment_matrix_old;

		int iter = 0;
		while (iter++ < max_iteration){
			//printf("	Softassign iter : %d\n", iter);

			// normalizing across all rows
#pragma omp parallel for
			for (int r = 0; r < assignment_matrix.rows() - 1; r++) {
				double row_sum = assignment_matrix.row(r).sum();
				if (row_sum < 1e-5) {
					continue;
				}
				assignment_matrix.row(r) /= row_sum;
			}

			// normalizing across all cols
#pragma omp parallel for
			for (int c = 0; c < assignment_matrix.cols() - 1; c++) {
				double col_sum = assignment_matrix.col(c).sum();
				if (col_sum < 1e-5) {
					continue;
				}
				assignment_matrix.col(c) /= col_sum;
			}

			if (_matrices_equal(assignment_matrix_old, assignment_matrix, epsilon)) {
				break;
			}
		}

		//printf("	Softassign iter : %d\n", iter);
	}

	inline double _distance(const MatrixXd &Y_, const MatrixXd& M, const rpm::ThinPLateSplineParams& params) {
		MatrixXd Y = rpm::apply_correspondence(Y_, M);
		MatrixXd XT = params.applyTransform(true);

		if (XT.rows() != Y.rows() || XT.cols() != Y.cols()) {
			throw std::invalid_argument("X size not same as Y in _distance!");
		}

		MatrixXd diff = (Y - XT).cwiseAbs();
		return diff.maxCoeff();
	}
}

void rpm::set_T_start(double T)
{
	T_start = T;
	T_end = T * 1e-5;
	lambda_start = T;

	cout << "Set T_start : " << T_start << endl;
	//getchar();
}

bool rpm::estimate(
	const MatrixXd& X,
	const MatrixXd& Y,
	MatrixXd& M,
	ThinPLateSplineParams& params)
{
	auto t1 = std::chrono::high_resolution_clock::now();

	try {
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
		double lambda = lambda_start;// *T_cur;

		if (!init_params(X, Y, T_start, M, params)) {
			throw std::runtime_error("init params failed!");
		}

		//char file[256];
		//sprintf_s(file, "res/data_%.2f.png", T_cur);
		//Mat result_image = data_visualize::visualize(params.applyTransform(false), Y);
		//imwrite(file, result_image);

		while (T_cur >= T_end) {

			//printf("T : %.2f\n\n", T_cur);
			//printf("lambda : %.2f\n\n", lambda);

			int iter = 0;

			while (iter++ < I0) {
				//printf("	Annealing iter : %d\n", iter);
				MatrixXd M_prev = M;
				ThinPLateSplineParams params_prev = params;
				if (!estimate_correspondence(X, Y, params, T_cur, T_start, M)) {
					throw std::runtime_error("estimate correspondence failed!");
				}
				//getchar();

				if (!estimate_transform(X, Y, M, T_cur, lambda, params)) {
					throw std::runtime_error("estimate transform failed!");
				}
				//getchar();

				if (_matrices_equal(M_prev, M, epsilon0)) {  // hack!!!
					//if (T_cur < 50.0) {
					//	MatrixXd M_diff = (M_prev - M);
					//	double max_m_diff = M_diff.maxCoeff();

					//	cout << "M_diff : " << max_m_diff << endl;
					//	cout << "_distance:" << _distance(Y, M, params) << endl;
					//	cout << "_distance prev:" << _distance(Y, M_prev, params_prev) << endl;
					//}

					//if (_distance(Y, M, params) > _distance(Y, M_prev, params_prev)) {
					//	M = M_prev;
					//	params = params_prev;
					//}

					M = M_prev;
					params = params_prev;

					break;
				}
			}

			T_cur *= r;
			lambda *= r;
			//lambda = lambda_start * T_cur;

			//sprintf_s(file, "res/data_%.2f.png", T_cur);
			//Mat result_image = data_visualize::visualize(params.applyTransform(false), Y);
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
	const ThinPLateSplineParams& params,
	const double T,
	const double T0,
	MatrixXd& M)
{
	const int K = X.rows(), N = Y.rows();
	const double beta = 1.0 / T;

	M = MatrixXd::Zero(K + 1, N + 1);

	MatrixXd XT = params.applyTransform(false);
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
		const Vector2d& x = XT.row(k);
		for (int n = 0; n < N; n++) {
			const Vector2d& y = Y.row(n);

			//assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
			double dist = ((y - x).squaredNorm());

			//assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
			M(k, n) = beta * std::exp(beta *  -dist);
		}
	};

	Vector2d center_x(XT.col(0).mean(), XT.col(1).mean()), center_y(Y.col(0).mean(), Y.col(1).mean());

	const double beta_start = 1.0 / T0;
#pragma omp parallel for
	for (int k = 0; k < K; k++) {
		const Vector2d& x = XT.row(k);
		double dist = ((center_y - x).squaredNorm());
		M(k, N) = beta_start * std::exp(beta_start * -dist);
	}

#pragma omp parallel for
	for (int n = 0; n < N; n++) {
		const Vector2d& y = Y.row(n);
		double dist = ((y - center_x).squaredNorm());
		M(K, n) = beta_start * std::exp(beta_start * -dist);
	}
	
	//if (T < 50.0) {
	//	cout << "M.maxCoeff() before soft assign : " << M.maxCoeff() << endl;
	//}
	_soft_assign(M);
	//if (T < 50.0) {
	//	cout << "M.maxCoeff() after soft assign : " << M.maxCoeff() << endl;
	//}

	M.conservativeResize(K, N);

	return true;
}

bool rpm::estimate_transform(
	const MatrixXd& X_,
	const MatrixXd& Y_,
	const MatrixXd& M_,
	const double T,
	const double lambda,
	ThinPLateSplineParams& params)
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

		//std::cout << "phi size: " << phi.rows() << ", " << phi.cols() << std::endl;
		//std::cout << phi << std::endl;

		const MatrixXd& Q = params.get_Q();
		const MatrixXd& R_ = params.get_R();

		////std::cout << "QR" << std::endl;
		////std::cout << Q * R_ << std::endl;
		////std::cout << "X" << std::endl;
		////std::cout << X << std::endl;
		////getchar();

		MatrixXd Q1 = Q.block(0, 0, K, dim), Q2 = Q.block(0, dim, K, K - dim);
		MatrixXd R = R_.block(0, 0, dim, dim);

		//std::cout << "Q1 size: " << Q1.rows() << ", " << Q1.cols() << std::endl;
		//std::cout << Q1 << std::endl;
		//std::cout << "Q2 size: " << Q2.rows() << ", " << Q2.cols() << std::endl;
		//std::cout << Q2 << std::endl;
		//std::cout << "R size: " << R.rows() << ", " << R.cols() << std::endl;
		//std::cout << R << std::endl;

#ifdef USE_SVD_SOLVER
		BDCSVD<MatrixXd> solver;
		MatrixXd L_mat = (Q2.transpose() * phi * Q2 + MatrixXd::Constant(K - dim, K - dim, lambda * T));

		solver.compute(L_mat, ComputeThinU | ComputeThinV);

		MatrixXd b_mat = Q2.transpose() * Y;
		MatrixXd gamma = solver.solve(b_mat);

		params.w = Q2 * gamma;

		//std::cout << "w" << std::endl;
		//std::cout << params.w << std::endl;
		//getchar();

		// Add regular term lambdaI * d = lambdaI * I
		L_mat = MatrixXd(R.rows() * 2, R.cols());
		L_mat << R,
			MatrixXd::Constant(R.rows(), R.cols(), lambda * 0.01 * T);

		solver.compute(L_mat, ComputeThinU | ComputeThinV);

		b_mat = MatrixXd(R.rows() * 2, R.cols());
		b_mat << Q1.transpose() * (Y - phi * params.w),
			MatrixXd::Constant(R.rows(), R.cols(), lambda * 0.01 * T);

		params.d = solver.solve(b_mat);
#else
		LDLT<MatrixXd> solver;
		MatrixXd L_mat = (Q2.transpose() * phi * Q2 + (MatrixXd::Identity(K - dim, K - dim) * lambda));

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

		// Add regular term lambdaI * d = lambdaI * I
		//L_mat = MatrixXd(R.rows() * 2, R.cols());
		//L_mat << R,
		//	MatrixXd::Identity(R.rows(), R.cols()) * lambda * 0.01;
		L_mat = R;

		solver.compute(L_mat.transpose() * L_mat);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt decomposition failed!");
		}

		//b_mat = MatrixXd(R.rows() * 2, R.cols());
		//b_mat << Q1.transpose() * (Y - phi * params.w),
		//	MatrixXd::Identity(R.rows(), R.cols()) * lambda * 0.01;
		b_mat = Q1.transpose() * (Y - phi * params.w);

		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("ldlt solve failed!");
		}

		params.d = solver.solve(L_mat.transpose() * b_mat);

#endif // USE_SVD_SOLVER


		//{  // Assume d = I(3, 3)
		//	LDLT<MatrixXd> solver;
		//	MatrixXd L_mat = (phi + MatrixXd::Constant(K, K, lambda));
		//	solver.compute(L_mat.transpose() * L_mat);
		//	if (solver.info() != Eigen::Success) {
		//		throw std::runtime_error("ldlt decomposition failed!");
		//	}

		//	MatrixXd b_mat = Y - X;
		//	params.w = solver.solve(L_mat.transpose() * b_mat);
		//	if (solver.info() != Eigen::Success) {
		//		throw std::runtime_error("ldlt solve failed!");
		//	}
		// 
		// params.d = MatrixXd::Identity(D + 1, D + 1);
		//}

		//std::cout << "d" << std::endl;
		//std::cout << params.d << std::endl;
		//std::cout << "Estimated transform distance : " << _distance(Y_, M, params) << std::endl;
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
	
	return M * Y;
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

MatrixXd rpm::ThinPLateSplineParams::applyTransform(bool homo) const
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

VectorXd rpm::ThinPLateSplineParams::applyTransform(int x_i) const
{
	VectorXd xt = X.row(x_i) * d + phi.row(x_i) * w;

	return xt.hnormalized();
}
