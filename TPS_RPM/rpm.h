// This file is for the thin-plate spline 2d point matching, a specific form of non-rigid transformation.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Dense>

using namespace Eigen;

namespace rpm {
	const static int D = 2;

	class ThinPLateSplineParams {
	public:
		ThinPLateSplineParams(const MatrixXd &X) {
			this->X = X;

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
		}

		// (D + 1) * (D + 1) matrix representing the affine transformation.
		MatrixXd d;
		// K * (D + 1) matrix representing the non-affine deformation.
		MatrixXd w;

		MatrixXd applyTransform() const;
		VectorXd applyTransform(int x_i) const;

	private:
		MatrixXd X;

		// K * K matrix
		MatrixXd phi;
	};

	// Compute the thin-plate spline params and 2d point correspondence from two point sets.
	//
	// Input:
	//   X, Y		source and target points set.
	// Output:
	//	 M			correspondence between X and Y
	//	 params		thin-plate spline params
	// Returns true on success, false on failure
	//
	bool estimate(
		const MatrixXd& X,
		const MatrixXd& Y,
		MatrixXd& M,
		ThinPLateSplineParams& params
	);

	bool init_params(
		const MatrixXd& X,
		const MatrixXd& Y,
		const double T,
		MatrixXd& M,
		ThinPLateSplineParams& params
	);

	// Compute the thin-plate spline parameters from two point sets.
	//
	// Input:
	//   X, Y		source and target points set.
	//	 params		thin-plate spline params
	//	 T			temperature
	// Output:
	//	 M			correspondence between X and Y
	// Returns true on success, false on failure
	//
	bool estimate_correspondence(
		const MatrixXd& X,
		const MatrixXd& Y,
		const ThinPLateSplineParams& params,
		const double T,
		MatrixXd& M
	);

	// Compute the thin-plate spline parameters from two point sets.
	//
	// Input:
	//   X, Y		source and target points set.
	//	 M			correspondence between X and Y
	// Output:
	//	 params		thin-plate spline params
	// Returns true on success, false on failure
	//
	bool estimate_transform(
		const MatrixXd& X,
		const MatrixXd& Y,
		const MatrixXd& M,
		const double lambda,
		ThinPLateSplineParams& params
	);
}


