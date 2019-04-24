#include <iostream>
#include <random>

#include "rpm.h"

MatrixXd generate_random_points(int point_num) {
	std::random_device rd;
	std::mt19937 mt;
	std::uniform_real_distribution<double> dist(0.0, 100.0);

	MatrixXd X(point_num, 2);
	for (int i = 0; i < point_num; i++) {
		Vector2d x(dist(mt), dist(mt));
		X.row(i) = x;
	}

	return X;
}

int main() {
	int point_num = 50;

	MatrixXd X = generate_random_points(point_num);
	MatrixXd Y = X;
	MatrixXd M = MatrixXd::Identity(point_num, point_num);

	double lambda = 0.1;

	rpm::ThinPLateSplineParams params;
	rpm::estimate_transform(X, Y, M, lambda, params);

	getchar();

	return 0;
}