#include <iostream>

#include "rpm.h"
#include "data.h"

int main() {
	int data_point_num = 10;
	cout << "Enter data_point_num : ";
	cin >> data_point_num;
	double data_range_min = 0.0, data_range_max = 1000.0;
	double data_noise_mu = 0.0, data_noise_sigma = 20.0;

	MatrixXd X = data_generate::generate_random_points(data_point_num, data_range_min, data_range_max);
	MatrixXd offset(X.rows(), X.cols());
	offset.col(0).setConstant(20.0);
	offset.col(1).setConstant(30.0);
	MatrixXd Y = data_generate::add_gaussian_noise(X, data_noise_mu, data_noise_sigma) + offset;
	//MatrixXd Y = X + offset;

	Mat origin_image = data_visualize::visualize(X, Y);
	imwrite("data_origin.png", origin_image);

	//MatrixXd Y = X;
	//MatrixXd M = MatrixXd::Identity(data_point_num, data_point_num);

	//double lambda = 0.1;

	rpm::ThinPLateSplineParams params(X);
	MatrixXd M;
	rpm::estimate(X, Y, M, params);

	cout << "XT - Y" << endl;
	cout << params.applyTransform() - Y << endl;
	//rpm::estimate_transform(X, Y, M, lambda, params);

	Mat result_image = data_visualize::visualize(params.applyTransform(), Y);
	imwrite("data_result.png", result_image);

	getchar();
	getchar();

	return 0;
}