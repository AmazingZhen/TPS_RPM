#include <iostream>
#include <experimental/filesystem>

#include "rpm.h"
#include "data.h"

namespace fs = std::experimental::filesystem;

void delete_directory(string dir) {
	fs::path p(dir);
	if (fs::exists(p) && fs::is_directory(p))
	{
		fs::directory_iterator end;
		for (fs::directory_iterator it(p); it != end; ++it)
		{
			try
			{
				if (fs::is_regular_file(it->status()))
				{
					fs::remove(it->path());
				}
			}
			catch (const std::exception &ex)
			{
				ex;
			}
		}
	}
}

int main() {
	delete_directory("res/");
	//getchar();

	//int data_point_num = 100;
	////cout << "Enter data_point_num : ";
	////cin >> data_point_num;
	//double data_range_min = 0.0, data_range_max = 1000.0;
	//double data_noise_mu = 0.0, data_noise_sigma = 10.0;
	cout << "Enter scale : ";
	cin >> rpm::scale;
	getchar();

	//MatrixXd X = data_generate::generate_random_points(data_point_num, data_range_min, data_range_max);
	//MatrixXd offset(X.rows(), X.cols());
	//offset.col(0).setConstant(10.0);
	//offset.col(1).setConstant(10.0);
	//MatrixXd Y = data_generate::add_gaussian_noise(X, data_noise_mu, data_noise_sigma) + offset;
	//MatrixXd Y = X + offset;
	 
	MatrixXd X = data_generate::read_from_file("data/fish_source.txt");
	
	//MatrixXd offset = MatrixXd::Zero(X.rows(), X.cols());
	//offset.col(0).setConstant(10.0);
	//offset.col(1).setConstant(10.0);
	//MatrixXd Y = X + offset;
	MatrixXd Y = data_generate::read_from_file("data/fish_target.txt");

	/*double scale = 50;
	cout << "Enter scale :";
	cin >> scale;
	data_generate::preprocess(X, Y, scale);*/

	//getchar();
	//getchar();

	Mat origin_image = data_visualize::visualize(X, Y, rpm::scale);
	imwrite("data_origin.png", origin_image);
	data_generate::add_outlier(Y, 10);
	Mat origin_image_outlier = data_visualize::visualize(X, Y, rpm::scale);
	imwrite("data_origin_outlier.png", origin_image_outlier);
	//getchar();

	//MatrixXd Y = X;
	//MatrixXd M = MatrixXd::Identity(data_point_num, data_point_num);

	//double lambda = 0.1;

	rpm::ThinPLateSplineParams params(X);
	MatrixXd M;
	rpm::estimate(X, Y, M, params);

	//MatrixXd diff = (params.applyTransform() - Y).cwiseAbs();
	//cout << "XT - Y" << endl;
	//cout << diff << endl;
	//cout << diff.maxCoeff() << endl;
	//rpm::estimate_transform(X, Y, M, lambda, params);

	Mat result_image = data_visualize::visualize(params.applyTransform(false), Y, rpm::scale);
	imwrite("data_result.png", result_image);

	getchar();
	//getchar();
	//getchar();

	return 0;
}