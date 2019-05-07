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

	cout << "Enter scale : ";
	cin >> rpm::scale;
	getchar();
	
	MatrixXd X = data_generate::read_from_file("data/fish2_source.txt");
	MatrixXd Y = data_generate::read_from_file("data/fish2_target.txt");

	Mat origin_image = data_visualize::visualize(X, Y, rpm::scale);
	imwrite("data_origin.png", origin_image);
	data_generate::add_outlier(X, 0.3);
	Mat origin_image_outlier = data_visualize::visualize(X, Y, rpm::scale);
	imwrite("data_origin_outlier.png", origin_image_outlier);
	getchar();

	rpm::ThinPlateSplineParams params(X);
	MatrixXd M;
	rpm::estimate(X, Y, M, params);

	Mat result_image = data_visualize::visualize(params.applyTransform(false), Y, rpm::scale);
	imwrite("data_result.png", result_image);

	getchar();
	//getchar();
	//getchar();

	return 0;
}