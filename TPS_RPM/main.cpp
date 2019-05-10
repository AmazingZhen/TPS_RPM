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
	const string data_dir = "data/";
	const string source_suffix = "_source.txt", target_suffix = "_target.txt";
	const string outlier_suffix = "_outlier";
	string file_name = "fish2";

	cout << "Enter file_name : ";
	cin >> file_name;

	rpm::scale = 500;
	cout << "Enter scale : ";
	cin >> rpm::scale;
	//getchar();

	int sample_num = 200;

	const bool need_generate_outlier = false;
	const bool source_load_outlier = false, target_load_outlier = false;

	MatrixXd X, Y;
	if (source_load_outlier) {
		if (!data_generate::load(X, data_dir + file_name + outlier_suffix + source_suffix)) {
			data_generate::load(X, data_dir + file_name + source_suffix);
			data_generate::add_outlier(X, 0.3);
			data_generate::save(X, data_dir + file_name + outlier_suffix + source_suffix);
		}
	}
	else {
		data_generate::load(X, data_dir + file_name + source_suffix);
	}

	if (target_load_outlier) {
		if (!data_generate::load(Y, data_dir + file_name + outlier_suffix + target_suffix)) {
			data_generate::load(Y, data_dir + file_name + target_suffix);
			data_generate::add_outlier(Y, 0.3);
			data_generate::save(Y, data_dir + file_name + outlier_suffix + target_suffix);
		}
	}
	else {
		data_generate::load(Y, data_dir + file_name + target_suffix);
	}


	//data_process::sample(X, sample_num);
	//data_process::sample(Y, sample_num);
	cout << "Num of X : " << X.rows() << endl;
	cout << "Num of Y : " << Y.rows() << endl;
	data_process::preprocess(X, Y);

	data_generate::res_dir = file_name;
	fs::create_directory(data_generate::res_dir);
	delete_directory(data_generate::res_dir);

	char file_buf[256];
	Mat origin_image = data_visualize::visualize(X, Y, rpm::scale);
	sprintf_s(file_buf, "%s/data_origin.png", data_generate::res_dir.c_str());
	imwrite(file_buf, origin_image);
	getchar();

	rpm::ThinPlateSplineParams params(X);
	MatrixXd M;
	if (rpm::estimate(X, Y, M, params)) {
		//Mat result_image = data_visualize::visualize(params.applyTransform(false), Y, 1);
		//sprintf_s(file_buf, "%s/data_result.png", data_generate::res_dir.c_str());
		//imwrite(file_buf, result_image);
	}

	getchar();
	//getchar();
	//getchar();

	return 0;
}