#include <iostream>

#include "rpm.h"
#include "data.h"

int main() {
	const string data_dir = "data/";
	const string source_suffix = "_source.txt", target_suffix = "_target.txt";
	const string outlier_suffix = "_outlier";
	string file_name = "fish2";

	cout << "Enter file_name : ";
	cin >> file_name;

	//rpm::scale = 500;
	//cout << "Enter scale : ";
	//cin >> rpm::scale;
	//getchar();

	int sample_num = 200;

	const bool need_outlier = true;
	const int outlier_num = 1;

	MatrixXd X_origin, Y_origin;
	data_generate::load(X_origin, data_dir + file_name + source_suffix);
	data_generate::load(Y_origin, data_dir + file_name + target_suffix);
	MatrixXd X = X_origin, Y = Y_origin;

	if (need_outlier) {
		if (!data_generate::load(X, data_dir + file_name + outlier_suffix + source_suffix)) {
			data_generate::add_outlier(X, outlier_num);
			data_generate::save(X, data_dir + file_name + outlier_suffix + source_suffix);
		}

		if (!data_generate::load(Y, data_dir + file_name + outlier_suffix + target_suffix)) {
			data_generate::add_outlier(Y, outlier_num);
			data_generate::save(Y, data_dir + file_name + outlier_suffix + target_suffix);
		}
	}

	//data_process::remove_rows(X, 10, 25);
	//data_process::remove_rows(Y, 50, 65);
	//data_process::sample(X, sample_num);
	//data_process::sample(Y, sample_num);
	cout << X << endl;
	cout << "Num of X : " << X.rows() << endl;
	cout << Y << endl;
	cout << "Num of Y : " << Y.rows() << endl;
	MatrixXd X_norm = X, Y_norm = Y;
	Matrix3d preprocess_trans = data_process::preprocess(X_norm, Y_norm);
	Matrix3d preprocess_trans_inv = preprocess_trans.inverse();

	data_visualize::res_dir = file_name;
	data_visualize::create_directory();
	data_visualize::clean_directory();

	data_visualize::visualize_origin("data_origin.png", X_origin, Y_origin, X, Y);

	vector<pair<int, int> > matched_point_indices;
	matched_point_indices.push_back({ 0, 0 });
	matched_point_indices.push_back({ 1, 1 });
	matched_point_indices.push_back({ 2, 2 });
	matched_point_indices.push_back({ 3, 3 });

	rpm::ThinPlateSplineParams params(X_norm);
	MatrixXd M;
	if (rpm::estimate(X, Y, M, params, matched_point_indices)) {
		//Mat result_image = data_visualize::visualize(params.applyTransform(false), Y, 1);
		//sprintf_s(file_buf, "%s/data_result.png", data_generate::res_dir.c_str());
		//imwrite(file_buf, result_image);
	}

	cout << "M" << endl;
	cout << M << endl;

	data_visualize::visualize_result("data_result.png", X, Y, params);


	//for (auto point_pair : matched_point_indices) {
	//	int k = point_pair.first, n = point_pair.second;
	//	if (k < 0 || k >= M.rows() || n < 0 || n >= M.cols()) {
	//		continue;
	//	}

	//	printf("M(%d, %d) : %.2f\n", k, n, M(k, n));
	//}

	const int width = 500, height = 500;
	Mat src_img(width, height, CV_8UC3), dst_img(width, height, CV_8UC3);
	src_img = cv::Scalar(230, 230, 230);
	dst_img = cv::Scalar(230, 230, 230);

	const int radius = 4, radius_grid = 1;
	const int thickness = -1;
	const int lineType = 4;
	const cv::Scalar color(0, 0, 0), color_grid(120, 120, 120);

	const int grid_step = 20;

	for (int y = grid_step; y < height; y += grid_step) {
		for (int x = grid_step; x < width; x += grid_step) {
			Vector2d coord(x, y);

			circle(src_img,
				cv::Point2f(coord.x(), coord.y()),
				radius_grid,
				color_grid,
				thickness);

			// Apply preprocess transform.
			data_process::apply_transform(coord, preprocess_trans);
			// Apply tps transform.
			Vector2d target_coord = params.applyTransform(coord, true);
			// Inverse preprocess transform.
			data_process::apply_transform(target_coord, preprocess_trans_inv);
			//cout << target_coord << endl;

			if (target_coord.x() < 0 || target_coord.x() >= width || target_coord.y() < 0 || target_coord.y() >= height) {
				continue;
			}

			circle(dst_img,
				cv::Point2f(target_coord.x(), target_coord.y()),
				radius_grid,
				color_grid,
				thickness);
		}
	}

	for (int i = 0; i < X.rows(); i++) {
		const Vector2d& x = X.row(i);
		circle(src_img,
			cv::Point2f(x.x(), x.y()),
			radius,
			color,
			thickness);

		Vector2d y = x;
		data_process::apply_transform(y, preprocess_trans);
		// Apply tps transform.
		Vector2d target_coord = params.applyTransform(y, true);
		// Inverse preprocess transform.
		data_process::apply_transform(target_coord, preprocess_trans_inv);

		cout << target_coord << endl;

		circle(dst_img,
			cv::Point2f(target_coord.x(), target_coord.y()),
			radius,
			color,
			thickness);
	}

	imwrite("1.png", src_img);
	imwrite("2.png", dst_img);

	//getchar();
	//getchar();
	//getchar();

	return 0;
}