#include "Toolset.hpp"
using namespace Eigen;

// 3d coord rotation (Rot x Pos' form)
MatrixXd rot_func(Eigen::MatrixXd& pos, Eigen::VectorXd& center, double angle) {
	angle = angle * M_PI_LOCAL / 180; // deg to rad
	RowVectorXd center_row = center.transpose();
	MatrixXd pos_offset = pos.rowwise() - center_row;
	MatrixXd RotMat(3, 3);
	RotMat << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1.0;
	MatrixXd rot_pos_center = (RotMat * pos_offset.transpose()).transpose();
	MatrixXd rot_pos = rot_pos_center.rowwise() + center_row;
	return rot_pos;
}