#include "Farm.hpp"
using namespace Eigen;

inline static double cosd(double deg) { return std::cos(deg * M_PI_LOCAL / 180.0); }
inline static double sind(double deg) { return std::sin(deg * M_PI_LOCAL / 180.0); }

//参数说明：
// x_d: std::vector<double>，相对位置（rotate_x-x_sorted）
// turbine: Turbine对象，包含风机参数
// idx: std::vector<int>，逻辑索引（0/1）(1x159, 表示哪些点命中compact)
// y_model: std::vector<std::vector<VectorXd>>，偏转模型输出，多项式拟合结果
VectorXd deflection_function(
    const VectorXd& x_d,
    const Turbine& turbine,
    const std::vector<int>& idx,
    const std::vector<VectorXd>& y_model
) {
    // 拟合版本
    double rotor_D = turbine.rotor_diameter;
	int n = x_d.size();

    // 用Eigen VectorXd作为中间量
    VectorXd y_c = VectorXd::Zero(n);

    int index;
    if (rotor_D - 228 == 0) 
        index = 0;
    else if (rotor_D - 158 == 0) 
        index = 1;
    else
        index = 2;


    // 预计算因子
    double factor = turbine.getCt() * std::pow(cosd(turbine.yaw_angle), 2) * sind(turbine.yaw_angle);

    // idx转Eigen类型
    std::vector<int> idx_vec = idx;
    bool any_idx = false;
	for (int i : idx_vec) if (i) { any_idx = true; break; } // 检查是否有任何命中点，有则为true
	if (any_idx) {
		// y_model[0][index] 是 std::vector<double>
		//y_c = factor * y_model[index]; // for all indexes in idx_vec, otherwise set to 0
		// extract all non-zero idx from idx_vec
		std::vector<int> non_zero_idx;
		for (int i = 0; i < n; i++) {
			if (idx_vec[i] != 0)
				non_zero_idx.push_back(i);
		}
		y_c(non_zero_idx) = factor * y_model[index];
    }

	return y_c;
}
