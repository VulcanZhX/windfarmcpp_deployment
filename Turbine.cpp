#include "Turbine.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
using namespace Eigen;


// 构造函数实现
Turbine::Turbine(
	const std::vector<std::vector<double>>& PT,
	const std::array<std::vector<int>, 3>& t_qz,
	const std::vector<double>& turbulence_sheer_veer,
	const std::vector<double>& turbine_diameter_vector,
	const std::vector<double>& turbine_hub_height_vector,
	const std::vector<double>& rated_power_vector,
	const std::vector<double>& life_total_vector,
	const std::vector<double>& repair_c_vector,
	// 机组寿命计算系数
	const std::vector<double>& fatigue,
	const std::vector<double>& fatigue_p,
	const std::vector<double>& serial_coeff_val,
	int status_val = 1, // binary status: 1-active, 0-inactive(converted from 0-34 to 0,1)
	int count_tn = 1 // 机组编号，从1开始
) {
	// 机组类型判定
	if (std::find(t_qz[0].begin(), t_qz[0].end(), count_tn) != t_qz[0].end()) {
		rotor_diameter = turbine_diameter_vector[0];
		rotor_radius = rotor_diameter / 2.0;
		hub_height = turbine_hub_height_vector[0];
		rated_power = rated_power_vector[0] * 1e6;
		life_total = life_total_vector[0];
		repair_c = repair_c_vector[0];
	}
	else if (std::find(t_qz[1].begin(), t_qz[1].end(), count_tn) != t_qz[1].end()) {
		rotor_diameter = turbine_diameter_vector[1];
		rotor_radius = rotor_diameter / 2.0;
		hub_height = turbine_hub_height_vector[1];
		rated_power = rated_power_vector[1] * 1e6;
		life_total = life_total_vector[1];
		repair_c = repair_c_vector[1];
	}
	else if (std::find(t_qz[2].begin(), t_qz[2].end(), count_tn) != t_qz[2].end()) {
		rotor_diameter = turbine_diameter_vector[2];
		rotor_radius = rotor_diameter / 2.0;
		hub_height = turbine_hub_height_vector[2];
		rated_power = rated_power_vector[2] * 1e6;
		life_total = life_total_vector[2];
		repair_c = repair_c_vector[2];
	}
	else {
		throw std::invalid_argument("count_tn not found in t_qz");
	}
	turbine_idx_org = count_tn;
	power_thrust_table_11.clear();
	power_thrust_table_68.clear();
	power_thrust_table_83.clear();

	/*std::vector<std::vector<double>> PT_cpy = PT;

	MatrixXd PT_eigen = Map<MatrixXd>(PT_cpy.data(), PT_cpy.size());*/
	for (const auto& row : PT) {
		// 取第1列到第3列
		power_thrust_table_11.push_back(std::vector<double>(row.begin(), row.begin() + 3));
		// 取第4列到第6列
		power_thrust_table_68.push_back(std::vector<double>(row.begin() + 3, row.begin() + 6));
		// 取第7列到第9列
		power_thrust_table_83.push_back(std::vector<double>(row.begin() + 6, row.begin() + 9));
	}
	turbulence_ambient = turbulence_sheer_veer[0];
	turbulence = turbulence_sheer_veer[0];
	past_comprehensive_fatigue_coefficient = fatigue[count_tn-1];
	annual_average_power = rated_power * 0.7;
	life_work_coeff = 1.0 / (rated_power * life_total * (1 + repair_c));
	life_turbulence_coeff = dis_coefficient / (fatigue_p[1] * life_total * (1 + repair_c));
	velocities_u.reserve(3); //预分配空间
	// additional params
	serial_coeff = serial_coeff_val;
	status = status_val;
	updateRadius();
	updateGrid();
}

// 线性插值工具
static double linearInterp(const std::vector<double>& x, const std::vector<double>& y, double xi) {
	if (xi < x.front() || xi > x.back()) return 0.0;
	auto it = std::lower_bound(x.begin(), x.end(), xi);
	if (it == x.begin()) return y.front();
	if (it == x.end()) return y.back();
	size_t idx = std::distance(x.begin(), it);
	double x0 = x[idx - 1], x1 = x[idx];
	double y0 = y[idx - 1], y1 = y[idx];
	return y0 + (y1 - y0) * (xi - x0) / (x1 - x0);
}

// 功率系数查表
double Turbine::fCp(double at_wind_speed) const {
	const std::vector<std::vector<double>>* table = nullptr;
	if (abs(rated_power - 11e6) < 1) table = &power_thrust_table_11;
	else if (abs(rated_power - 6.8e6) < 1) table = &power_thrust_table_68;
	else if (abs(rated_power - 8.3e6) < 1) table = &power_thrust_table_83;
	else return 0.0;

	std::vector<double> wind_speed_column, cp_column;
	for (const auto& row : *table) {
		wind_speed_column.push_back(row[0]);
		cp_column.push_back(row[1]);
	}
	return linearInterp(wind_speed_column, cp_column, at_wind_speed);
}

// 推力系数查表
double Turbine::fCt(double at_wind_speed) const {
	const std::vector<std::vector<double>>* table = nullptr;
	if (abs(rated_power - 11e6) < 1) table = &power_thrust_table_11;
	else if (abs(rated_power - 6.8e6) < 1) table = &power_thrust_table_68;
	else if (abs(rated_power - 8.3e6) < 1) table = &power_thrust_table_83;
	else return 0.0;

	std::vector<double> wind_speed_column, ct_column;
	wind_speed_column.reserve(power_thrust_table_68.size());
	ct_column.reserve(power_thrust_table_68.size());
	for (const auto& row : *table) {
		wind_speed_column.push_back(row[0]);
		ct_column.push_back(row[2]);
	}
	if (at_wind_speed < wind_speed_column.front()) return 0.99;
	if (at_wind_speed > wind_speed_column.back()) return 0.0001;
	return linearInterp(wind_speed_column, ct_column, at_wind_speed);
}

// 计算风机网格点的风速（其实就是风机点处风速...)
// local_wind_speed 格式：1x159(1 grid points per turbine for all turbines, to be fixed)
std::vector<double> Turbine::calculateTurbineVelocities(const std::vector<double>& local_wind_speed, int nt) const {
	std::vector<double> data(grid[2], 0.0);
	// 这里只是示意，实际应根据 grid 的定义和 local_wind_speed 的结构实现
	for (size_t i = 0; i < grid[2]; ++i) {
		int idx = static_cast<int>(grid[2]);
		data[i] = local_wind_speed[nt-1]; // 需根据实际数据结构调整
	}
	return data;
}

// 更新风速
void Turbine::updateVelocities(const std::vector<double>& u_wake,
	const double u_initial,
	int nt) {
	// 拷贝
	std::vector<double> u_wake_cpy = u_wake; // copy wake data
	VectorXd u_wake_eigen = Map<VectorXd>(u_wake_cpy.data(), u_wake_cpy.size()); //map wake data
	VectorXd local_wind_speed_u_eigen = u_initial - u_wake_eigen.array();
	std::vector<double>local_wind_speed_u(local_wind_speed_u_eigen.data(), local_wind_speed_u_eigen.data() + local_wind_speed_u_eigen.size());
	velocities_u = calculateTurbineVelocities(local_wind_speed_u, nt);
}

//// 更新湍流强度
void Turbine::updateTurbulenceIntensity(const std::vector<double>& u_turbulence_wake, int nt) {
	std::vector<double> data = calculateTurbineVelocities(u_turbulence_wake, nt);
	// calculate the norm of data
	VectorXd data_eigen = Map<VectorXd>(data.data(), data.size());
	double norm_data = data_eigen.norm();
	turbulence = std::sqrt(turbulence_ambient*turbulence_ambient + norm_data*norm_data);
}

// 重新定义转子半径
void Turbine::setRotorRadius(double r) {
	rotor_radius = r;
	updateGrid();
}

// 重新定义转子直径
void Turbine::setRotorDiameter(double d) {
	rotor_diameter = d;
	updateRadius();
}

// 更新半径
void Turbine::updateRadius() {
	rotor_radius = rotor_diameter / 2.0;
}
// 
// 更新网格点, 1 per turbine
void Turbine::updateGrid() {
	grid.clear();
	double xs = rotor_radius;
	grid = std::vector<double>(3, 0);
	grid[0] = xs;
	grid[2] = 1;
}

// 平均风速
double Turbine::getAverageVelocity() const {
	double sum = 0.0;
	for (double v : velocities_u) sum += std::pow(v, 3);
	return std::pow(sum / velocities_u.size(), 1.0 / 3.0);
}

// Cp
double Turbine::getCp() const {
	return fCp(getAverageVelocity());
}

// Ct
double Turbine::getCt() const {
	return fCt(getAverageVelocity());
}

// 功率
double Turbine::getPower() const {
	double yaw_effective_velocity = getAverageVelocity();
	double cptmp = getCp();
	double efficiency_product = serial_coeff[0] * serial_coeff[1] * serial_coeff[2];
	return 0.5 * air_density * M_PI_LOCAL * std::pow(rotor_radius, 2) * generator_efficiency *
		std::pow(yaw_effective_velocity, 3) * cptmp * 
		std::pow(std::cos(yaw_angle * M_PI_LOCAL / 180.0), 3) * efficiency_product;
}

// 综合疲劳系数
double Turbine::getComprehensiveFatigueCoefficient() const {
	double accumulated_c = consumed_comprehensive_fatigue_coefficient;
	double optimized_c_power = getPower() * optimization_period / (rated_power * life_total * (1 + repair_c));
	double optimized_c_turbulence = 1.5 * turbulence * optimization_period / (ref_turbulence * life_total * (1 + repair_c));
	double optimized_c = optimized_c_power + optimized_c_turbulence;
	return optimized_c + accumulated_c;
}

// 单机目标
double Turbine::getSingleTurbineObjective() const {
	return getPower() * optimization_period - annual_average_power * (getComprehensiveFatigueCoefficient() - past_comprehensive_fatigue_coefficient) * optimization_period;
}

// 单机发电量
double Turbine::getSingleTurbineGeneration() const {
	return getPower() * optimization_period;
}
