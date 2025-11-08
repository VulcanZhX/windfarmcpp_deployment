#include "Farm.hpp"
#include <numeric>

using namespace Eigen;


inline int encase180(int angle) {
	angle = angle % 360;
	if (angle > 180) {
		angle -= 360;
	}
	else if (angle < -180) {
		angle += 360;
	}
	return angle;
}

WindFarmOptimization::WindFarmOptimization(
	const std::vector<std::vector<double>>& PT,
	const std::array<std::vector<int>, 3>& t_qz,
	const std::vector<double>& turbulence_sheer_veer,
	const std::vector<double>& turbine_diameter_vector,
	const std::vector<double>& turbine_hub_height_vector,
	const std::vector<double>& rated_power_vector,
	const std::vector<double>& life_total_vector,
	const std::vector<double>& repair_c_vector,
	const std::vector<double>& yaw_vec,
	const std::vector<double>& fatigue,
	const std::vector<double>& fatigue_p,
	const std::vector<std::vector<double>>& layout_farm,
	double wind_speed,
	double wind_direction
) {
	n_turbines = t_qz[0].size() + t_qz[1].size() + t_qz[2].size();
	Turbine turbine_c(PT, t_qz, turbulence_sheer_veer, turbine_diameter_vector, turbine_hub_height_vector, rated_power_vector, life_total_vector, repair_c_vector, fatigue, fatigue_p, 1);
	std::vector<Turbine> turbine_chart_input(n_turbines, turbine_c);

	// flatten layout_farm and store into Matrix layout... to be changed!
	std::vector<double> flat_layout;
	flat_layout.reserve(layout_farm.size() * layout_farm[0].size());
	for (const auto& row : layout_farm) {
		flat_layout.insert(flat_layout.end(), row.begin(), row.end());
	}
	MatrixXd layout_temp = Eigen::Map<Eigen::MatrixXd>(flat_layout.data(), 1, flat_layout.size());
	layout_temp.resize(layout_farm[0].size(), layout_farm.size());
	layout = layout_temp.transpose();

	int count_tn = 1; // label
	for (auto& elem : turbine_chart_input) {
		elem = Turbine(PT, t_qz, turbulence_sheer_veer, turbine_diameter_vector, turbine_hub_height_vector, rated_power_vector, life_total_vector, repair_c_vector, fatigue, fatigue_p, count_tn);
		elem.yaw_angle = yaw_vec[count_tn - 1];
		count_tn++;
	}
	turbine_chart = turbine_chart_input;

	//idx_org: 1:159 (Note: subscript: 0-158)
	idx_org.fill(1);
	std::iota(idx_org.begin(), idx_org.end(), 1);

	auto turbine_tuple = std::make_tuple(layout, turbine_chart, idx_org);
	turbulence_intensity = turbulence_sheer_veer[0];
	wind_shear = turbulence_sheer_veer[1];
	wind_veer = turbulence_sheer_veer[2];
	this->wind_speed = wind_speed;
	this->wind_direction = encase180(wind_direction - 270);  //Hint: default '0' is West(270°)
	qz_12 = t_qz[0];
	qz_3.reserve(t_qz[1].size() + t_qz[2].size());
	qz_3.insert(qz_3.end(), t_qz[1].begin(), t_qz[1].end());
	qz_3.insert(qz_3.end(), t_qz[2].begin(), t_qz[2].end());
}


//风场边界
Eigen::VectorXd WindFarmOptimization::getBounds() const {
	// x 和 y 列
	const auto& x_col = layout.col(0); // 所有涡轮机 x 坐标
	const auto& y_col = layout.col(1); // 所有涡轮机 y 坐标

	// 轮毂高度取第一台风机的第三列（MATLAB 的 layout(1,3)）
	double h = layout(0, 2);

	// 转子直径，假设 turbines 存储的是 shared_ptr<Turbine>
	double d = this->turbine_chart[0].rotor_diameter;

	// 计算边界
	double x_min = x_col.minCoeff() - d;
	double x_max = x_col.maxCoeff() + d;
	double y_min = y_col.minCoeff() - d;
	double y_max = y_col.maxCoeff() + d;
	double z_min = 0.1;
	double z_max = 1.5 * h;

	VectorXd bounds(6);
	bounds << x_min, x_max, y_min, y_max, z_min, z_max;
	return bounds;
}

// 获取所有风机网格点的x坐标
VectorXd WindFarmOptimization::getX() const {
	VectorXd x = this->layout.col(0);
	return x;
}

// 获取所有风机网格点的y坐标 
VectorXd WindFarmOptimization::getY() const {
	int nt = static_cast<int>(layout.rows());
	int num_points = 1;

	VectorXd y(num_points * nt);
	for (int i = 0; i < nt; ++i) {
		double R = this->turbine_chart[0].rotor_diameter;
		Eigen::VectorXd yi = Eigen::VectorXd::LinSpaced(num_points, -R, R);
		yi.array() += layout(i, 1); // translate to global y
		y.segment(i * num_points, num_points) = yi;
	}
	return y;
}

// 获取所有风机网格点的z坐标
VectorXd WindFarmOptimization::getZ() const {
	VectorXd z = this->layout.col(2);
	return z;
}

// private method for rotating turbine layout
Turbine_cell WindFarmOptimization::rotated_turbine(Eigen::VectorXd& center, double angle) {
	angle = angle * M_PI_LOCAL / 180.0; // deg to rad
	RowVectorXd center_row = center.transpose();
	MatrixXd pos = this->layout;
	MatrixXd pos_offset = pos.rowwise() - center_row;
	MatrixXd RotMat(3, 3);
	RotMat << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1.0;
	MatrixXd rot_pos_center = (RotMat * pos_offset.transpose()).transpose();
	MatrixXd rot_layout = rot_pos_center.rowwise() + center_row;
	std::array<int, MaxTurbines> idx_o = this->idx_org;
	Turbine_cell rot_turbinearray = { rot_layout, this->turbine_chart,  idx_o };
	return rot_turbinearray;
}

// private method for rotation cache computation
RotatedResult WindFarmOptimization::compute_rotated() {
	VectorXd bd = getBounds();
	VectorXd center(3);
	center << 0.5 * (bd(0) + bd(1)), 0.5 * (bd(2) + bd(3)), 0;

	//绕center旋转
	Turbine_cell rot_turbinearray = rotated_turbine(center, wind_direction);

	// 旋转后按x坐标排序
	Turbine_cell sorted_rot_turbinearray = sortInX(rot_turbinearray);
	MatrixXd sorted_layout = sorted_rot_turbinearray.layout;
	std::array<int, MaxTurbines> sorted_idx = sorted_rot_turbinearray.idx;

	// 网格点旋转后坐标(this.x, this.y, this.z)
	VectorXd x_grid = getX();
	VectorXd y_grid = getY();
	VectorXd z_grid = getZ();
	MatrixXd xyz_grid(x_grid.rows(), 3);
	xyz_grid.col(0) = x_grid;
	xyz_grid.col(1) = y_grid;
	xyz_grid.col(2) = z_grid;

	MatrixXd rot_xyz_grid = rot_func(xyz_grid, center, wind_direction);
	VectorXd rot_x_grid = rot_xyz_grid.col(0);
	VectorXd rot_y_grid = rot_xyz_grid.col(1);
	VectorXd rot_z_grid = rot_xyz_grid.col(2);

	// precached wake model computation results

	const int N = x_grid.rows();
	std::vector<VectorXd> delta_u(159);
	std::vector<VectorXd> sigma_square(159);
	std::vector<double> coeff(159);
	std::vector<VectorXd> exp_term(159);
	std::vector<VectorXd> sigma_tm(159);
	std::vector<VectorXd> R_half(159);
	// storage format:std::vector<VectorXd(159)> xx(159);
	// VectorXd x_d(N), y_d(N), z_d(N);
	std::vector<VectorXd> x_d(N), y_d(N), z_d(N); //159 doublearr in 159
	for (int i = 0; i < N; i++) {
		x_d[i] = rot_x_grid.array() - sorted_layout(i, 0);
		y_d[i] = rot_y_grid.array() - sorted_layout(i, 1);
		z_d[i] = rot_z_grid.array() - sorted_layout(i, 2);
		double rotor_D = this->turbine_chart[0].rotor_diameter;
		VectorXd d_w = 1 + 0.0834 * log(1 + exp((x_d[i].array() - rotor_D) / rotor_D * 2));
		double sigma_0 = 0.25 * rotor_D;
		VectorXd sigma_y = sigma_0 * d_w.array();
		VectorXd erf_val = x_d[i].unaryExpr([rotor_D](double x_di) { return erf(x_di * sqrt(2.0) / rotor_D); });
		delta_u[i] = (1.0 / (2 * (d_w.array().square()))).array() * (1 + erf_val.array()).array();
		sigma_square[i] = 2 * sigma_0 * sigma_0 * d_w.array().square();
		coeff[i] = rotor_D * rotor_D / (8 * sigma_0);
		exp_term[i] = (-((z_d[i].array()).square())).exp() / sigma_square[i].array();
		sigma_tm[i] = sigma_y / (2 * log(2));
		R_half[i] = sigma_y * sqrt(2*log(2));
	}

	// wake deflection model
	double x_max = 12000;
	//polynomial coefficients for delta_y calculation (order=6)
	VectorXd p11(7), p22(7), p33(7);
	p11 << 10.0036, -17.567, -11.4904, 10.5864, 42.2003, -73.2544, -634.79;
	p22 << 11.2166, -16.8436, -21.0378, 20.0129, 34.061, -45.3254, -465.76;
	p33 << 11.0171, -17.3562, -18.3961, 17.581, 36.6349, -53.5914, -520.945;
	std::vector<std::vector<VectorXd>> y_model(N);
	std::vector<std::vector<int>> idx_compact(N);
	std::vector<std::vector<int>> idx_mask(N);
	for (int i = 0; i < N; i++) {
		// 返回命中的idx索引并紧凑存储
		for (int i_idx = 0; i_idx < N; i_idx++) {
			if (x_d[i](i_idx) > 0 && x_d[i](i_idx) < x_max) {
				idx_compact[i].push_back(i_idx);
				idx_mask[i].push_back(1);
			}
			else
			 idx_mask[i].push_back(0);
		}
		//根据idx_compact 切片
		VectorXd zz_sel = (x_d[i](idx_compact[i]).array()-6000)/3465.54;
		VectorXd y_sel1 = polyval(zz_sel, p11);
		VectorXd y_sel2 = polyval(zz_sel, p22);
		VectorXd y_sel3 = polyval(zz_sel, p33);
		y_model[i].push_back(y_sel1);
		y_model[i].push_back(y_sel2);
		y_model[i].push_back(y_sel3);

		////反向映射回原始位置（未命中位置赋值为-1）
		//VectorXd pos_to_compact = VectorXd::Constant(N, -1);
		//for (int j = 0; j < idx_compact[i].size(); j++) {
		//	pos_to_compact(idx_compact[i][j]) = j;
		//}
		// y_d[i] = pos_to_compact;
	}

	//construct and return RotatedResult
	struct RotatedResult result;
	result.sorted_coords = sorted_layout;
	result.sorted_indexes = sorted_idx;
	result.rot_x_grid = rot_x_grid;
	result.rot_y_grid = rot_y_grid;
	result.rot_z_grid = rot_z_grid;
	result.x_d = x_d;
	result.y_d = y_d;
	result.z_d = z_d;
	result.delta_u = delta_u;
	result.sigma_square = sigma_square;
	result.coeff = coeff;
	result.exp_term = exp_term;
	result.sigma_tm = sigma_tm;
	result.R_half = R_half;
	result.y_model = y_model;
	result.idx_mask = idx_mask;

	return result;
}


RotatedResult WindFarmOptimization::prepareStaticCache() {
	if (cache_stored == 0) {
		rotated_result_cache = compute_rotated();
		cache_stored = 1;
	}
	return rotated_result_cache;
}

// calculatewake
void WindFarmOptimization::calculateWake() {
	VectorXd u_init = VectorXd::Constant(n_turbines, wind_speed); // initialize u field
	VectorXd turbulence_init = VectorXd::Constant(n_turbines, added_turbulence_intensity); // initialize turbulence field

	// for each turbine grid(1xN)
	this->u = u_init;
	this->turbulence = turbulence_init;

	// Read cached rotated results (to be implemented)
	RotatedResult rotated_result = prepareStaticCache();
	std::array<int, MaxTurbines> sorted_indexes = rotated_result.sorted_indexes;
	std::vector<VectorXd>& x_d = rotated_result.x_d;
	std::vector<VectorXd>& y_d = rotated_result.y_d;
	std::vector<VectorXd>& z_d = rotated_result.z_d;

	// Wake steering model
	std::vector<std::vector<VectorXd>>& y_model = rotated_result.y_model;
	std::vector<std::vector<int>>& idx_mask = rotated_result.idx_mask;

	// Wake deficit model
	auto& delta_u = rotated_result.delta_u;
	auto& sigma_square = rotated_result.sigma_square;
	auto& coeff = rotated_result.coeff;
	auto& exp_term = rotated_result.exp_term;
	auto& sigma_tm = rotated_result.sigma_tm;
	auto& R_half = rotated_result.R_half;
	// two fields to be computed (for each turbine grid, 1xN)
	VectorXd u_wake = MatrixXd::Zero(u.rows(), u.cols());
	VectorXd u_turb_wake = MatrixXd::Zero(u.rows(), u.cols());

	// Iterate through turbines and calculate wake effects from each turbine
	//omp_set_num_threads(8); // set number of threads
 //#pragma omp parallel for shared(u_wake, u_turb_wake)
	for (int i = 0; i < n_turbines; i++) {
		std::vector<double> u_wake_vec(u_wake.data(), u_wake.data() + u_wake.size());
		turbine_chart[sorted_indexes[i] - 1].updateVelocities(u_wake_vec, this->wind_speed, sorted_indexes[i]);

		// compute deflection
		VectorXd deflection = deflection_function(
			x_d[i],
			turbine_chart[sorted_indexes[i] - 1],
			idx_mask[i],
			y_model[i]
		);

		// compute velocity deficit
		VectorXd u_wake_induced = velocity_function(
			delta_u[i],
			coeff[i],
			sigma_square[i],
			exp_term[i],
			x_d[i],
			y_d[i],
			z_d[i],
			turbine_chart[sorted_indexes[i] - 1],
			deflection,
			this->wind_speed
		);

		// compute added turbulence
		VectorXd add_turb_induced = turbulence_function(
			sigma_tm[i],
			R_half[i],
			x_d[i],
			y_d[i],
			z_d[i],
			turbine_chart[sorted_indexes[i] - 1],
			deflection,
			turbulence_init
		);

		// update turbulence intensity field
		std::vector<double> add_turb_induced_vec(add_turb_induced.data(),
			add_turb_induced.data() + add_turb_induced.size());
		turbine_chart[sorted_indexes[i] - 1].updateTurbulenceIntensity(
			add_turb_induced_vec,
			n_turbines
		);

		// combination of velocity deficits
		u_wake = combination_function(u_wake, u_wake_induced);
		// combination of added turbulence
		u_turb_wake = turbulence_combination_function(u_turb_wake, add_turb_induced);

	}
	// accumulate wake effects
	this->u = u_init - u_wake;
	this->turbulence = u_turb_wake;
}

// Calculate affecting turbines and return wake matrix
MatrixXi WindFarmOptimization::calculate_affturbines() {
    VectorXd u_init = VectorXd::Constant(n_turbines, wind_speed); // initialize u field
    
    // for each turbine grid(1xN)
    this->u = u_init;

    // Read cached rotated results
    RotatedResult rotated_result = prepareStaticCache();
    std::array<int, MaxTurbines> sorted_indexes = rotated_result.sorted_indexes;
    MatrixXd sorted_coords = rotated_result.sorted_coords;
    std::vector<VectorXd> x_d = rotated_result.x_d;
    std::vector<VectorXd> y_d = rotated_result.y_d;
    std::vector<VectorXd> z_d = rotated_result.z_d;

    // Wake steering model
    std::vector<std::vector<VectorXd>> y_model = rotated_result.y_model;
    std::vector<std::vector<int>> idx_mask = rotated_result.idx_mask;

    // Wake deficit model
    auto delta_u = rotated_result.delta_u;
    auto sigma_square = rotated_result.sigma_square;
    auto coeff = rotated_result.coeff;
    auto exp_term = rotated_result.exp_term;

    // Initialize wake matrix
    MatrixXi wake_matrix = MatrixXi::Zero(n_turbines, n_turbines);
    
    // Initialize wake field
    VectorXd u_wake = VectorXd::Zero(n_turbines);

    // Iterate through turbines
    for (int i = 0; i < n_turbines; i++) {
        std::vector<double> u_wake_vec(u_wake.data(), u_wake.data() + u_wake.size());
        turbine_chart[sorted_indexes[i] - 1].updateVelocities(u_wake_vec, this->wind_speed, sorted_indexes[i]);

        // compute deflection
        VectorXd deflection = deflection_function(
            x_d[i],
            turbine_chart[sorted_indexes[i] - 1],
            idx_mask[i],
            y_model[i]
        );

        // compute velocity deficit
        VectorXd turb_u_wake = velocity_function(
            delta_u[i],
            coeff[i],
            sigma_square[i],
            exp_term[i],
            x_d[i],
            y_d[i],
            z_d[i],
            turbine_chart[sorted_indexes[i] - 1],
            deflection,
            this->wind_speed
        );

        // Check wake effects on downstream turbines
        for (int j = 0; j < n_turbines; j++) {
            // Only check turbines downstream (sc(j,1) > sc(i,1))
            if (sorted_coords(j, 0) > sorted_coords(i, 0)) {
                // Calculate undisturbed velocities
                std::vector<double> u_init_vec(u_init.data(), u_init.data() + u_init.size());
                std::vector<double> undisturbed_velocities_vec = turbine_chart[sorted_indexes[j] - 1].calculateTurbineVelocities(
					u_init_vec, sorted_indexes[j]);
				VectorXd undisturbed_velocities = VectorXd::Map(undisturbed_velocities_vec.data(), undisturbed_velocities_vec.size());

                // Calculate disturbed velocities
                VectorXd u_disturbed = u_init - turb_u_wake;
                std::vector<double> u_disturbed_vec(u_disturbed.data(), u_disturbed.data() + u_disturbed.size());
				std::vector<double> disturbed_velocities_vec = turbine_chart[sorted_indexes[j] - 1].calculateTurbineVelocities(
					u_disturbed_vec, sorted_indexes[j]);
				VectorXd disturbed_velocities = VectorXd::Map(disturbed_velocities_vec.data(), disturbed_velocities_vec.size());

                // Calculate area overlap
                double area_overlap = calculate_overlap(
                    undisturbed_velocities, 
                    disturbed_velocities, 
                    turbine_chart[sorted_indexes[j] - 1]);

                // Set wake matrix if there's overlap
                if (area_overlap > 0) {
                    wake_matrix(sorted_indexes[i] - 1, sorted_indexes[j] - 1) = 1;
                }
            }
        }

        // Combine wake effects
        u_wake = combination_function(u_wake, turb_u_wake);
    }

    return wake_matrix;
}

// optim_objective
// 获取所有风机偏航角
VectorXd WindFarmOptimization::getYawAngles() const {
	int nt = static_cast<int>(layout.rows());
	VectorXd yaw_angles(nt);
	for (int i = 0; i < nt; i++) {
		yaw_angles(i) = this->turbine_chart[i].yaw_angle;
	}
	return yaw_angles;
}

// 设置所有风机偏航角
void WindFarmOptimization::setYawAngles(const std::vector<double>& yaw_angles) {
	int nt = static_cast<int>(layout.rows());
	for (int i = 0; i < nt; i++) {
		this->turbine_chart[i].yaw_angle = yaw_angles[i];
	}
}

// 获取所有风机功率
VectorXd WindFarmOptimization::getTurbinesPower() const {
	int nt = static_cast<int>(layout.rows());
	VectorXd powers(nt);
	for (int i = 0; i < nt; i++) {
		powers(i) = this->turbine_chart[i].getPower();
	}
	return powers;
}

// 获取所有风机湍流
VectorXd WindFarmOptimization::getTurbinesTurbulence() const {
	int nt = static_cast<int>(layout.rows());
	VectorXd turbulences(nt);
	for (int i = 0; i < nt; i++) {
		turbulences(i) = this->turbine_chart[i].turbulence;
	}
	return turbulences;
}

// 获取风场总功率
double WindFarmOptimization::getFarmPower() const {
	double total_power = 0.0;
	int nt = static_cast<int>(layout.rows());
	for (int i = 0; i < nt; i++) {
		total_power += this->turbine_chart[i].getPower();
	}
	return total_power;
}

// 获取青州12风场功率
double WindFarmOptimization::getFarmQingzhou12Power() const {
	double total_power = 0.0;
	for (int idx : qz_12) {
		total_power += this->turbine_chart[idx - 1].getPower(); // idx assumed 1-based
	}
	return total_power;
}

// 获取青州3风场功率
double WindFarmOptimization::getFarmQingzhou3Power() const {
	double total_power = 0.0;
	for (int idx : qz_3) {
		total_power += this->turbine_chart[idx - 1].getPower(); // idx assumed 1-based
	}
	return total_power;
}

// 获取场群各风机寿命

// // 优化相关函数
//
// // grad_f_yaw_i
// // NOTE: idx is 1-based!
// double WindFarmOptimization::grad_f_yaw_i(int idx) {
// 	// center-differencing method
// 	WindFarmOptimization w = *this; // make a copy to avoid modifying original
// 	double yaw_org = w.turbine_chart[idx - 1].yaw_angle;
// 	double h = 1; // small perturbation
// 	w.turbine_chart[idx - 1].yaw_angle = yaw_org + h;
// 	w.calculateWake();
// 	double pwr_plus = getFarmPower();
// 	w.turbine_chart[idx - 1].yaw_angle = yaw_org - h;
// 	w.calculateWake();
// 	double pwr_minus = getFarmPower();
// 	double grad_fi = (pwr_plus - pwr_minus) / (2 * h);
// 	return grad_fi;
// }
//
//
// // grad_f
// VectorXd WindFarmOptimization::grad_f(const std::vector<double>& yaw_angles) {
// 	omp_set_num_threads(15);
// 	int nt = static_cast<int>(layout.rows());
// 	setYawAngles(yaw_angles);
// 	VectorXd grad_f_vec(nt);
// 	// compute grad_f for each turbine, idx is 1-based
// 	// parallel computation to be implemented later
// 	std::vector<double> grad_f_vector(nt);
// 	#pragma omp parallel for
// 		for (int i = 0; i < nt; i++) {
// 			grad_f_vector[i] = grad_f_yaw_i(i + 1); // idx is 1-based
// 	}
// 	grad_f_vec = Eigen::Map<VectorXd>(grad_f_vector.data(), grad_f_vector.size());
// 	return grad_f_vec;
// }
//
// // grad_g_yaw_i
// VectorXd WindFarmOptimization::grad_g_yaw_i(int idx) {
// 	// center-differencing method
// 	WindFarmOptimization w = *this; // make a copy to avoid modifying original
// 	double yaw_org = w.turbine_chart[idx - 1].yaw_angle;
// 	double h = 0.1; // small perturbation
// 	w.turbine_chart[idx - 1].yaw_angle = yaw_org + h;
// 	w.calculateWake();
// 	VectorXd g_plus(2);
// 	g_plus << getFarmQingzhou12Power(), getFarmQingzhou3Power();
// 	w.turbine_chart[idx - 1].yaw_angle = yaw_org - h;
// 	w.calculateWake();
// 	VectorXd g_minus(2);
// 	g_minus << getFarmQingzhou12Power(), getFarmQingzhou3Power();
// 	return (g_plus - g_minus) / (2 * h);
// }
//
// // grad_g
// MatrixXd WindFarmOptimization::grad_g(const std::vector<double>& yaw_angles) {
// 	omp_set_num_threads(15);
//
// 	setYawAngles(yaw_angles);
// 	int nt = static_cast<int>(layout.rows());
// 	MatrixXd grad_g_mat(nt, 2); // nt x 2 matrix, 2 indicates two constraints
// 	// compute grad_g for each turbine, idx is 1-based
// 	// parallel computation to be implemented later
// 	#pragma omp parallel for
// 		for (int i = 0; i < nt; i++) {
// 			VectorXd grad_gi = grad_g_yaw_i(i + 1); // 1x2 vector
// 			RowVectorXd grad_gi_row(grad_gi);
// 			grad_g_mat.row(i) = grad_gi_row; // idx is 1-based
// 		}
//
// 	return grad_g_mat;
// }


// 坐标排序函数（按x轴）
// 返回新的排序后turbine_cell
// sorted_indices 是 0-based 的行索引（原 layout 中每行在排序后的位置）
Turbine_cell sortInX(const Turbine_cell& turbine_arr) {

	MatrixXd layout = turbine_arr.layout;
	std::vector<Turbine> sorted_turbine_chart = turbine_arr.turbine_chart;
	std::array<int, MaxTurbines> idx = turbine_arr.idx;

	const int N = layout.rows();

	// 按 layout(i,0) 排序（x 值），stable_sort 保持相等元素的相对顺序
	std::stable_sort(idx.begin(), idx.end(),
		[&layout](int a, int b) {
			return layout(a - 1, 0) < layout(b - 1, 0); //idx 1-159, subscript 0-158
		}
	);

	// 构造排序后的layout
	MatrixXd sorted_layout = layout;
	for (int i = 0; i < N; i++)
		sorted_layout.row(i) = layout.row(idx[i] - 1); //idx->subscript

	//构造排序后的turbine_chart
	for (int i = 0; i < N; i++)
		sorted_turbine_chart[i] = turbine_arr.turbine_chart[idx[i] - 1];

	// 构造 Eigen::VectorXi 索引返回（0-based）
	std::array<int, MaxTurbines> sorted_idx{};
	for (int i = 0; i < N; i++)
		sorted_idx[i] = idx[i];

	//return { sorted_layout, sorted_indices };
	return Turbine_cell{ sorted_layout, sorted_turbine_chart, sorted_idx };
}

VectorXd polyval(const VectorXd& x, const VectorXd& coeffs) {
	int degree = coeffs.size() - 1;
	VectorXd result = VectorXd::Zero(x.size());
	for (int i = 0; i <= degree; ++i) {
		result.array() += coeffs(i) * x.array().pow(degree - i);
	}
	return result;
}

double calculate_overlap(const VectorXd& undisturbed_velocity, const VectorXd& disturbed_velocity, const Turbine& turbine) {
	const double tol = 0.00001; // absolute threshold

	VectorXd diff = undisturbed_velocity - disturbed_velocity;

	// Check if any point has a decrease exceeding the threshold
	bool isAffected = (diff.array() >= tol).any();

	return isAffected ? 1.0 : 0.0; // return 1 or 0 for if area_overlap > 0 check
}