/*MAIN PROGRAM*/

#include <iostream>
#include "wrappedfunc.hpp"
#include <chrono>
#include "omp.h"

int main(int argc, char** argv)
{
	// initialize windfarm only once
	initializeWindFarm();
	// g_farmopt->calculateWake();
	// std::cout << "Testing optimizeWindFarm..." << std::endl;
	// std::cout << "Verify power: " << g_farmopt->getFarmPower() << " W" << std::endl;


	// // // test calculate_affturbines
	// Eigen::MatrixXi wakemat = g_farmopt->calculate_wake_matrix();
	// // print the sum of all elems in g_farmopt
	// int sumofone = wakemat.sum();
	// std::cout << "Sum of wake matrix elements: " << sumofone << std::endl;


	// // Test yawgradSingle function
	// std::vector<double> test_yaw_angles(159, 5.0);
	// g_farmopt->setYawAngles(test_yaw_angles);
	// g_farmopt->calculateWake();
	// Eigen::MatrixXi test_wake_matrix = g_farmopt->calculate_wake_matrix();
	// // turbine idx must be 1-based
	// double yaw_grad_0 = g_farmopt->grad_p_gamma_single(157, test_wake_matrix, 2);
	// std::cout << "Yaw gradient for turbine 159 at 10 deg yaw: " << yaw_grad_0 << std::endl;

	// // test omp parallel computation
	// omp_set_num_threads(10);
	// std::vector<double> results(159, 0.0);
	// #pragma omp parallel for
	// for (int i = 0; i < 159; i++) {
	// 	results[i] = g_farmopt->grad_p_gamma_single(i+1, test_wake_matrix, 2); // idx is 1-based
	// }

	// for (int i = 0; i < 159; i++) {
	// 	std::cout << "Turbine " << i + 1 << " yaw gradient: " << results[i] << std::endl;
	// } 

	// update windfarm and perform optimization
	std::vector<double> new_wind_speed(5, 9.5);
	std::vector<double> new_wind_direction(5, 100.0);
	std::vector<std::vector<double>> init_yaw_angles = generateRandomPT(1, 159, -30, 30); // all set to 0.0 for testing
	std::vector<double> current_yaw_angles_vec = init_yaw_angles[0];
	std::vector<double> new_yaw_angles(159, 0.0);
	double new_power_12 = 0, new_power_3 = 0;
	std::vector<double> new_power_all(159, 0.0);
	// efficiency: 3 serial process, each process has its efficiency value. mul the 3 values to get final eff.
	std::vector<std::vector<double>> mEER_turbines(3, std::vector<double>(159, 1.0));
	std::vector<int> turbine_start(159, 1); // suppose all turbines on
	turbine_start[2] = 0; // turbine 2 off for testing
	int analytic_grad_flag = 1; // use analytic gradient
	// update status and efficiency
	auto start_time = std::chrono::high_resolution_clock::now();
	bool optflag = optimizeWindFarm(new_wind_speed,  // 输入：风速
		new_wind_direction, // 输入：方向
		mEER_turbines, // 输入：各风机效能系数值
		current_yaw_angles_vec, // 输入：各风机初始偏航角度
		turbine_start, // 输入：各风机启停状态
		analytic_grad_flag, // 输入：是否使用解析梯度
		new_yaw_angles, // 输出：各风机偏航角度
		new_power_all, // 输出青州12功率
		new_power_12,  // 输出青州3功率
		new_power_3); // 输出全场功率); // 输入：并行计算线程数
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	std::cout << "Optimization Time: " << elapsed.count() << " seconds" << std::endl;
	if (!optflag)
		return 0;
	// save the result here
	// print original yaw angles (first 5)
	std::cout << "Current yaw angles (first 5): ";
	for (size_t i = 0; i < 5; ++i) {
		std::cout << current_yaw_angles_vec[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "yaw angles (first 5): ";
	for (size_t i = 0; i < 5; ++i) {
		std::cout << new_yaw_angles[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Qingzhou 12 Power: " << new_power_12 << " W" << std::endl;
	std::cout << "Qingzhou 3 Power: " << new_power_3 << " W" << std::endl;
	std::cout << "Final Farm Power: " << g_farmopt->getFarmPower() << " W" << std::endl;

	// test getFarmObj 
	double final_obj = g_farmopt->getFarmObj();
	std::cout << "Final Farm Objective: " << final_obj << std::endl;
	// Test optimizeWindFarmCheck
	// std::cout << "Testing optimizeWindFarmCheck..." << std::endl;
	// double check_power_12 = 0.0, check_power_3 = 0;
	// bool checkflag = optimizeWindFarmCheck(
	// 	7.5, // 输入：风速
	// 	100.0, // 输入：方向
	// 	mEER_turbines,
	// 	new_yaw_angles, // 输入：各风机初始偏航角度
	// 	turbine_start, // 输入：各风机启停状态
	// 	check_power_12,  // 输出青州12功率
	// 	check_power_3); // 输出青州3功率); // 输入：并行计算线程数
	// std::cout << "Check Qingzhou 12 Power: " << check_power_12 << " W" << std::endl;
	// std::cout << "Check Qingzhou 3 Power: " << check_power_3 << " W" << std::endl;
	// Final cleanup
	cleanupWindFarm();
	return 0;
}