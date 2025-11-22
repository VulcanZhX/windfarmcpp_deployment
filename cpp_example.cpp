/*MAIN PROGRAM*/

#include <iostream>
#include "wrappedfunc.hpp"

int main(int argc, char** argv)
{
	// initialize windfarm only once
	initializeWindFarm();
	std::cout << "Testing optimizeWindFarm..." << std::endl;
	// update windfarm and perform optimization
	std::vector<double> new_wind_speed(5, 7.5);
	std::vector<double> new_wind_direction(5, 100.0);
	std::vector<std::vector<double>> init_yaw_angles = generateRandomPT(1, 159, -15, 15); // all set to 0.0 for testing
	std::vector<double> current_yaw_angles_vec = init_yaw_angles[0];
	std::vector<double> new_yaw_angles(159, 0.0);
	double new_power_12 = 0, new_power_3 = 0;
	std::vector<double> new_power_all(159, 0.0);
	// efficiency: 3 serial process, each process has its efficiency value. mul the 3 values to get final eff.
	std::vector<std::vector<double>> mEER_turbines(3, std::vector<double>(159, 1.0));
	std::vector<int> turbine_start(159, 1); // suppose all turbines on
	// update status and efficiency
	bool optflag = optimizeWindFarm(new_wind_speed,  // 输入：风速
		new_wind_direction, // 输入：方向
		mEER_turbines,
		current_yaw_angles_vec, // 输入：各风机初始偏航角度
		turbine_start, // 输入：各风机启停状态
		new_yaw_angles, // 输出：各风机偏航角度
		new_power_all, // 输出青州12功率
		new_power_12,  // 输出青州3功率
		new_power_3); // 输出全场功率); // 输入：并行计算线程数
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

	// Test optimizeWindFarmCheck
	std::cout << "Testing optimizeWindFarmCheck..." << std::endl;
	double check_power_12 = 0.0, check_power_3 = 0;
	bool checkflag = optimizeWindFarmCheck(
		7.5, // 输入：风速
		100.0, // 输入：方向
		mEER_turbines,
		new_yaw_angles, // 输入：各风机初始偏航角度
		turbine_start, // 输入：各风机启停状态
		check_power_12,  // 输出青州12功率
		check_power_3); // 输出青州3功率); // 输入：并行计算线程数
	std::cout << "Check Qingzhou 12 Power: " << check_power_12 << " W" << std::endl;
	std::cout << "Check Qingzhou 3 Power: " << check_power_3 << " W" << std::endl;
	// Final cleanup
	cleanupWindFarm();
	return 0;
}