/*MAIN PROGRAM*/

#include <iostream>
#include "Farm.hpp"
#include "wrappedfunc.hpp"

int main(int, char**)
{
	// initialize windfarm only once
	initializeWindFarm();
	// update windfarm and perform optimization
	double new_wind_speed = 9.5;
	double new_wind_direction = 275.0;
	std::vector<double> new_yaw_angles = generateRandomPT(1, MaxTurbines, -30.0, 30.0)[0];
	optimizeWindFarm(new_wind_speed, new_wind_direction, new_yaw_angles);
	// save the result here
	Eigen::VectorXd final_yaw_angles = g_farmopt->getYawAngles();
	double final_farm_power = g_farmopt->getFarmPower();
	std::cout << "Final Farm Power: " << final_farm_power << " W" << std::endl;
	cleanupWindFarm();
	return 0;
}