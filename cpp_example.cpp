/*MAIN PROGRAM*/

#include <iostream>
#include "Farm.hpp"
#include "wrappedfunc.hpp"

int main(int, char**)
{
	// initialize windfarm only once
	initializeWindFarm();
	// update windfarm and perform optimization
	double new_wind_speed = 6.5;
	double new_wind_direction = 120.0;
	std::vector<double> new_yaw_angles = generateRandomPT(1, MaxTurbines, -25.0, 25.0)[0];
	double new_power_12 = 0, new_power_3 = 0, new_power_all = 0;
	optimizeWindFarm(new_wind_speed, 
		new_wind_direction, 
		new_yaw_angles, 
		new_power_12, 
		new_power_3, 
		new_power_all);
	// save the result here
	std::cout << "yaw angles (first 5): ";
	for (size_t i = 0; i < 5; ++i) {
		std::cout << new_yaw_angles[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Qingzhou 12 Power: " << new_power_12 << " W" << std::endl;
	std::cout << "Qingzhou 3 Power: " << new_power_3 << " W" << std::endl;
	std::cout << "Final Farm Power: " << new_power_all << " W" << std::endl;
	// Final cleanup
	cleanupWindFarm();
	return 0;
}