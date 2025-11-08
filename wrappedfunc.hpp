#include "Farm.hpp"

#ifndef WFOPTIM_WRAPPEDFUNC_H
#define WFOPTIM_WRAPPEDFUNC_H
// declaration of wrapped functions
// Global static instance of WindFarmOptimization
extern WindFarmOptimization* g_farmopt; // global pointer declaration
bool initializeWindFarm();
bool optimizeWindFarm(double new_wind_speed, double new_wind_direction, std::vector<double>& new_yaw_angles);
void cleanupWindFarm();
#endif //WFOPTIM_WRAPPEDFUNC_H