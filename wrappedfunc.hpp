#include "Farm.hpp"

#ifndef WFOPTIM_WRAPPEDFUNC_H
#define WFOPTIM_WRAPPEDFUNC_H
// declaration of wrapped functions
// Global static instance of WindFarmOptimization
extern WindFarmOptimization* g_farmopt; // global pointer declaration
bool initializeWindFarm();
bool optimizeWindFarm(double new_wind_speed, // wind speed
    double new_wind_direction, // wind direction
    std::vector<double>& new_yaw_angles); // yaw angles
void cleanupWindFarm();

// 工具函数集
// 生成 m 行 n 列的二维随机 double vector
std::vector<std::vector<double>> generateRandomPT(int m, int n, double lwr, double upr);
// 读取所有数据到二维 double vector
std::vector<std::vector<double>> readCSV(const std::string& filename);
std::vector<std::vector<int>> readCSVInt(const std::string& filename);

#endif //WFOPTIM_WRAPPEDFUNC_H

