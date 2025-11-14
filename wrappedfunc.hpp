#include "Farm.hpp"

#ifndef WFOPTIM_WRAPPEDFUNC_H
#define WFOPTIM_WRAPPEDFUNC_H
// declaration of wrapped functions
// Global static instance of WindFarmOptimization
extern WindFarmOptimization* g_farmopt; // global pointer declaration
bool initializeWindFarm();
bool optimizeWindFarm(
    double new_wind_speed, // wind speed
    double new_wind_direction, // wind direction
    std::vector<double>& new_yaw_angles); // yaw angles

bool optimizeWindFarm(
    const double new_wind_speed,        // 输入：风速（序列 5个） double 类
    const double new_wind_direction,    // 输入：方向（序列 5个）
    std::vector<double>& new_yaw_angles,      // 输出：各风机偏航角度，对应风机编号见 文件wind farm layout
    double& new_power_12, // 输出青州12功率
    double& new_power_3,  // 输出青州3功率
    double& new_power_all // 输出全场功率
    );
void cleanupWindFarm();

// 工具函数集
// 生成 m 行 n 列的二维随机 double vector
std::vector<std::vector<double>> generateRandomPT(int m, int n, double lwr, double upr);
// 读取所有数据到二维 double vector
std::vector<std::vector<double>> readCSV(const std::string& filename);
std::vector<std::vector<int>> readCSVInt(const std::string& filename);

#endif //WFOPTIM_WRAPPEDFUNC_H

