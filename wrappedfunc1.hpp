#include "Farm.hpp"

#ifndef WFOPTIM_WRAPPEDFUNC_H
#define WFOPTIM_WRAPPEDFUNC_H
// declaration of wrapped functions
// Global static instance of WindFarmOptimization
extern WindFarmOptimization * g_farmopt; // global pointer declaration  
//变量是。。。。。。。。。


extern bool initializeWindFarm();    //初始化

// 功能
extern bool optimizeWindFarm(
    std::vector<double> new_wind_speed,        // 输入：风速（序列 5个） double 类
    std::vector<double> new_wind_direction,    // 输入：方向（序列 5个）
    std::vector<double> efficents,             // 输入：效能函数 对应风机编号见 文件wind farm layout
    std::vector<double>& new_yaw_angles,      // 输出：各风机偏航角度，对应风机编号见 文件wind farm layout
    std::vector<double>& new_yaw_power         // 输出：各风机功率  对应风机编号见 文件wind farm layout
    ); 
void cleanupWindFarm();                    // 清理 内存中的 g_farmopt
#endif //WFOPTIM_WRAPPEDFUNC_H