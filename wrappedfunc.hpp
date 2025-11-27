#include "Farm.hpp"

#ifndef WFOPTIM_WRAPPEDFUNC_H
#define WFOPTIM_WRAPPEDFUNC_H


 /*  version: 2025.11.17@sjtu */
 

// 定义全局风场对象，记录风场信息，由initializeWindFarm（）进行初始化
extern WindFarmOptimization* g_farmopt;  



/**-----------------initializeWindFarm------------------------------
 功能：读取配置文件初始化风场对象，该函数只在初始算法时调用一次。
      配置信息在文件** .csv中，包括，风机性能曲线，风机布局、风机等数据，以及
      预计算数值。
 ------------------------------------------------------------------- */
bool initializeWindFarm();


/**-----------------optimizeWindFarm--------------------------------
/* 该函数负责在每个优化周期对各风场全寿命周期发电量进行优化。
 *------------------------------------------------------------------
 * 功能：基于当前各风机偏航，在满足风机偏航实时调节约束、各场发电量大于原发电量
 *      的前提下，根据当前整个场群的环境风速与环境风向，以各风机的偏航角为优化
 *      变量，联合优化各风机以最大化全寿命周期内整个场群的发电量。
 * 特点: 1）并行优化算法，大幅降低场群规模下求解时间，提高优化效率
 *      2）结合基于启发式的算法，进行多点初值搜索，构建初值表，避免局部最优
 * 
 * 返回值：如果为true说明计算正常
 * -----------------------------------------------------------------
 */
bool optimizeWindFarm(
    const std::vector<double>& mwind_speed,     // 输入：环境风速, double 类型，单位为 m/s
    const std::vector<double>& mwind_direction, // 输入：环境风方向, double 类型，范围0-360, 以正北方向为0，单位为度
    const std::vector<std::vector<double>>& mEER_turbines,      // 输入：全场每个风机的能效系数，3x159 double matrix类型
    const std::vector<double>& current_yaw_angles,    // 输入：各风机初始偏航角度，1x159 double vector 类型，单位为度
                                                   // 该vector的编号与现场风机布局图（青洲1-3风机布局图.pdf）的编号一致，vector的第i个元素对应风机布局图编号为i的风机
    const std::vector<int>& turbine_status,        // 输入：各风机启停状态，1x159 int vector 类型
    const int analytic_grad_flag,                   // 输入：是否使用解析梯度，int 类型，0-否，1-是
    std::vector<double>& opt_yaw_angles,           // 输出：各风机偏航角度，1x159 double vector 类型，单位为度
                                                   // 该vector的编号与现场风机布局图（青洲1-3风机布局图.pdf）的编号一致，vector的第i个元素对应风机布局图编号为i的风机
    std::vector<double>& opt_power_turbines,       // 输出：全场每个风机的设定功率，1x159double vector类型，单位为W 功率<1e-2(实际<50)时 风机不发电（代表发电功率为0）
    double& opt_power_farm_12,                     // 输出：青洲12总的最大可发功率，double 类型，单位为W （用于判断电网友好模式下的发电方式）
    double& opt_power_farm_3                       // 输出：青洲3总的最大可发功率，double 类型，单位为W（用于判断电网友好模式下的发电方式）
    );

/**-----------------optimizeWindFarmCheck---------------------------
/* 该函数负责对各风场在当前风速条件下可发的最大电量进行优化计算,
 * 主要用于判断电网友好模式下的发电方式
 *------------------------------------------------------------------
 * 功能：基于当前各风机偏航，在满足风机偏航实时调节约束的前提下，
 *      根据当前整个场群的环境风速与环境风向，以各风机偏航角为优化变量，优化并
 *      输青洲12、青洲3的最大可发功率。
 * 
 * 返回值：如果为true说明计算正常
 * -----------------------------------------------------------------
 */
bool optimizeWindFarmCheck(
    const double wind_speed,           // 输入：环境风速, double 类型，单位为 m/s
    const double wind_direction,       // 输入：环境风方向, double 类型，范围0-360, 以正北方向为0，单位为度
    const std::vector<std::vector<double>>& mEER_turbines,    // 输入：全场每个风机的能效系数，3x159 double matrix类型
    const std::vector<double>& current_yaw_angles,    // 输入：各风机初始偏航角度，1x159 double vector 类型，单位为度
                                                   // 该vector的编号与现场风机布局图（青洲1-3风机布局图.pdf）的编号一致，vector的第i个元素对应风机布局图编号为i的风机
    const std::vector<int>& turbine_status, // 输入：各风机启停状态，1x159 int vector 类型
    const int analytic_grad_flag,               // 输入：是否使用解析梯度，int 类型，0-否，1-是
    double& opt_power_farm_12,                  // 输出：青洲12总的最大可发功率，double 类型，单位为W （用于判断电网友好模式下的发电方式）
    double& opt_power_farm_3                    // 输出：青洲3总的最大可发功率，double 类型，单位为W（用于判断电网友好模式下的发电方式）
    );

/**-------------------cleanupWindFarm-------------------------------
/** 功能：从内存中清除由initializeWindFarm创建的风场对象
 *                                          version: 2025.11.17@sjtu
 */
void cleanupWindFarm();
#endif

