#ifndef TURBINE_HPP
#define TURBINE_HPP
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2
#include <Dense>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include "mkl.h"

const double M_PI_LOCAL = 3.1416;
//using namespace Eigen;

class Turbine {
public:
    // 构造函数
    Turbine(
        const std::vector<std::vector<double>>& PT, // 功率推力查表
        const std::array<std::vector<int>, 3>& t_qz, // 青州123风机编号
        const std::vector<double>& turbulence_sheer_veer, // 湍流切变偏转
        const std::vector<double>& turbine_diameter_vector, // 风机直径
        const std::vector<double>& turbine_hub_height_vector, // 风机轮毂高度
        const std::vector<double>& rated_power_vector, // 风机额定功率
        const std::vector<double>& life_total_vector, // 风机寿命
        const std::vector<double>& repair_c_vector, // 维修成本
        const std::vector<double>& fatigue, // 疲劳
        const std::vector<double>& fatigue_p, // 疲劳参数
        const std::vector<double>& serial_coeff_val, // 效能系数（3个串联环节）
        int status_val, // binary status: 1-active, 0-inactive(converted from 0-34 to 0,1)
		int count_tn // 机组编号，从1开始
    );

    // 属性
    double getAverageVelocity() const; // 获取风机平均风速
    double getCp() const; // 获取功率系数
    double getCt() const; // 获取推力系数
    double getPower() const; // 获取功率
    double getComprehensiveFatigueCoefficient() const; // 获取综合疲劳系数
    double getSingleTurbineObjective() const; // 获取单台风机目标函数
    double getSingleTurbineGeneration() const; // 获取单台风机发电量

    // 功率推力查表
    double fCp(double at_wind_speed) const; // 获取功率系数
    double fCt(double at_wind_speed) const; // 获取推力系数 

    // 更新风速、湍流等
    //u_initial: 1x1 scalar indicating the global natural windspeed
    std::vector<double> calculateTurbineVelocities(const std::vector<double>& local_wind_speed, int nt) const;
	void updateVelocities(const std::vector<double>& u_wake, const double u_initial, int nt); // 更新风机风速
    void updateTurbulenceIntensity(const std::vector<double>& u_turbulence_wake, int nt); // 更新湍流强度

    // 重新定义转子半径、直径
    void setRotorRadius(double r);
    void setRotorDiameter(double d);
    
    // 更新网格
    void updateGrid();
    // 更新转子半径
    void updateRadius(); 

    double rotor_diameter; // 直径
    double hub_height; // 轮毂高度
    double rated_power; // 额定功率
    double life_total; // 寿命
    double repair_c; // 维修成本
    double optimization_period = 1; // 优化周期
    double annual_average_power; // 年平均功率
    double past_comprehensive_fatigue_coefficient; // 过去综合疲劳系数
    double ref_turbulence = 0.1; // 参考湍流强度
    double dis_coefficient = 0.7; // 衰减系数
    double consumed_comprehensive_fatigue_coefficient = 0; // 综合疲劳系数
    double life_work_coeff; // 寿命工作系数
    double life_turbulence_coeff; // 寿命湍流系数
    double generator_efficiency = 1; // 发电机效率
    // additional params
    std::vector<double> serial_coeff; // 效能系数（3个串联环节）
    int status; // 风机状态，1-启用，0-停用
	// optim variable
    double yaw_angle = 0; // 偏航角

    double air_density = 1.225; // 空气密度
    double turbulence_ambient; // 环境湍流
    double rotor_radius; // 转子半径
    std::vector<double> velocities_u; // 风速网格点，待更新
    double turbulence; // 湍流强度，待更新
    std::vector<double> efficiency_value; // 效能函数值，待更新
    std::vector<std::vector<double>> power_thrust_table_11; // 功率推力查表11
    std::vector<std::vector<double>> power_thrust_table_68; // 功率推力查表68
    std::vector<std::vector<double>> power_thrust_table_83; // 功率推力查表83
    std::vector<double> grid; // 网格
    int turbine_idx_org; // 风机原始索引
    static constexpr int points_turbine_grid = 1; // 风机网格点数
    
};

#endif // TURBINE_HPP
