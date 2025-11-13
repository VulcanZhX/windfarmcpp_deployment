#ifndef FARM_HPP
#define FARM_HPP
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include "Turbine.hpp"
// #include "Toolset.hpp"
/*注意：除输入/最终输出与特殊情况下（非数值类数组）时使用vector,其余一律使用Eigen库*/

typedef std::vector<std::tuple<Eigen::MatrixXd, Turbine, int>> TURB_CELLARR; // 风场所有风机布局
constexpr std::size_t MaxTurbines = 159; //必须与输入风机数量一致

struct Turbine_cell {
    Eigen::MatrixXd layout; // 布局
    std::vector<Turbine> turbine_chart; // 风机数组
    std::array<int, MaxTurbines> idx; // 编号
};

//// 结果结构体：包含所有需要返回的 rotated 信息
// 结构体定义
struct RotatedResult {
    // 排序后的风机坐标 (N x 3)
    Eigen::MatrixXd sorted_coords;      // (N, 3)
    std::array<int, MaxTurbines> sorted_indexes;     // (N,)

    // 所有风机所有网格点的旋转坐标
    Eigen::VectorXd rot_x_grid;
    Eigen::VectorXd rot_y_grid;
    Eigen::VectorXd rot_z_grid;

    // x_d, y_d, z_d
    std::vector<Eigen::VectorXd> x_d;
    std::vector<Eigen::VectorXd> y_d;
    std::vector<Eigen::VectorXd> z_d;

    // 尾流赤字模型
    std::vector<Eigen::VectorXd> delta_u;    
    std::vector<Eigen::VectorXd> sigma_square;
    std::vector<double> coeff; 
    std::vector<Eigen::VectorXd> exp_term; 

    // 湍流模型
    std::vector<Eigen::VectorXd> sigma_tm; 
    std::vector<Eigen::VectorXd> R_half;

    // 偏转模型 y_model: 每台风机3组多项式输出
    std::vector<std::vector<Eigen::VectorXd>> y_model;

    // mask: 1表示命中，0表示未命中
    std::vector<std::vector<int>> idx_mask;
};

class WindFarmOptimization {
public:
    // 风资源参数
    double wind_speed; // 风速
    double wind_direction; // 风向（度）
    double turbulence_intensity; // 湍流强度
    double added_turbulence_intensity = 0.0; // 额外湍流强度
    double wind_shear; // 风切变
    double wind_veer; // 风偏转
    int n_turbines; // 风机数量
    std::array<int, MaxTurbines> idx_org; // 原始风机编号顺序

    // 风场布局(originally included in turbinechart class)
    Eigen::MatrixXd layout; // 每行一个风机的[x, y, z]
    std::vector<Turbine> turbine_chart; // 风机数组
    TURB_CELLARR turbine_tuple;

    // 优化相关参数
    double yaw_lower = -30.0; // 偏航角下限
    double yaw_upper = 30.0;  // 偏航角上限
    // 风机数量(92+67=159)
    std::vector<int> qz_12; // 青州12风机编号
    std::vector<int> qz_3;  // 青州3风机编号

    // 风场网格相关
    Eigen::VectorXd u; // 风速场
    Eigen::VectorXd turbulence; // 湍流场
    
	// Methods declarations

    // 构造函数
    WindFarmOptimization(
        const std::vector<std::vector<double>>& PT, // 功率推力查表
        const std::array<std::vector<int>, 3>& t_qz, // 青州123风机编号
        const std::vector<double>& turbulence_sheer_veer, // 湍流切变偏转
        const std::vector<double>& turbine_diameter_vector, // 风机直径
        const std::vector<double>& turbine_hub_height_vector, // 风机轮毂高度
        const std::vector<double>& rated_power_vector, // 风机额定功率
        const std::vector<double>& life_total_vector, // 风机寿命
        const std::vector<double>& repair_c_vector, // 维修成本
        const std::vector<double>& yaw_vec, // 偏航角向量
        const std::vector<double>& fatigue, // 疲劳
        const std::vector<double>& fatigue_p, // 疲劳参数
        const std::vector<std::vector<double>>& layout_farm, // 风场布局
        double wind_speed, // 风速
        double wind_direction // 风向
        );

    // 析构函数
	~WindFarmOptimization() = default;

    // 风场边界
    Eigen::VectorXd getBounds() const;

    // 风机索引
    std::vector<int> getIndexes();

    //获取所有xyz坐标
    Eigen::VectorXd getX() const;
    Eigen::VectorXd getY() const;
    Eigen::VectorXd getZ() const;


    // 获取所有风机偏航角
    Eigen::VectorXd getYawAngles() const;

    // 设置所有风机偏航角
    void setYawAngles(const std::vector<double>& yaw_angles);

    // 获取所有风机功率
    Eigen::VectorXd getTurbinesPower() const;

    // 获取所有风机湍流
    Eigen::VectorXd getTurbinesTurbulence() const;

    // 获取风场总功率
    double getFarmPower() const;

    // 获取青州12风场功率
    double getFarmQingzhou12Power() const;

    // 获取青州3风场功率
    double getFarmQingzhou3Power() const;

    // 风场效能函数
    double farmEfficiencyFunction() const; // to be implemented

    // 计算尾流影响
    void calculateWake();
    Eigen::MatrixXi calculate_affturbines();

    // 其他接口和成员函数根据需要补充...

private:
    
	RotatedResult rotated_result_cache; // 预计算缓存等
	int cache_stored = 0; // 标志，指示缓存是否已存储

	Turbine_cell rotated_turbine(Eigen::VectorXd& center, double angle); // turbine layout rotation

    // 预计算部分
    RotatedResult compute_rotated();

    // 准备安装缓存
    RotatedResult prepareStaticCache();

};

Turbine_cell sortInX(const Turbine_cell& turbine_arr); // 按x坐标排序
Eigen::VectorXd polyval(const Eigen::VectorXd& x, const Eigen::VectorXd& coeffs); // 多项式求值
// 计算风机尾流重叠面积比例
double calculate_overlap(const Eigen::VectorXd& undisturbed_velocity, const Eigen::VectorXd& disturbed_velocity, const Turbine& turbine);


// 工具函数

// 生成 m 行 n 列的二维随机 double vector
std::vector<std::vector<double>> generateRandomPT(int m, int n, double lwr, double upr);
// 读取所有数据到二维 double vector
std::vector<std::vector<double>> readCSV(const std::string& filename);
std::vector<std::vector<int>> readCSVInt(const std::string& filename);

// 风场风速计算
Eigen::VectorXd velocity_function(
    const Eigen::VectorXd& delta_u_coff,
    double coeff,
    const Eigen::VectorXd& sigma_square,
    const Eigen::VectorXd& exp_term,
    const Eigen::VectorXd & x_d,
    const Eigen::VectorXd & y_d,
    const Eigen::VectorXd & z_d,
    const Turbine& turbine,
    const Eigen::VectorXd & deflection,
    double u_initial
);

// 风场偏转计算
Eigen::VectorXd deflection_function(
    const Eigen::VectorXd& x_d,
    const Turbine& turbine,
    const std::vector<int>& idx,
    const std::vector<Eigen::VectorXd>& y_model
);

// 风场湍流计算
Eigen::VectorXd turbulence_function(
    const Eigen::VectorXd& sigma_tm,
    const Eigen::VectorXd& R_half,
    const Eigen::VectorXd& x_d,
    const Eigen::VectorXd& y_d,
    const Eigen::VectorXd& z_d,
    const Turbine& turbine,
    const Eigen::VectorXd& deflection,
    const Eigen::VectorXd& turbulence_init);

// 风场尾流合成
Eigen::VectorXd combination_function(
    const Eigen::VectorXd& u_wake,
    const Eigen::VectorXd& u_wake_induced
);

// 风场湍流合成
Eigen::VectorXd turbulence_combination_function(
    const Eigen::VectorXd& turbulence_wake,
    const Eigen::VectorXd& turbulence_wake_induced
);

// 风机布局旋转
Eigen::MatrixXd rot_func(Eigen::MatrixXd& pos, Eigen::VectorXd& center, double angle);

#endif