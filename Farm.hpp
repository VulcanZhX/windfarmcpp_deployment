#ifndef FARM_HPP
#define FARM_HPP

#include "Toolset.hpp"
/*注意：除输入/最终输出与特殊情况下（非数值类数组）时使用vector,其余一律使用Eigen库*/

typedef std::vector<std::tuple<Eigen::MatrixXd, Turbine, int>> TURB_CELLARR;
constexpr std::size_t MaxTurbines = 159; //必须与输入风机数量一致

struct Turbine_cell {
    Eigen::MatrixXd layout;
    std::vector<Turbine> turbine_chart;
    std::array<int, MaxTurbines> idx;
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

    // mask: (P, N) 逻辑掩码（0/1）
    std::vector<std::vector<int>> idx_mask;
};

class WindFarmOptimization {
public:
    // 风资源参数
    double wind_speed;
    double wind_direction;
    double turbulence_intensity;
    double added_turbulence_intensity = 0.0;
    double wind_shear;
    double wind_veer;
    int n_turbines;
    std::array<int, MaxTurbines> idx_org;

    // 风场布局(originally included in turbinechart class)
    Eigen::MatrixXd layout; // 每行一个风机的[x, y, z]
    std::vector<Turbine> turbine_chart;
    TURB_CELLARR turbine_tuple;

    // 优化相关参数
    double yaw_lower = -30.0;
    double yaw_upper = 30.0;
    // 风机数量(92+67=159)
    std::vector<int> qz_12;
    std::vector<int> qz_3;

    // 风场网格相关
    Eigen::VectorXd u; // 风速场
    Eigen::VectorXd turbulence; // 湍流场
    
	// Methods declarations

    // 构造函数
    WindFarmOptimization(
        const std::vector<std::vector<double>>& PT,
        const std::array<std::vector<int>, 3>& t_qz,
        const std::vector<double>& turbulence_sheer_veer,
        const std::vector<double>& turbine_diameter_vector,
        const std::vector<double>& turbine_hub_height_vector,
        const std::vector<double>& rated_power_vector,
        const std::vector<double>& life_total_vector,
        const std::vector<double>& repair_c_vector,
        const std::vector<double>& yaw_vec,
        const std::vector<double>& fatigue,
        const std::vector<double>& fatigue_p,
        const std::vector<std::vector<double>>& layout_farm,
        double wind_speed,
        double wind_direction
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

    // 获取场群各风机寿命

	// 获取场群所有风机平均寿命

    // 获取场群各风机优化指标

    // 获取场群总指标

    // 优化主入口（示例：功率优化）
    //Eigen::VectorXd optimizeYawAngles();
    
    // grad_f calculation for optimizer
    double grad_f_yaw_i(int idx);
	double grad_f_gamma_k_analytic(int kIdx);
	Eigen::VectorXd grad_f(const std::vector<double>& yaw_angles);

	// grad_g calculation for optimizer
    Eigen::VectorXd grad_g_yaw_i(int idx);
	Eigen::MatrixXd grad_g(const std::vector<double>& yaw_angles);

    // 计算尾流影响
    void calculateWake();
    Eigen::MatrixXi calculate_affturbines();

    // 其他接口和成员函数根据需要补充...

private:
    
	RotatedResult rotated_result_cache; // 预计算缓存等
	int cache_stored = 0; // 标志，指示缓存是否已存储

	Turbine_cell rotated_turbine(Eigen::VectorXd& center, double angle); // turbine layout rotation

    // rotated相关预计算
    RotatedResult compute_rotated();

    // prepare static cache
    RotatedResult prepareStaticCache();

};

Turbine_cell sortInX(const Turbine_cell& turbine_arr);
Eigen::VectorXd polyval(const Eigen::VectorXd& x, const Eigen::VectorXd& coeffs);
// Calculate overlap between undisturbed and disturbed velocities
double calculate_overlap(const Eigen::VectorXd& undisturbed_velocity, const Eigen::VectorXd& disturbed_velocity, const Turbine& turbine);

#endif