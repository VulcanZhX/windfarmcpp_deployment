#include "Farm.hpp"
using namespace Eigen;

// MATLAB的cosd/sind是角度，C++标准库是弧度
inline static double cosd(double deg) { return std::cos(deg * M_PI_LOCAL / 180.0); }
inline static double sind(double deg) { return std::sin(deg * M_PI_LOCAL / 180.0); }

Eigen::VectorXd velocity_function(
    const VectorXd& delta_u_coff,
    double coeff,
    const VectorXd& sigma_square,
    const VectorXd& exp_term,
    const VectorXd& x_d,
    const VectorXd& y_d,
    const VectorXd& z_d,
    const Turbine& turbine,
    const VectorXd& deflection,
    double u_initial
) {
    int n = x_d.size();
    
    
    double cos_yaw = cosd(turbine.yaw_angle);
    VectorXd delta_u_eigen = (1.0 - std::sqrt(1.0 - turbine.getCt() * cos_yaw * cos_yaw)) * delta_u_coff;
    VectorXd percent_deficit = delta_u_eigen.array() * coeff * (-(y_d - deflection).array().square() / sigma_square.array()).exp()*exp_term.array(); 
    VectorXd deficit = u_initial * percent_deficit;

    // 条件置零（向量化写法）
    deficit = (x_d.array() < 0).select(0, deficit);
    deficit = (x_d.array() > 12000).select(0, deficit);

    // 统计u方向的turbulence
    return deficit;
}
