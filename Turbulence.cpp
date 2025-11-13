#include "Farm.hpp"
using namespace Eigen;

VectorXd turbulence_function(
    const VectorXd& sigma_tm,
    const VectorXd& R_half,
    const VectorXd& x_d,
    const VectorXd& y_d,
    const VectorXd& z_d,
    const Turbine& turbine,
    const VectorXd& deflection,
    const VectorXd& turbulence_init)
{
    int n = x_d.size();
    VectorXd I_d = VectorXd::Zero(n);

    // mask: 只计算下游点
    VectorXi mask = (x_d.array() >= 0).cast<int>();

    // y_d(abs(y_d) < 0.0001) = 0.0001;
    VectorXd y_d_mod = y_d.array().unaryExpr([](double v) { return std::abs(v) < 0.0001 ? 0.0001 : v; });

    // dr = sqrt((y_d - deflection).^2 + z_d.^2);
    VectorXd dr = ((y_d_mod - deflection).array().square() + z_d.array().square()).sqrt();

    // 参数
    double D = turbine.rotor_diameter;
    double gamma_deg = turbine.yaw_angle;
    double gamma_rad = gamma_deg * M_PI_LOCAL / 180.0;
    double Ct = turbine.getCt();
    double Ia = turbine.turbulence;

    // 推力系数接近0，附加湍流强度为0
    double d = 2.3 * std::pow(Ct * std::cos(gamma_rad), -1.2);
    double e = std::pow(Ia, 0.1);
    VectorXd q = 0.7 * std::pow(Ct * std::cos(gamma_rad), -3.2) * std::pow(Ia, -0.45) * (1.0 + x_d.array() / D).array().pow(-2);

    VectorXd dImax = (d + e * x_d.array() / D + q.array()).array().inverse();

    // 横向分布
    VectorXd shape_tm = VectorXd::Zero(n);
    VectorXd k1 = VectorXd::Zero(n);

	// shape_tm and k1 calculation using triple conditional
	shape_tm = (dr.array() < R_half.array()).select(
		1 - 0.15 * (1 + (M_PI_LOCAL * dr.array() / R_half.array()).cos()),
		(-((dr - R_half).array().square() / (2 * sigma_tm.array().square())).exp())
	);
	k1 = (dr.array() < R_half.array()).select(
		(dr.array() / R_half.array()).sin(),
		1.0
	);


    // alpha_tm = atan2d(abs(z_d), abs(y_d))
    VectorXd alpha_tm(n);
    alpha_tm = y_d_mod.binaryExpr(z_d, [](double y, double z) {
        return std::atan2(std::abs(z), std::abs(y)) * 180.0 / M_PI_LOCAL;
    });
    // delta_tm
    VectorXd delta_tm = VectorXd::Zero(n);
    VectorXd exp_part = ((dr - R_half).array().square() / (2 * sigma_tm.array().square())).array().unaryExpr([](double v) { return std::exp(-v); });


    // 三元表达式条件赋值
    delta_tm = (z_d.array() >= 0).select(
        0.23 * Ia * (alpha_tm.array() * M_PI_LOCAL / 180.0).sin() * k1.array() * exp_part.array(),
        -1.23 * Ia * (alpha_tm.array() * M_PI_LOCAL / 180.0).sin() * k1.array() * exp_part.array()
	);


	I_d = (mask.array() != 0).select(I_d, 0.0);

    VectorXd wake_turbulence = I_d.array() * turbulence_init.array();
    return wake_turbulence;
}