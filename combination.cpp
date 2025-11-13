#include "Farm.hpp"
using namespace Eigen;

// linear combination for velocity deficit
VectorXd combination_function(
    const VectorXd& u_wake,
    const VectorXd& u_wake_induced
) {
    VectorXd u_combined = ((u_wake.array().square()) + (u_wake_induced.array().square())).sqrt();
    return u_combined;
}

// sqrt combination for turbulence
VectorXd turbulence_combination_function(
    const VectorXd& turbulence_wake,
    const VectorXd& turbulence_wake_induced
) {
    VectorXd turbulence_combined = ((turbulence_wake.array().square()) + (turbulence_wake_induced.array().square())).sqrt();
    return turbulence_combined;
}