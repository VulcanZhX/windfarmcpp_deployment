// Initialization will only be done once, subsequent calls will skip initialization.

#include "wrappedfunc.hpp"
#include "MyNLP.hpp"
#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include <IpOptionsList.hpp>

WindFarmOptimization* g_farmopt = nullptr;

// Initialization function
bool initializeWindFarm() {
	if (g_farmopt == nullptr) {
		try {
			int sqz_12 = 92;
			// manually set params for verification
			std::vector<int> t_qz_12(sqz_12); // 1-92
			std::iota(t_qz_12.begin(), t_qz_12.end(), 1);
			std::vector<std::vector<int>> t_qz_3 = readCSVInt("../t_qz.csv");
			std::array<std::vector<int>, 3> t_qz = { t_qz_12, t_qz_3[0], t_qz_3[1] };

			// misc params read from csv
			std::vector<std::vector<double>> params = readCSV("../params.csv");
			const std::vector<double>& turbulence_sheer_veer = params[0];
			const std::vector<double>& turbine_diameter_vector = params[1];
			const std::vector<double>& turbine_hub_height_vector = params[2];
			const std::vector<double>& rated_power_vector = params[3];
			const std::vector<double>& life_total_vector = params[4];
			const std::vector<double>& repair_c_vector = params[5];

			std::vector<double> fatigue(159, 0);
			const std::vector<double>& fatigue_p = params[6];

			// Cp and Ct
			std::vector<std::vector<double>> PT = readCSV("../Cpt.csv");
			// x,y,z position(layout)
			std::vector<std::vector<double>> XYZ = readCSV("../xyz.csv");

			std::vector<std::vector<double>> yaw_vec = generateRandomPT(1, 159, -30.0, 30.0);
			double wind_speed = 10;
			double wind_direction = 60;

			// Create static instance
			g_farmopt = new WindFarmOptimization(PT, t_qz,
				turbulence_sheer_veer, turbine_diameter_vector,
				turbine_hub_height_vector, rated_power_vector,
				life_total_vector, repair_c_vector,
				yaw_vec[0], fatigue, fatigue_p,
				XYZ, wind_speed, wind_direction);

			return true;
			g_farmopt->calculateWake();
		}
		catch (const std::exception& e) {
			std::cerr << "Initialization failed: " << e.what() << std::endl;
			return false;
		}
	}
	else
		std::cout << "WindFarm already initialized." << std::endl;
	return true;
}

// Optimization function
bool optimizeWindFarm(double new_wind_speed, double new_wind_direction, std::vector<double>& new_yaw_angles) {
	if (!g_farmopt) {
		std::cerr << "WindFarm not initialized!" << std::endl;
		exit(-1);
	}
	// check args
	if (new_yaw_angles.size() != MaxTurbines) {
		std::cerr << "Yaw angles not correct." << std::endl;
		// use default yaw angles
		new_yaw_angles = generateRandomPT(1, MaxTurbines, -30.0, 30.0)[0];
	}
	g_farmopt->wind_speed = new_wind_speed;
	g_farmopt->wind_direction = new_wind_direction;
	g_farmopt->setYawAngles(new_yaw_angles);
	g_farmopt->calculateWake();
	std::cout << "Wind Farm Power: " << 0.998*g_farmopt->getFarmPower() << " W" << std::endl;
	try {
		SmartPtr<TNLP> mynlp = new MyNLP(g_farmopt);
		SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
		ApplicationReturnStatus status = app->Initialize();
		if (status != Solve_Succeeded) {
			std::cout << "*** Error during initialization!" << std::endl;
			return false;
		}

		// Set optimization options
		app->Options()->SetIntegerValue("print_level", 3);
		app->Options()->SetStringValue("linear_solver", "ma57");
		app->Options()->SetStringValue("linear_system_scaling", "none");
		app->Options()->SetStringValue("output_file", "ipopt_out.txt");
		app->Options()->SetStringValue("hessian_approximation", "limited-memory");
		app->Options()->SetIntegerValue("max_iter", 2);

		status = app->OptimizeTNLP(mynlp);

		if (status == Solve_Succeeded || status == Maximum_Iterations_Exceeded) {
			Index iter_count = app->Statistics()->IterationCount();
			std::cout << "*** Problem solved in " << iter_count << " iterations!" << std::endl;

			// Number final_obj = app->Statistics()->FinalObjective();
			// std::cout << "*** Final objective value: " << final_obj << std::endl;
			return true;
		}
		else {
			std::cout << "*** Optimization failed with status: " << status << std::endl;
			return false;
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Optimization failed: " << e.what() << std::endl;
		return false;
	}
}

// Cleanup function
void cleanupWindFarm() {
	if (g_farmopt) {
		delete g_farmopt;
		g_farmopt = nullptr;
	}
}
