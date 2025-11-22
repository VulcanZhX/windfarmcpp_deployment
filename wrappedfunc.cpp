// Initialization will only be done once, subsequent calls will skip initialization.

#include "wrappedfunc.hpp"
#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include <IpOptionsList.hpp>
#include <cassert>
#include <random>
#include "omp.h"

#include "IpTNLP.hpp"
#include "Farm.hpp"
WindFarmOptimization* g_farmopt = nullptr;
// 参数读取器（一次完成）
// 支持多段落读取，段落以#开头标识，后续可能扩展
// 第一段为整数表格，后续段落为 double 表格
// 返回结构体包含所有段落数据
struct SegmentsData {
    std::vector<std::vector<std::vector<int>>> firstSegment;    // 整数表(t_qz, 风场12/3风机编号)
    std::vector<std::vector<std::vector<double>>> otherSegments; // double表（风场参数表，功率/推力表，风机位置）
    int states = 0; // 读取状态标志
};

SegmentsData readMultiSegmentCSV(const std::string& filename);
std::vector<std::vector<double>> generateRandomPT(int m, int n, double lwr, double upr);


using namespace Ipopt;

class MyNLP : public TNLP
{
public:

	//params
	int pthreads = 8; // number of threads for parallel computing
	double g1_lwr = 0.0; // constraint 1 lower bound
	double g2_lwr = 0.0; // constraint 2 lower bound
	/** default constructor */
	MyNLP(WindFarmOptimization* farmopt_cpy, int _pthreads);

	/** default destructor */
	virtual ~MyNLP();

	/** windfarm class to be optimized*/
	WindFarmOptimization* farmopt;
	/**@name Overloaded from TNLP */
	//@{
	/** Method to return some info about the nlp */
	virtual bool get_nlp_info(
		Index& n,
		Index& m,
		Index& nnz_jac_g,
		Index& nnz_h_lag,
		IndexStyleEnum& index_style
	);

	/** Method to return the bounds for my problem */
	virtual bool get_bounds_info(
		Index   n,
		Number* x_l,
		Number* x_u,
		Index   m,
		Number* g_l,
		Number* g_u
	);

	/** Method to return the starting point for the algorithm */
	virtual bool get_starting_point(
		Index   n,
		bool    init_x,
		Number* x,
		bool    init_z,
		Number* z_L,
		Number* z_U,
		Index   m,
		bool    init_lambda,
		Number* lambda
	);

	/** Method to return the objective value */
	virtual bool eval_f(
		Index         n,
		const Number* x,
		bool          new_x,
		Number& obj_value
	);

	/** Method to return the gradient of the objective */
	virtual bool eval_grad_f(
		Index         n,
		const Number* x,
		bool          new_x,
		Number* grad_f
	);

	/** Method to return the constraint residuals */
	virtual bool eval_g(
		Index         n,
		const Number* x,
		bool          new_x,
		Index         m,
		Number* g
	);

	/** Method to return:
	 *   1) The structure of the Jacobian (if "values" is NULL)
	 *   2) The values of the Jacobian (if "values" is not NULL)
	 */
	virtual bool eval_jac_g(
		Index         n,
		const Number* x,
		bool          new_x,
		Index         m,
		Index         nele_jac,
		Index* iRow,
		Index* jCol,
		Number* values
	);

	 /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
	virtual void finalize_solution(
		SolverReturn status,
		Index n,
		const Number *x,
		const Number *z_L,
		const Number *z_U,
		Index m,
		const Number *g,
		const Number *lambda,
		Number obj_value,
		const IpoptData *ip_data, IpoptCalculatedQuantities *ip_cq
	);
	//@}

private:
	/**@name Methods to block default compiler methods.
	 *
	 * The compiler automatically generates the following three methods.
	 *  Since the default compiler implementation is generally not what
	 *  you want (for all but the most simple classes), we usually
	 *  put the declarations of these methods in the private section
	 *  and never implement them. This prevents the compiler from
	 *  implementing an incorrect "default" behavior without us
	 *  knowing. (See Scott Meyers book, "Effective C++")
	 */
	 //@{
	MyNLP(
		const MyNLP&
	);

	MyNLP& operator=(
		const MyNLP&
		);
	//@}

	double objPower(const Number *x); // FarmPwr(yaw_angle)
	Eigen::VectorXd gval(const Number *x, std::vector<double> &lwrbnd); // constraint values
	double grad_f_i(const Number* x, int idx);
	Eigen::VectorXd grad_f_all(const Number* x, int n, int pthreads);
	Eigen::VectorXd grad_g_i(const Number* x, int idx);
	Eigen::MatrixXd grad_g_all(const Number* x, int n);
};



// Impl
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

/* Constructor. */
MyNLP::MyNLP(WindFarmOptimization* farmopt_cpy, int _pthreads)
	: farmopt(farmopt_cpy), pthreads(_pthreads)
{
	farmopt->calculateWake(); // initial wake calculation
	g1_lwr = farmopt->getFarmQingzhou12Power() * 0.99;
	g2_lwr = farmopt->getFarmQingzhou3Power() * 0.99;
	// No additional initialization required
}

MyNLP::~MyNLP() = default;

bool MyNLP::get_nlp_info(
   Index&          n,
   Index&          m,
   Index&          nnz_jac_g,
   Index&          nnz_h_lag,
   IndexStyleEnum& index_style
)
{
   // The problem described in MyNLP.hpp has n variables
   // add checking whether turbine i is active later
   n = MaxTurbines;

   // m equality constraint,
   m = 2;

   // nonzeros in the jacobian
   nnz_jac_g = m * n; // for simplicity, the jacobian matrix is assumed to be dense

   //// and 2 nonzeros in the hessian of the lagrangian
   //// (one in the hessian of the objective for x2,
   ////  and one in the hessian of the constraints for x1)

   // We use the standard fortran index style for row/col entries
   index_style = C_STYLE;

   return true;
}

bool MyNLP::get_bounds_info(
   Index   n,
   Number* x_l,
   Number* x_u,
   Index   m,
   Number* g_l,
   Number* g_u
)
{
   // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
   // If desired, we could assert to make sure they are what we think they are.
   assert(n == MaxTurbines);
   assert(m == 2);

   // lower bound and upper bound on x
   for (Index i = 0; i < n; i++) {
	   x_l[i] = farmopt->yaw_lower; // x lower bound
	   x_u[i] = farmopt->yaw_upper; // x upper bound
	   double baseline_yaw = farmopt->turbine_chart[i].yaw_angle;
	   // restrict to +/- 8 deg from baseline and bounded in global bounds
	   x_l[i] = Max(farmopt->yaw_lower, baseline_yaw - 8.0);
	   x_u[i] = Min(farmopt->yaw_upper, baseline_yaw + 8.0);
   }

   // we have two inequality constraints. the lower bounds are zero
   g_l[0] = g1_lwr;
   g_l[1] = g2_lwr;
   g_u[0] = 1.0e19;
   g_u[1] = 1.0e19;

   return true;
}

bool MyNLP::get_starting_point(
   Index   n,
   bool    init_x,
   Number* x,
   bool    init_z,
   Number* z_L,
   Number* z_U,
   Index   m,
   bool    init_lambda,
   Number* lambda
)
{
   // Here, we assume we only have starting values for x, if you code
   // your own NLP, you can provide starting values for the others if
   // you wish.
   assert(init_x == true);
   assert(init_z == false);
   assert(init_lambda == false);

   // we initialize x in bounds, in the upper right quadrant
   Eigen::VectorXd yaw_vec = farmopt->getYawAngles();
   for (Index i = 0; i < n; i++) {
       x[i] = yaw_vec(i);
   }

   return true;
}

bool MyNLP::eval_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number&       obj_value
)
{
	// return the value of the objective function
	obj_value = -objPower(const_cast<Number*>(x)); // negative for maximization
   	return true;
}

bool MyNLP::eval_grad_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number*       grad_f
)
{
   // return the gradient of the objective function grad_{x} f(x)
    Eigen::VectorXd grad_f_vec = grad_f_all(x, n, this->pthreads);
    for (Index i = 0; i < n; i++) {
        grad_f[i] = grad_f_vec(i);
	}

   return true;
}

bool MyNLP::eval_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Number*       g
)
{
   // return the value of the constraints: g(x)
   //Number x1 = x[0];
   //Number x2 = x[1];

   //g[0] = -(x1 * x1 + x2 - 1.0);
   std::vector<double> lwrbnd = {g1_lwr, g2_lwr};
   Eigen::VectorXd g_vec = gval(x, lwrbnd);
   for (Index i = 0; i < m; i++) {
       g[i] = g_vec(i);
   }

   return true;
}

bool MyNLP::eval_jac_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Index         nele_jac,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
	std::vector<double> x_vec(MaxTurbines); // to be fixed
   if( values == nullptr )
   {
      // return the structure of the jacobian of the constraints
      // literally fill the 1d-idx into 2d grid (mxn)
      for (Index row = 0; row < m; row++) {
          for (Index col = 0; col < n; col++) {
              Index idx = row * n + col;
              iRow[idx] = row;
              jCol[idx] = col;
          }
	  }
   }
   else
   {
      // return the values of the jacobian of the constraints
       Eigen::MatrixXd grad_g_mat = grad_g_all(x, n);
	   Eigen::MatrixXd jacob_mat = grad_g_mat.transpose();
       // literally fill the 1d-idx into 2d grid (mxn)
       for (Index row = 0; row < m; row++) {
           for (Index col = 0; col < n; col++) {
               Index idx = row * n + col;
               values[idx] = jacob_mat(row, col); // note the transpose here
           }
	   }
   }

   return true;
}



void MyNLP::finalize_solution(
   SolverReturn               status,
   Index                      n,
   const Number*              x,
   const Number*              z_L,
   const Number*              z_U,
   Index                      m,
   const Number*              g,
   const Number*              lambda,
   Number                     obj_value,
   const IpoptData*           ip_data,
   IpoptCalculatedQuantities* ip_cq
)
{
   // here is where we would store the solution to variables, or write to a file, etc
   // so we could use the solution.
}


// private methods imported

double MyNLP::objPower(const Number* x) {
    // convert x to vector<double>
	std::vector<double> x_vec(MaxTurbines); // to be fixed
    for (Index i = 0; i < MaxTurbines; i++) {
        x_vec[i] = x[i];
    }
	// set yaw angles
	farmopt->setYawAngles(x_vec);
    double total_power = 0.0;
    int nt = static_cast<int>(this->farmopt->layout.rows());
    for (int i = 0; i < nt; i++) {
        total_power += this->farmopt->turbine_chart[i].getPower();
    }
    return total_power;
}

// g(x) (constriants) implementation can be added here if needed
Eigen::VectorXd MyNLP::gval(const Number* x, std::vector<double>& lwrbnd) {
    Eigen::VectorXd g_vec(2);
    std::vector<double> x_vec(MaxTurbines); // to be fixed
    for (Index i = 0; i < MaxTurbines; i++) {
        x_vec[i] = x[i];
    }
	farmopt->setYawAngles(x_vec);
    double total_power = 0.0;
    for (int idx : farmopt->qz_12) {
        total_power += farmopt->turbine_chart[idx - 1].getPower(); // idx assumed 1-based
    }
    g_vec(0) = total_power; // Farm12
    total_power = 0.0;
    for (int idx : farmopt->qz_3) {
        total_power += farmopt->turbine_chart[idx - 1].getPower();
    }
    g_vec(1) = total_power; // Farm3
    return g_vec;
}


double MyNLP::grad_f_i(const Number* x, int idx) {
	// center-differencing method
	WindFarmOptimization w = *farmopt; // make a copy to avoid modifying original
	double yaw_org = w.turbine_chart[idx - 1].yaw_angle;
	double h = 1; // small perturbation
	w.turbine_chart[idx - 1].yaw_angle = yaw_org + h;
	w.calculateWake();
	double pwr_plus = w.getFarmPower();
	w.turbine_chart[idx - 1].yaw_angle = yaw_org - h;
	w.calculateWake();
	double pwr_minus = w.getFarmPower();
	double grad_fi = (pwr_plus - pwr_minus) / (2 * h);
	return -grad_fi; // Note the negative sign for maximization
}


// grad_f
Eigen::VectorXd MyNLP::grad_f_all(const Number* x, int n, int pthreads) {
	omp_set_num_threads(this->pthreads);
	int nt = static_cast<int>(farmopt->layout.rows());
	// convert x to vector<double>
	std::vector<double> yaw_angles(n); // to be fixed
	for (Index i = 0; i < n; i++) {
		yaw_angles[i] = x[i];
	}
	farmopt->setYawAngles(yaw_angles);
	Eigen::VectorXd grad_f_vec(nt);
	// compute grad_f for each turbine, idx is 1-based
	// parallel computation to be implemented later
	std::vector<double> grad_f_vector(nt);
	#pragma omp parallel for
		for (int i = 0; i < nt; i++) {
			grad_f_vector[i] = grad_f_i(x, i + 1); // idx is 1-based
	}
	grad_f_vec = Eigen::Map<Eigen::VectorXd>(grad_f_vector.data(), grad_f_vector.size());
	return grad_f_vec;
}

Eigen::VectorXd MyNLP::grad_g_i(const Number* x, int idx) {
	// center-differencing method
	WindFarmOptimization w = *farmopt; // make a copy to avoid modifying original
	double yaw_org = w.turbine_chart[idx - 1].yaw_angle;
	double h = 0.1; // small perturbation
	w.turbine_chart[idx - 1].yaw_angle = yaw_org + h;
	w.calculateWake();
	Eigen::VectorXd g_plus(2);
	double total_power = 0.0;
	for (int idx : farmopt->qz_12) {
		total_power += w.turbine_chart[idx - 1].getPower(); // idx assumed 1-based
	}
	g_plus(0) = total_power;
	total_power = 0.0;
	for (int idx : farmopt->qz_3) {
		total_power += w.turbine_chart[idx - 1].getPower();
	}
	g_plus(1) = total_power;
	w.turbine_chart[idx - 1].yaw_angle = yaw_org - h;
	w.calculateWake();
	Eigen::VectorXd g_minus(2);
	total_power = 0.0;
	for (int idx : farmopt->qz_12) {
		total_power += w.turbine_chart[idx - 1].getPower(); // idx assumed 1-based
	}
	g_minus(0) = total_power;
	total_power = 0.0;
	for (int idx : farmopt->qz_3) {
		total_power += w.turbine_chart[idx - 1].getPower();
	}
	g_minus(1) = total_power;
	return -(g_plus - g_minus) / (2 * h); // Note the negative sign for maximization
}

// grad_g
Eigen::MatrixXd MyNLP::grad_g_all(const Number* x, int n) {
	omp_set_num_threads(this->pthreads);
	std::vector<double> yaw_angles(n); // to be fixed
	for (Index i = 0; i < n; i++) {
		yaw_angles[i] = x[i];
	}
	farmopt->setYawAngles(yaw_angles);
	int nt = static_cast<int>(farmopt->layout.rows());
	Eigen::MatrixXd grad_g_mat(nt, 2); // nt x 2 matrix, 2 indicates two constraints
	// compute grad_g for each turbine, idx is 1-based
	// parallel computation to be implemented later
#pragma omp parallel for
	for (int i = 0; i < nt; i++) {
		Eigen::VectorXd grad_gi = grad_g_i(x, i + 1); // 1x2 vector
		Eigen::RowVectorXd grad_gi_row(grad_gi);
		grad_g_mat.row(i) = grad_gi_row; // idx is 1-based
	}
	return grad_g_mat;
}

/**** Custom Function*****/

// Initialization function
bool initializeWindFarm() {
	if (g_farmopt == nullptr) {
		try {
			// Eigen::setNbThreads(1);
    		// 设置 MKL 线程数
    		// mkl_set_num_threads(15);
			int sqz_12 = 92;
			// manually set params for verification
			std::vector<int> t_qz_12(sqz_12); // 1-92
			std::iota(t_qz_12.begin(), t_qz_12.end(), 1);

			// read all segments using readMultiSegmentCSV
			SegmentsData segData = readMultiSegmentCSV("../data/config.csv");
			if (segData.states != 0) {
				std::cerr << "Error reading configuration CSV file, state: " << segData.states << std::endl;
				return false;
			}
			std::vector<std::vector<int>> t_qz_3 = segData.firstSegment[0];
			//std::vector<std::vector<int>> t_qz_3 = readCSVInt("../t_qz.csv");
			std::array<std::vector<int>, 3> t_qz = { t_qz_12, t_qz_3[0], t_qz_3[1] };

			// misc params read from csv
			std::vector<std::vector<double>> params = segData.otherSegments[0];
			//std::vector<std::vector<double>> params = readCSV("../params.csv");
			const std::vector<double> turbulence_sheer_veer = params[0];
			const std::vector<double> turbine_diameter_vector = params[1];
			const std::vector<double> turbine_hub_height_vector = params[2];
			const std::vector<double> rated_power_vector = params[3];
			const std::vector<double> life_total_vector = params[4];
			const std::vector<double> repair_c_vector = params[5];

			std::vector<double> fatigue(159, 0);
			const std::vector<double> fatigue_p = params[6];

			// Cp and Ct
			std::vector<std::vector<double>> PT = segData.otherSegments[1];
			//std::vector<std::vector<double>> PT = readCSV("../Cpt.csv");
			// x,y,z position(layout)
			std::vector<std::vector<double>> XYZ = segData.otherSegments[2];
			//std::vector<std::vector<double>> XYZ = readCSV("../xyz.csv");

			std::vector<std::vector<double>> yaw_vec = generateRandomPT(1, 159, -30.0, 30.0);
			double wind_speed = 10;
			double wind_direction = 60;

			// additional params: serial_coeff_all_val, status_all_val
			std::vector<std::vector<double>> serial_coeff_all_val(3, std::vector<double>(159, 0.99)); // suppose all 0.99 for simplicity
			std::vector<int> status_all_val(159, 14); // all active

            status_convert2bin(status_all_val);

            // Create static instance
			g_farmopt = new WindFarmOptimization(PT, t_qz,
				turbulence_sheer_veer, turbine_diameter_vector,
				turbine_hub_height_vector, rated_power_vector,
				life_total_vector, repair_c_vector,
				yaw_vec[0], fatigue, fatigue_p,
				serial_coeff_all_val, status_all_val,
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

void status_convert2bin(std::vector<int> &status_all_val)
{
    // convert original 0-34 status to binary active/inactive (14->1, others->0)
    for (size_t i = 0; i < status_all_val.size(); i++)
    {
        if (status_all_val[i] == 14)
            status_all_val[i] = 1;
        else
            status_all_val[i] = 0;
    }
}

bool optimizeWindFarm(
    const std::vector<double>& mwind_speed,        // 输入：风速（序列 5个） double 类
    const std::vector<double>& mwind_direction,    // 输入：方向（序列 5个）
    const std::vector<std::vector<double>>& mEER_turbines,		  // 输入：各风机mEER值
	const std::vector<double>& init_yaw_angles, // 输入：各风机初始偏航角度，对应风机编号见 文件wind farm layout
	const std::vector<int>& turbine_start, // 输入：各风机启停状态，1为启用，0为停用
	std::vector<double>& opt_yaw_angles, // 输出：各风机偏航角度，对应风机编号见 文件wind farm layout
	std::vector<double>& opt_power_turbines,   // 输出：全场功率
    double& opt_power_farm_12, // 输出：青州12功率
    double& opt_power_farm_3 // 输出：青州3功率
	)
	{
		if (!g_farmopt) {
		std::cerr << "WindFarm not initialized!" << std::endl;
		exit(-1);
	}
	// check args
	if (init_yaw_angles.size() != MaxTurbines) {
		std::cerr << "Yaw angle size not correct." << std::endl;
		// use default yaw angles
		exit(-2);
	}
	g_farmopt->wind_speed = mwind_speed[0];
	g_farmopt->wind_direction = mwind_direction[0];
	g_farmopt->setYawAngles(init_yaw_angles); // set initial yaw angles from the actual input


	// update status and mEER_turbines for all turbines
	for (size_t i = 0; i < MaxTurbines; i++) {
		g_farmopt->turbine_chart[i].status = turbine_start[i];
		g_farmopt->turbine_chart[i].serial_coeff = {mEER_turbines[0][i], mEER_turbines[1][i], mEER_turbines[2][i]};
	}
	g_farmopt->status_all = turbine_start;
	g_farmopt->calculateWake();
	std::cout << "Wind Farm Power: " << g_farmopt->getFarmPower() << " W" << std::endl;
	try {
		int pthreads_ = 16;
		SmartPtr<TNLP> mynlp = new MyNLP(g_farmopt, pthreads_);
		SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
		ApplicationReturnStatus status = app->Initialize();
		if (status != Solve_Succeeded) {
			std::cout << "*** Error during initialization!" << std::endl;
			return false;
		}

		// Set optimization options
		app->Options()->SetIntegerValue("print_level", 0);
		app->Options()->SetStringValue("linear_solver", "ma57");
		app->Options()->SetStringValue("linear_system_scaling", "none");
		app->Options()->SetStringValue("output_file", "ipopt_out.txt");
		app->Options()->SetStringValue("hessian_approximation", "limited-memory");
		app->Options()->SetIntegerValue("max_iter", 2);
		app->Options()->SetStringValue("sb", "yes"); // get rid of IPOPT banner

		status = app->OptimizeTNLP(mynlp);

		if (status == Solve_Succeeded || status == Maximum_Iterations_Exceeded) {
			Index iter_count = app->Statistics()->IterationCount();
			std::cout << "*** Problem solved in " << iter_count << " iterations!" << std::endl;

			// Number final_obj = app->Statistics()->FinalObjective();
			// std::cout << "*** Final objective value: " << final_obj << std::endl;
			Eigen::VectorXd yaw_eigen_new = g_farmopt->getYawAngles();
			std::vector<double> yaw_vec_new(yaw_eigen_new.data(), yaw_eigen_new.data() + yaw_eigen_new.size());
			opt_yaw_angles = yaw_vec_new;
			opt_power_farm_12 = g_farmopt->getFarmQingzhou12Power();
			opt_power_farm_3 = g_farmopt->getFarmQingzhou3Power();

			Eigen::VectorXd opt_power_all_eigen = g_farmopt->getTurbinesPower();
			opt_power_turbines = std::vector<double>(opt_power_all_eigen.data(), opt_power_all_eigen.data() + opt_power_all_eigen.size());
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


bool optimizeWindFarmCheck(
    const double wind_speed,           // 输入：环境风速, double 类型，单位为 m/s
    const double wind_direction,       // 输入：环境风方向, double 类型，范围0-360, 以正北方向为0，单位为度
    const std::vector<std::vector<double>>& mEER_turbines,    // 输入：全场每个风机的能效系数，3x159 double matrix类型
    const std::vector<double>& init_yaw_angles,    // 输入：各风机初始偏航角度，1x159 double vector 类型，单位为度
                                                   // 该vector的编号与现场风机布局图（青洲1-3风机布局图.pdf）的编号一致，vector的第i个元素对应风机布局图编号为i的风机
    const std::vector<int>& turbine_status, // 输入：各风机启停状态，1x159 int vector 类型
    double& opt_power_farm_12,                  // 输出：青洲12总的最大可发功率，double 类型，单位为W （用于判断电网友好模式下的发电方式）
    double& opt_power_farm_3                    // 输出：青洲3总的最大可发功率，double 类型，单位为W（用于判断电网友好模式下的发电方式）
    ){
		if (!g_farmopt) {
		std::cerr << "WindFarm not initialized!" << std::endl;
		exit(-1);
	}
	// check args
	if (init_yaw_angles.size() != MaxTurbines) {
		std::cerr << "Yaw angle size not correct." << std::endl;
		// use default yaw angles
		exit(-2);
	}
	g_farmopt->wind_speed = wind_speed;
	g_farmopt->wind_direction = wind_direction;
	g_farmopt->setYawAngles(init_yaw_angles); // set initial yaw angles from the actual input
	g_farmopt->calculateWake();
	std::cout << "Wind Farm Power: " << g_farmopt->getFarmPower() << " W" << std::endl;
	try {
		int pthreads_ = 16;
		SmartPtr<TNLP> mynlp = new MyNLP(g_farmopt, pthreads_);
		SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
		ApplicationReturnStatus status = app->Initialize();
		if (status != Solve_Succeeded) {
			std::cout << "*** Error during initialization!" << std::endl;
			return false;
		}

		// Set optimization options
		app->Options()->SetIntegerValue("print_level", 0);
		app->Options()->SetStringValue("linear_solver", "ma57");
		app->Options()->SetStringValue("linear_system_scaling", "none");
		app->Options()->SetStringValue("output_file", "ipopt_out.txt");
		app->Options()->SetStringValue("hessian_approximation", "limited-memory");
		app->Options()->SetIntegerValue("max_iter", 2);
		app->Options()->SetStringValue("sb", "yes"); // get rid of IPOPT banner

		status = app->OptimizeTNLP(mynlp);

		if (status == Solve_Succeeded || status == Maximum_Iterations_Exceeded) {
			Index iter_count = app->Statistics()->IterationCount();
			std::cout << "*** Problem solved in " << iter_count << " iterations!" << std::endl;

			// Number final_obj = app->Statistics()->FinalObjective();
			// std::cout << "*** Final objective value: " << final_obj << std::endl;
			opt_power_farm_12 = g_farmopt->getFarmQingzhou12Power();
			opt_power_farm_3 = g_farmopt->getFarmQingzhou3Power();
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


// Toolset functions
std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return data; // 返回空vector
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

std::vector<std::vector<int>> readCSVInt(const std::string& filename) {
    std::vector<std::vector<int>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return data; // 返回空vector
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stoi(cell));
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    file.close();
    return data;
}

SegmentsData readMultiSegmentCSV(const std::string& filename) {
    SegmentsData result;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
		result.states = -1; // indicate error
        return result;
    }

    std::string line;
    int count = 0;
    bool isFirstSegment = true;
    std::vector<std::vector<int>> firstSegmentData;
    std::vector<std::vector<double>> currentDoubleSegment;
    std::vector<std::vector<int>> currentIntSegment;

    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == '#') {
            count ++;
            // 分段标志，保存当前段数据
            if (isFirstSegment) {
                if (!currentIntSegment.empty()) {
                    result.firstSegment.push_back(currentIntSegment);
                    currentIntSegment.clear();
                }
            } else {
                if (!currentDoubleSegment.empty()) {
                    result.otherSegments.push_back(currentDoubleSegment);
                    currentDoubleSegment.clear();
                }
            }
            if (count > 1) {
                isFirstSegment = false;
            }
            continue;
        }

        if (isFirstSegment) {
            // 读取整数段
            std::stringstream ss(line);
            std::string cell;
            std::vector<int> row;
            bool emptyFieldEncountered = false;

            while (std::getline(ss, cell, ',')) {
                if (cell.empty()) {
                    emptyFieldEncountered = true;
                    break;
                }
                try {
                    int val = std::stoi(cell);
                    row.push_back(val);
                } catch (const std::exception& e) {
                    std::cerr << "整数转换错误: " << cell << std::endl;
					result.states = 1; // indicate error
					return result;
                }
            }
            if (!row.empty()) {
                currentIntSegment.push_back(row);
            }
        } else {
            // 读取double段
            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            bool emptyFieldEncountered = false;

            while (std::getline(ss, cell, ',')) {
                if (cell.empty()) {
                    emptyFieldEncountered = true;
                    break;
                }
                try {
                    double val = std::stod(cell);
                    row.push_back(val);
                } catch (const std::exception& e) {
                    std::cerr << "浮点转换错误: " << cell << std::endl;
					result.states = 2; // indicate error
					return result;
                }
            }
            if (!row.empty()) {
                currentDoubleSegment.push_back(row);
            }
        }
    }

    // 文件结束，保存最后一段
    
    if (!currentIntSegment.empty()) {
        result.firstSegment.push_back(currentIntSegment);
    }
    if (!currentDoubleSegment.empty()) {
        result.otherSegments.push_back(currentDoubleSegment);
    }
    

    file.close();
    return result;
}

std::vector<std::vector<double>> generateRandomPT(int m, int n, double lwr, double upr) {
    std::vector<std::vector<double>> PT(m, std::vector<double>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(lwr, upr);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            PT[i][j] = dis(gen);
        }
    }
    return PT;
}

// yaw_delta/yaw_limit examples
// -3, 3, 0.15,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
// -30, 30,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,