#include "wrappedfunc.hpp" // Eigen is included here
#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include <IpOptionsList.hpp>
#include <cassert>
#include <chrono>
#include <random>
#include "omp.h"
#include "unistd.h"
#include "IpTNLP.hpp"
#include "Farm.hpp"
#include "rapidcsv.h"

// global resource handles

WindFarmOptimization* g_farmopt = nullptr;
rapidcsv::Document* g_csv_doc = nullptr;
std::ofstream log_hdl; // global log file handle

// 全局表格数据：风向-风速-偏航角

struct SegmentsData {
    std::vector<std::vector<std::vector<int>>> firstSegment;    // 整数表(t_qz, 风场12/3风机编号)
    std::vector<std::vector<std::vector<double>>> otherSegments; // double表（风场参数表，功率/推力表，风机位置）
    int states = 0; // 读取状态标志
};

// 参数读取器（一次完成）
// 支持多段落读取，段落以#开头标识，后续可能扩展
// 第一段为整数表格，后续段落为 double 表格
// 返回结构体包含所有段落数据
SegmentsData readMultiSegmentCSV(const std::string& filename);

// mxn 随机数生成器
// 生成范围在[lwr, upr]之间的随机数
std::vector<std::vector<double>> generateRandomPT(int m, int n, double lwr, double upr);


// 双线性插值函数
// 输入：
//   x, y - 目标点坐标
//   x0, y0 - 左下角点坐标
//   x1, y1 - 右上角点坐标
//   k00, k10, k01, k11 - 四个邻近点对应的值
template<typename T>
T bilinearInterpolation(double x, double y,
                        double x0, double y0, double x1, double y1,
                        const T& k00, const T& k10, const T& k01, const T& k11);

// 	string to eigen vector	
Eigen::VectorXd stringToVectorXd(const std::string& str);
// status vector to binary conversion
void status_convert2bin(std::vector<int> &status_all_val);
// Helper function to find indices of elements in a vector
std::vector<int> find_indices(const std::vector<int> &vec, const std::vector<int> &targets);
using namespace Ipopt;

class MyNLP : public TNLP
{
public:

	//params
	int pthreads = 8; // number of threads for parallel computing
	double g1_lwr = 0.0; // constraint 1 lower bound
	double g2_lwr = 0.0; // constraint 2 lower bound
	std::vector<double> current_yaw; // current yaw angles
	int analytic_grad_flag = 0; // flag for analytic gradient, 0 - numerical, 1 - analytic
	int mode = 0; // optimization mode
	/** default constructor */
	MyNLP(WindFarmOptimization* farmopt_cpy, int _pthreads, 
		const std::vector<double>& current_yaw_val, int _analytic_grad_flag, int mode_);
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

	double objPower(const Number *x); // FarmPwr(yaw_angle)
	double objPowerLife(const Number* x); // FarmObj(yaw_angle)
	Eigen::VectorXd gval(const Number *x, std::vector<double> &lwrbnd); // constraint values
	double grad_f_i(const Number* x, int idx); // power gradient for i
	double grad_f_obj_i(const Number* x, int idx); // objective gradient for i
	double grad_f_i_analytic(const Number* x, int idx);
	double grad_f_obj_i_analytic(const Number* x, int idx);
	Eigen::VectorXd grad_f_all(const Number* x, int n, int pthreads);
	Eigen::VectorXd grad_f_obj_all(const Number* x, int idx, int pthreads); // objective gradient
	Eigen::VectorXd grad_g_i(const Number* x, int idx);
	// to be implemented
	Eigen::VectorXd grad_g_i_analytic(const Number* x, int idx);
	Eigen::MatrixXd grad_g_all(const Number* x, int n);
	double grad_single_analytic(const Number* x, int kIdx, 
		const Eigen::MatrixXi& wake_matrix, int mode=0);
	double grad_single_obj_analytic(const Number* x, int kIdx, 
		const Eigen::MatrixXi& wake_matrix, int mode=0);

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

	
};



// Impl
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

/* Constructor. */
MyNLP::MyNLP(WindFarmOptimization* farmopt_cpy, int _pthreads, 
	const std::vector<double>& current_yaw_val, int _analytic_grad_flag, int mode_ = 0)
	: farmopt(farmopt_cpy), pthreads(_pthreads), current_yaw(current_yaw_val), analytic_grad_flag(_analytic_grad_flag)
{
	farmopt->calculateWake(); // initial wake calculation
	g1_lwr = farmopt->getFarmQingzhou12Power();
	g2_lwr = farmopt->getFarmQingzhou3Power();
	this->mode = mode_;
	if (mode_ == 1)
		g1_lwr = 0.5*farmopt->getFarmQingzhou12Power(); // 50% of max power for tracking mode
	else if (mode_ == 2)
		g2_lwr = 0.5*farmopt->getFarmQingzhou3Power(); // 50% of max power for tracking mode
	else if (mode_ == 3)
	{
		g1_lwr = 0.5*farmopt->getFarmQingzhou12Power(); // 50% of max power for tracking mode
		g2_lwr = 0.5*farmopt->getFarmQingzhou3Power(); // 50% of max power for tracking mode
	}
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
   // sum of all elems in status
//    int active_turbines = 0;
//    for (int i = 0; i < farmopt->n_turbines; i++) {
// 	   if (farmopt->turbine_chart[i].status == 1) {
// 		   active_turbines += 1;
// 	   }
//    }
   n = MaxTurbines; // 根据activate的风机数量修改

   // m equality constraint,
   m = 2;

   // nonzeros in the jacobian
   nnz_jac_g = m * n; // for simplicity, the jacobian matrix is assumed to be dense

   //// and 2 nonzeros in the hessian of the lagrangian
   //// (one in the hessian of the objective for x2,
   ////  and one in the hessian of the constraints for x1)

   // We use the standard c index style for row/col entries
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
	   double baseline_yaw = this->current_yaw[i];
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
	double T = farmopt->turbine_chart[0].optimization_period; // optimization period
	// return the value of the objective function
	obj_value = -objPowerLife(const_cast<Number*>(x))*T; // negative for maximization
   	return true;
}

bool MyNLP::eval_grad_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number*       grad_f
)
{
	// set mkl threads
	mkl_set_dynamic(0);
	mkl_set_num_threads(2);  // set MKL threads for
   // return the gradient of the objective function grad_{x} f(x)
    Eigen::VectorXd grad_f_vec = grad_f_obj_all(x, n, this->pthreads);
	mkl_set_dynamic(1); // reset mkl threads to default (dynamic)
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
	  // set mkl threads
	   mkl_set_dynamic(0);
	   mkl_set_num_threads(2);  // set MKL threads for
       Eigen::MatrixXd grad_g_mat = grad_g_all(x, n);
	   mkl_set_dynamic(1); // reset mkl threads to default (dynamic)
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
   // let all status = 0 turbine remain the current yaw
   for (Index i = 0; i < n; i++) {
	   if (this->farmopt->turbine_chart[i].status == 0) {
		   this->farmopt->turbine_chart[i].yaw_angle = this->current_yaw[i];
	   }
   }
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
		if (this->farmopt->turbine_chart[i].status == 0) {
			continue;
		}
        total_power += this->farmopt->turbine_chart[i].getPower();
    }
    return total_power;
}

double MyNLP::objPowerLife(const Number* x) {
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
		if (this->farmopt->turbine_chart[i].status == 0) {
			continue;
		}
        total_power += this->farmopt->turbine_chart[i].getSingleTurbineObjective();
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
		if (farmopt->turbine_chart[idx - 1].status == 0) {
			continue;
		}
        total_power += farmopt->turbine_chart[idx - 1].getPower(); // idx assumed 1-based
    }
    g_vec(0) = total_power; // Farm12
    total_power = 0.0;
    for (int idx : farmopt->qz_3) {
		if (farmopt->turbine_chart[idx - 1].status == 0) {
			continue;
		}
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


double MyNLP::grad_f_obj_i(const Number* x, int idx) {
	// center-differencing method
	WindFarmOptimization w = *farmopt; // make a copy to avoid modifying original
	double yaw_org = w.turbine_chart[idx - 1].yaw_angle;
	double h = 1; // small perturbation
	w.turbine_chart[idx - 1].yaw_angle = yaw_org + h;
	w.calculateWake();
	double obj_plus = w.getFarmObj();
	w.turbine_chart[idx - 1].yaw_angle = yaw_org - h;
	w.calculateWake();
	double obj_minus = w.getFarmObj();
	double grad_fi = (obj_plus - obj_minus) / (2 * h);
	return -grad_fi; // Note the negative sign for maximization
}
// analytic?
// to be added


double MyNLP::grad_f_i_analytic(const Number* x, int idx){
	Eigen::MatrixXi wake_mat = this->farmopt->wake_matrix;
	int mode = 0;
	double grad_fi = grad_single_analytic(x, idx, wake_mat, mode); //this->farmopt->grad_p_gamma_single(idx, wake_mat, mode);
	return -grad_fi; // Note the negative sign for maximization
}

double MyNLP::grad_f_obj_i_analytic(const Number* x, int idx){
	Eigen::MatrixXi wake_mat = this->farmopt->wake_matrix;
	int mode = 0;
	double grad_fi = grad_single_obj_analytic(x, idx, wake_mat, mode); //this->farmopt->grad_p_gamma_single(idx, wake_mat, mode);
	return -grad_fi; // Note the negative sign for maximization
}


// grad_f
Eigen::VectorXd MyNLP::grad_f_all(const Number* x, int n, int pthreads) {
	omp_set_num_threads(this->pthreads);
	omp_set_nested(1);  // 允许嵌套并行
	
	// close mkl_dynamic
	mkl_set_dynamic(0);
	mkl_set_num_threads(4);  // set MKL threads
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
			if (farmopt->turbine_chart[i].status == 0) {
				grad_f_vector[i] = 0.0;
				continue;
			}
			if (analytic_grad_flag == 0)
				grad_f_vector[i] = grad_f_i(x, i + 1); // idx is 1-based
			else
				grad_f_vector[i] = grad_f_i_analytic(x, i + 1); // idx is 1-based
	}
	grad_f_vec = Eigen::Map<Eigen::VectorXd>(grad_f_vector.data(), grad_f_vector.size());
	// open mkl_dynamic
	mkl_set_dynamic(1); // reset mkl threads to default (dynamic)
	return grad_f_vec;
}

Eigen::VectorXd MyNLP::grad_f_obj_all(const Number* x, int idx, int pthreads) {
	omp_set_num_threads(this->pthreads);
	omp_set_nested(1);  // 允许嵌套并行
	
	// close mkl_dynamic
	mkl_set_dynamic(0);
	mkl_set_num_threads(4);  // set MKL threads
	int nt = static_cast<int>(farmopt->layout.rows());
	// convert x to vector<double>
	std::vector<double> yaw_angles(nt); // to be fixed
	for (Index i = 0; i < nt; i++) {
		yaw_angles[i] = x[i];
	}
	farmopt->setYawAngles(yaw_angles);
	Eigen::VectorXd grad_f_vec(nt);
	// compute grad_f for each turbine, idx is 1-based
	// parallel computation to be implemented later
	std::vector<double> grad_f_vector(nt);
	#pragma omp parallel for
		for (int i = 0; i < nt; i++) {
			if (farmopt->turbine_chart[i].status == 0) {
				grad_f_vector[i] = 0.0;
				continue;
			}
			if (analytic_grad_flag == 0)
				grad_f_vector[i] = grad_f_obj_i(x, i + 1); // idx is 1-based
			else
				grad_f_vector[i] = grad_f_obj_i_analytic(x, i + 1); // idx is 1-based
	}
	grad_f_vec = Eigen::Map<Eigen::VectorXd>(grad_f_vector.data(), grad_f_vector.size());
	// open mkl_dynamic
	mkl_set_dynamic(1); // reset mkl threads to default (dynamic)
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
	omp_set_nested(1);  // 允许嵌套并行
	// close dynamic mkl
	mkl_set_dynamic(0);
	mkl_set_num_threads(2);  // set MKL threads
	// convert x to vector<double>
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
		if (farmopt->turbine_chart[i].status == 0) {
			grad_g_mat.row(i).setZero();
			continue;
		}
		Eigen::VectorXd grad_gi = grad_g_i(x, i + 1); // 1x2 vector
		Eigen::RowVectorXd grad_gi_row(grad_gi);
		// grad_gi_row(1) = 0, grad_gi_row(2) = 0; // to be fixed
		grad_g_mat.row(i) = grad_gi_row; // idx is 1-based
	}
	// open dynamic mkl
	mkl_set_dynamic(1); // reset mkl threads to default (dynamic)
	return grad_g_mat;
}


double MyNLP::grad_single_analytic(const Number* x, int kIdx, 
	const Eigen::MatrixXi& wake_matrix, int mode)
{
	// 1. Prepare static cache if not already done
	if (!(farmopt->cache_stored))
	{
		farmopt->rotated_result_cache = farmopt->prepareStaticCache();
	}
	const RotatedResult &rot = farmopt->rotated_result_cache;

	// 2. Find turbines affected by kIdx
	// if mode = 0: pick all affected idx
	// if mode = 1: pick idx in qz_12 only
	// if mode = 2: pick idx in qz_3 only

	// set two flags to show whether kIdx or affected_idx is empty according to the target mode
	int empty_flag_i = 0;
	int empty_flag_affected_idx = 0;

	// hard-coded judgement for kIdx
	if (kIdx < 92 && mode == 2)
		empty_flag_i = 1;
	else if (kIdx >= 92 && mode == 1)
		empty_flag_i = 1;

	std::vector<int> affected_indices;
	for (int i = 0; i < wake_matrix.cols(); ++i)
	{
		if (wake_matrix(kIdx - 1, i) == 1)
		{
			if(mode == 0)
				affected_indices.push_back(i);
			else if(mode == 1)
			{
				// check if i in qz_12
				if (std::find(farmopt->qz_12.begin(), farmopt->qz_12.end(), i + 1) != farmopt->qz_12.end()) // i+1 for 1-based
					affected_indices.push_back(i);
			}
			else if(mode == 2)
			{
				// check if i in qz_3
				if (std::find(farmopt->qz_3.begin(), farmopt->qz_3.end(), i + 1) != farmopt->qz_3.end()) // i+1 for 1-based
					affected_indices.push_back(i);
			}
		}
	}

	// 3. Map original kIdx to its sorted position
	auto it_k = std::find(rot.sorted_indexes.begin(), rot.sorted_indexes.end(), kIdx); // find kidx in 1 to 159
	if (it_k == rot.sorted_indexes.end())
	{
		// Handle error: kIdx not found in sorted indices
		return -1;
	}
	int kIdx_rot = std::distance(rot.sorted_indexes.begin(), it_k) + 1; // get kIdx_rot from ptr (also 1 based)

	// 4. Get parameters for the current turbine (k)
	const Turbine &k_turbine = farmopt->turbine_chart[kIdx_rot - 1]; // new kIdx is idx in rotated_turb_chart
	double D = k_turbine.rotor_diameter;
	double Ct = k_turbine.getCt();
	double gamma_deg = k_turbine.yaw_angle;
	double gamma_rad = gamma_deg * M_PI / 180.0;
	double U0 = farmopt->wind_speed;
	double p = 3.0; // Power exponent for yaw loss

	double P0 = k_turbine.getPower() / std::pow(cos(gamma_rad), p);

	// 5. Calculate derivatives related to yaw
	double c = cos(gamma_rad);
	double s = sin(gamma_rad);
	double du0 = (1.0 - sqrt(1.0 - Ct * c * c));
	double dv0 = 0.25 * Ct * c * c * s;
	double ddu0 = -(Ct * c * s) / sqrt(1.0 - Ct * c * c);
	double ddv0 = 0.25 * Ct * c * (3.0 * c * c - 2.0);

	// 6. Calculate the gradient term from the turbine itself
	// note if i is not in the target: just set to 0
	double grad_k_rad = 0.0;

	if(empty_flag_i == 0)
		grad_k_rad += (3.0 * P0 / U0) * (ddu0 * du0 + ddv0 * dv0);

	if (affected_indices.empty())
		return grad_k_rad * (M_PI / 180.0);

	// 7. Prepare data for downstream turbines
	Eigen::VectorXd Xsep(affected_indices.size());
	Eigen::VectorXd Ysep(affected_indices.size());
	Eigen::VectorXd Zsep(affected_indices.size());
	Eigen::VectorXd Fl(affected_indices.size());
	Eigen::VectorXd inv2l(affected_indices.size());
	Eigen::VectorXd Gl(affected_indices.size());
	Eigen::VectorXd Ai(affected_indices.size());

	// Get the indices within the y_model that correspond to the affected turbines
	std::vector<int> y_model_mask_indices;
	for (int i = 0; i < rot.idx_mask[kIdx_rot - 1].size(); ++i)
	{
		if (rot.idx_mask[kIdx_rot - 1][i] == 1)
		{
			y_model_mask_indices.push_back(i);
		}
	}
	std::vector<int> gl_lookup_indices = find_indices(y_model_mask_indices, affected_indices);

	int turbine_type_idx = 0;
	if (D == 228)
		turbine_type_idx = 0;
	else if (D == 158)
		turbine_type_idx = 1;
	else
		turbine_type_idx = 2;

	for (size_t i = 0; i < affected_indices.size(); ++i)
	{
		int aff_idx = affected_indices[i];
		const Turbine &i_turbine = farmopt->turbine_chart[aff_idx];

		Xsep(i) = rot.x_d[kIdx_rot - 1](aff_idx);
		Ysep(i) = rot.y_d[kIdx_rot - 1](aff_idx);
		Zsep(i) = rot.z_d[kIdx_rot - 1](aff_idx);
		Fl(i) = rot.delta_u[kIdx_rot - 1](aff_idx);
		inv2l(i) = 1.0 / rot.sigma_square[kIdx_rot - 1](aff_idx);

		if (i < gl_lookup_indices.size())
		{
			Gl(i) = rot.y_model[kIdx_rot - 1][turbine_type_idx](gl_lookup_indices[i]);
		}
		else
		{
			Gl(i) = 0; // Should not happen if logic is correct
		}

		double Ui = i_turbine.getAverageVelocity();
		double Pi = i_turbine.getPower();
		Ai(i) = (Ui > 1e-4) ? (3.0 * Pi / Ui) : 0.0;
	}

	// 8. Calculate the influence on downstream turbines
	double C0 = rot.coeff[kIdx_rot - 1];
	Eigen::VectorXd Yl = Ysep.array() + dv0 * Gl.array();
	Eigen::VectorXd Zl = Zsep;

	Eigen::VectorXd El = (-(Yl.array().square() + Zl.array().square()) * inv2l.array()).exp();

	Eigen::VectorXd termUk = -U0 * C0 * El.array() * Fl.array() *
							 (ddu0 - du0 * (Yl.array() / (2.0 * inv2l.array().inverse())) * ddv0 * Gl.array());

	// 9. Sum gradients and convert to degrees
	grad_k_rad += (Ai.array() * termUk.array()).sum();

	return grad_k_rad * (M_PI / 180.0);
}


double MyNLP::grad_single_obj_analytic(const Number* x, int kIdx, 
	const Eigen::MatrixXi& wake_matrix, int mode)
{
		// 1. Prepare static cache if not already done
	if (!(farmopt->cache_stored))
	{
		farmopt->rotated_result_cache = farmopt->prepareStaticCache();
	}
	const RotatedResult &rot = farmopt->rotated_result_cache;

	// 2. Find turbines affected by kIdx
	// if mode = 0: pick all affected idx
	// if mode = 1: pick idx in qz_12 only
	// if mode = 2: pick idx in qz_3 only

	// set two flags to show whether kIdx or affected_idx is empty according to the target mode
	int empty_flag_i = 0;
	int empty_flag_affected_idx = 0;

	// hard-coded judgement for kIdx
	if (kIdx < 92 && mode == 2)
		empty_flag_i = 1;
	else if (kIdx >= 92 && mode == 1)
		empty_flag_i = 1;

	std::vector<int> affected_indices;
	for (int i = 0; i < wake_matrix.cols(); ++i)
	{
		if (wake_matrix(kIdx - 1, i) == 1)
		{
			if(mode == 0)
				affected_indices.push_back(i);
			else if(mode == 1)
			{
				// check if i in qz_12
				if (std::find(farmopt->qz_12.begin(), farmopt->qz_12.end(), i + 1) != farmopt->qz_12.end()) // i+1 for 1-based
					affected_indices.push_back(i);
			}
			else if(mode == 2)
			{
				// check if i in qz_3
				if (std::find(farmopt->qz_3.begin(), farmopt->qz_3.end(), i + 1) != farmopt->qz_3.end()) // i+1 for 1-based
					affected_indices.push_back(i);
			}
		}
	}

	// 3. Map original kIdx to its sorted position
	auto it_k = std::find(rot.sorted_indexes.begin(), rot.sorted_indexes.end(), kIdx); // find kidx in 1 to 159
	if (it_k == rot.sorted_indexes.end())
	{
		// Handle error: kIdx not found in sorted indices
		return -1;
	}
	int kIdx_rot = std::distance(rot.sorted_indexes.begin(), it_k) + 1; // get kIdx_rot from ptr (also 1 based)

	// 4. Get parameters for the current turbine (k)
	const Turbine &k_turbine = farmopt->turbine_chart[kIdx_rot - 1]; // new kIdx is idx in rotated_turb_chart
	double D = k_turbine.rotor_diameter;
	double Ct = k_turbine.getCt();
	double gamma_deg = k_turbine.yaw_angle;
	double gamma_rad = gamma_deg * M_PI / 180.0;
	double U0 = farmopt->wind_speed;
	double p = 3.0; // Power exponent for yaw loss

	double P0 = k_turbine.getPower() / std::pow(cos(gamma_rad), p);

	// 5. Calculate derivatives related to yaw
	double c = cos(gamma_rad);
	double s = sin(gamma_rad);
	double du0 = (1.0 - sqrt(1.0 - Ct * c * c));
	double dv0 = 0.25 * Ct * c * c * s;
	double ddu0 = -(Ct * c * s) / sqrt(1.0 - Ct * c * c);
	double ddv0 = 0.25 * Ct * c * (3.0 * c * c - 2.0);

	// 6. Calculate the gradient term from the turbine itself
	// note if i is not in the target: just set to 0
	double grad_k_rad = 0.0;

	if(empty_flag_i == 0)
		grad_k_rad += (3.0 * P0 / U0) * (ddu0 * du0 + ddv0 * dv0);

	if (affected_indices.empty())
		return grad_k_rad * (M_PI / 180.0);

	// 7. Prepare data for downstream turbines
	Eigen::VectorXd Xsep(affected_indices.size());
	Eigen::VectorXd Ysep(affected_indices.size());
	Eigen::VectorXd Zsep(affected_indices.size());
	Eigen::VectorXd Fl(affected_indices.size());
	Eigen::VectorXd inv2l(affected_indices.size());
	Eigen::VectorXd Gl(affected_indices.size());
	Eigen::VectorXd Ai(affected_indices.size());
	// added life terms
	Eigen::VectorXd pavg_vector(affected_indices.size());
    // Eigen::VectorXd life_turbulence_c(affected_indices.size());
    Eigen::VectorXd life_work_c(affected_indices.size());

	// Get the indices within the y_model that correspond to the affected turbines
	std::vector<int> y_model_mask_indices;
	for (int i = 0; i < rot.idx_mask[kIdx_rot - 1].size(); ++i)
	{
		if (rot.idx_mask[kIdx_rot - 1][i] == 1)
		{
			y_model_mask_indices.push_back(i);
		}
	}
	std::vector<int> gl_lookup_indices = find_indices(y_model_mask_indices, affected_indices);

	int turbine_type_idx = 0;
	if (D == 228)
		turbine_type_idx = 0;
	else if (D == 158)
		turbine_type_idx = 1;
	else
		turbine_type_idx = 2;

	for (size_t i = 0; i < affected_indices.size(); ++i)
	{
		int aff_idx = affected_indices[i];
		const Turbine &i_turbine = farmopt->turbine_chart[aff_idx];

		Xsep(i) = rot.x_d[kIdx_rot - 1](aff_idx);
		Ysep(i) = rot.y_d[kIdx_rot - 1](aff_idx);
		Zsep(i) = rot.z_d[kIdx_rot - 1](aff_idx);
		Fl(i) = rot.delta_u[kIdx_rot - 1](aff_idx);
		inv2l(i) = 1.0 / rot.sigma_square[kIdx_rot - 1](aff_idx);

		if (i < gl_lookup_indices.size())
		{
			Gl(i) = rot.y_model[kIdx_rot - 1][turbine_type_idx](gl_lookup_indices[i]);
		}
		else
		{
			Gl(i) = 0; // Should not happen if logic is correct
		}

		double Ui = i_turbine.getAverageVelocity();
		double Pi = i_turbine.getPower();
		Ai(i) = (Ui > 1e-4) ? (3.0 * Pi / Ui) : 0.0;
		pavg_vector(i) = i_turbine.annual_average_power;
        // life_turbulence_c(i) = i_turbine.life_turbulence_coeff;
        life_work_c(i) = i_turbine.life_work_coeff;
	}

	// 8. Calculate the influence on downstream turbines
	double C0 = rot.coeff[kIdx_rot - 1];
	Eigen::VectorXd Yl = Ysep.array() + dv0 * Gl.array();
	Eigen::VectorXd Zl = Zsep;

	Eigen::VectorXd El = (-(Yl.array().square() + Zl.array().square()) * inv2l.array()).exp();

	Eigen::VectorXd termUk = -U0 * C0 * El.array() * Fl.array() *
							 (ddu0 - du0 * (Yl.array() / (2.0 * inv2l.array().inverse())) * ddv0 * Gl.array());

	// 9. Sum gradients and convert to degrees
	grad_k_rad += (Ai.array() * termUk.array()).sum();

	// 10. compute life term gradient
	double T = farmopt->turbine_chart[kIdx_rot - 1].optimization_period;

    // --- Work life gradient ---
    double pavg_k = k_turbine.annual_average_power;
    double life_work_k = k_turbine.life_work_coeff;
    double grad_k_power_only = -p * (P0 / std::pow(c, p)) * (p * std::pow(c, p - 1) * (-s)); // d(P_k)/d(gamma_k)
    double life_work_grad_k = -T * (
        pavg_k * life_work_k * grad_k_power_only +
        (pavg_vector.array() * life_work_c.array() * Ai.array() * termUk.array()).sum()
    );
	grad_k_rad = T * grad_k_rad;
	grad_k_rad += life_work_grad_k;
	return grad_k_rad * (M_PI / 180.0);
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

			// init table
			std::vector<double> yaw_vec(159, 30.0);
			//std::vector<std::vector<double>> yaw_vec = generateRandomPT(1, 159, -30.0, 30.0);
			double wind_speed = 10;
			double wind_direction = 60;

			// additional params: serial_coeff_all_val, status_all_val
			std::vector<std::vector<double>> serial_coeff_all_val(3, std::vector<double>(159, 1)); // suppose all 0.99 for simplicity
			std::vector<int> status_all_val(159, 14); // all active

            status_convert2bin(status_all_val);

			// read init_table to the null global csv document
			// have col headers and no row headers
			g_csv_doc = new rapidcsv::Document("../data/init_table.csv", rapidcsv::LabelParams(0, -1));
			// verify read (print all colNames)
			// std::vector<std::string> colNames = g_csv_doc->GetColumnNames();
			// for (const auto& name : colNames) {
			// 	std::cout << "Column: " << name << std::endl;
			// }

            // Create static instance
			g_farmopt = new WindFarmOptimization(PT, t_qz,
				turbulence_sheer_veer, turbine_diameter_vector,
				turbine_hub_height_vector, rated_power_vector,
				life_total_vector, repair_c_vector,
				yaw_vec, fatigue, fatigue_p,
				serial_coeff_all_val, status_all_val,
				XYZ, wind_speed, wind_direction);
			

			// create loginfo folder in src if not exists
			std::string log_folder = "../loginfo";
			std::string mkdir_cmd = "mkdir -p " + log_folder;
			system(mkdir_cmd.c_str());
			// create log file for debugging
			// set log file name using current date and time
			std::string log_filename = "optlog";
			// concatenate date time to log filename
			// change to yy-hh-dd-hh:mm:ss format
			auto logstart_datetime = 
				std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() + std::chrono::hours(8));
            std::tm* logstart_tm = std::localtime(&logstart_datetime);
            std::ostringstream datetime_ss;
            datetime_ss << std::put_time(logstart_tm, "%y-%m-%d-%H:%M:%S");
			log_filename = log_folder + "/" + "optlog_" + datetime_ss.str() + ".log";
            // create global file handle 
            log_hdl = std::ofstream(log_filename, std::ios::app);
			return true;
			
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
	const std::vector<int>& turbine_status, // 输入：各风机启停状态，1为启用，0为停用
	const int analytic_grad_flag, // 输入：是否使用解析梯度，0-否，1-是
	std::vector<double>& opt_yaw_angles, // 输出：各风机偏航角度，对应风机编号见 文件wind farm layout
	std::vector<double>& opt_power_turbines,   // 输出：全场功率
    double& opt_power_farm_12, // 输出：青州12功率
    double& opt_power_farm_3, // 输出：青州3功率
	int mode = 0) // 输入：优化模式，0-全场，1-青州12，2-青州3)
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
	g_farmopt->calculateWake(); // calculate initial wake
	sleep(2); // wait for 2 seconds to ensure wake calculation is done
	double qingzhou12_pwr = g_farmopt->getFarmQingzhou12Power();
	double qingzhou3_pwr = g_farmopt->getFarmQingzhou3Power();
	if (mode == 1){
		qingzhou12_pwr *= 0.5;
	}
	else if (mode == 2){
		qingzhou3_pwr *= 0.5;
	}
	else if (mode == 3){
		qingzhou12_pwr *= 0.5;
		qingzhou3_pwr *= 0.5;
	}
	// 返回单位MW并有两位小数
	qingzhou12_pwr = std::round(qingzhou12_pwr / 1e6 * 100) / 100.0;
	qingzhou3_pwr = std::round(qingzhou3_pwr / 1e6 * 100) / 100.0;
	std::cout << "青州12AGC指令: " << qingzhou12_pwr << "MW， 青州3AGC指令: " << qingzhou3_pwr << "MW" << std::endl;

	// test grad_f function in mynlp

	// set initial yaw value for optimization within limits (delta_yaw +/- 8 deg, yaw +/- yaw_bounds)
	std::vector<double> wind_direction_tbl = {30, 45, 100, 145, 190, 225, 280, 325};
	std::vector<double> wind_speed_tbl = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

	// find the two nearest wind directions and speeds for bilinear interpolation
	double wd = g_farmopt->wind_direction;
	double ws = g_farmopt->wind_speed;
	double wd0 = 0.0, wd1 = 0.0;
	double ws0 = 0.0, ws1 = 0.0;
	int i = 0, j = 0; // save indices
	for (i = 0; i < wind_direction_tbl.size() - 1; i++) {
		if (wd < wind_direction_tbl[0]) {
			wd0 = wd;
			wd1 = wind_direction_tbl[1];
			break;
		}
		else if (wd > wind_direction_tbl[wind_direction_tbl.size() - 1]) {
			wd0 = wind_direction_tbl[wind_direction_tbl.size() - 1];
			wd1 = wd;
			break;
		}
		if (wd >= wind_direction_tbl[i] && wd <= wind_direction_tbl[i + 1]) {
			wd0 = wind_direction_tbl[i];
			wd1 = wind_direction_tbl[i + 1];
			break;
		}
	}
	for (j = 0; j < wind_speed_tbl.size() - 1; j++) {
		if (ws < wind_speed_tbl[0]) {
			ws0 = ws;
			ws1 = wind_speed_tbl[1];
			break;
		}
		else if (ws > wind_speed_tbl[wind_speed_tbl.size() - 1]) {
			ws0 = wind_speed_tbl[wind_speed_tbl.size() - 2];
			ws1 = ws;
			break;
		}
		if (ws >= wind_speed_tbl[j] && ws <= wind_speed_tbl[j + 1]) {
			ws0 = wind_speed_tbl[j];
			ws1 = wind_speed_tbl[j + 1];
			break;
		}
	}
	// if the wd/ws out of the bound, just set to original
	if (wd < wind_direction_tbl[0] || wd > wind_direction_tbl[wind_direction_tbl.size() - 1] ||
		ws < wind_speed_tbl[0] || ws > wind_speed_tbl[wind_speed_tbl.size() - 1]) {
		// set initial yaw directly	
		g_farmopt->setYawAngles(std::vector<double>(MaxTurbines, 5.0));
	}
	else{
		// get yaw angles at the four corners
		std::string yaw_angle_str_ij = g_csv_doc->GetCell<std::string>(24, j*wind_direction_tbl.size() + i); // row 11 (yaw), col based on ws, wd indices
		std::string yaw_angle_str_i1j = g_csv_doc->GetCell<std::string>(24, j*wind_direction_tbl.size() + (i + 1));
		std::string yaw_angle_str_ij1 = g_csv_doc->GetCell<std::string>(24, (j + 1)*wind_direction_tbl.size() + i);
		std::string yaw_angle_str_i1j1 = g_csv_doc->GetCell<std::string>(24, (j + 1)*wind_direction_tbl.size() + (i + 1));
		// use bilinear interpolation to get init yaw for each turbine
		// convert string to vec
		Eigen::VectorXd yaw_angle_ij = stringToVectorXd(yaw_angle_str_ij);
		Eigen::VectorXd yaw_angle_i1j = stringToVectorXd(yaw_angle_str_i1j);
		Eigen::VectorXd yaw_angle_ij1 = stringToVectorXd(yaw_angle_str_ij1);
		Eigen::VectorXd yaw_angle_i1j1 = stringToVectorXd(yaw_angle_str_i1j1);
		// bilinear interpolation for new init yaw
		Eigen::VectorXd yaw_angle_interp2d = bilinearInterpolation<Eigen::VectorXd>(
			wd, ws,
			wd0, ws0, wd1, ws1,
			yaw_angle_ij, yaw_angle_i1j,
			yaw_angle_ij1, yaw_angle_i1j1
		);

		std::vector<double> yaw_vec_interp2d(yaw_angle_interp2d.data(), yaw_angle_interp2d.data() + yaw_angle_interp2d.size());
		g_farmopt->setYawAngles(yaw_vec_interp2d); // set initial yaw angles from the bilinear interpolation
	}
	
	// limit the interpolation in the limit
	for (size_t k = 0; k < MaxTurbines; k++) {
		double baseline_yaw = g_farmopt->turbine_chart[k].yaw_angle;
		// restrict to +/- 8 deg from baseline and bounded in global bounds
		if (g_farmopt->turbine_chart[k].yaw_angle < Max(g_farmopt->yaw_lower, baseline_yaw - 8.0))
			g_farmopt->turbine_chart[k].yaw_angle = Max(g_farmopt->yaw_lower, baseline_yaw - 8.0);
		else if (g_farmopt->turbine_chart[k].yaw_angle > Min(g_farmopt->yaw_upper, baseline_yaw + 8.0))
			g_farmopt->turbine_chart[k].yaw_angle = Min(g_farmopt->yaw_upper, baseline_yaw + 8.0);
	}
	// update status and mEER_turbines for all turbines
	for (size_t i = 0; i < MaxTurbines; i++) {
		g_farmopt->turbine_chart[i].status = turbine_status[i];
		g_farmopt->turbine_chart[i].serial_coeff = {mEER_turbines[0][i], mEER_turbines[1][i], mEER_turbines[2][i]};
	}
	g_farmopt->status_all = turbine_status;
	g_farmopt->calculateWake(); // recalculate wake after status update
	g_farmopt->calculate_wake_matrix(); // recalculate wake matrix after status update
	
	// log params and time to file
	// get current time
	auto log_start_time = std::chrono::high_resolution_clock::now();
	auto log_start_datetime = std::chrono::system_clock::to_time_t(log_start_time);
	try {
		int pthreads_ = 32;
		SmartPtr<TNLP> mynlp = new MyNLP(g_farmopt, pthreads_, init_yaw_angles, analytic_grad_flag, mode);
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
		app->Options()->SetIntegerValue("max_iter", 3);
		app->Options()->SetStringValue("sb", "yes"); // get rid of IPOPT banner

		status = app->OptimizeTNLP(mynlp);

		if (status == Solve_Succeeded || status == Maximum_Iterations_Exceeded) {
			// log end time
			auto log_end_time = std::chrono::high_resolution_clock::now();
			auto log_end_datetime = std::chrono::system_clock::to_time_t(log_end_time);
			std::chrono::duration<double> elapsed_time = log_end_time - log_start_time;
			Index iter_count = app->Statistics()->IterationCount();
			// std::cout << "*** 优化完成，总计 " << iter_count << " 次迭代" << std::endl;
			// std::cout << "Wind Farm Power: " << g_farmopt->getFarmPower() << " W" << std::endl;
			// Number final_obj = app->Statistics()->FinalObjective();
			// std::cout << "*** Final objective value: " << final_obj << std::endl;
			Eigen::VectorXd yaw_eigen_new = g_farmopt->getYawAngles();
			std::vector<double> yaw_vec_new(yaw_eigen_new.data(), yaw_eigen_new.data() + yaw_eigen_new.size());
			opt_yaw_angles = yaw_vec_new;
			opt_power_farm_12 = g_farmopt->getFarmQingzhou12Power();
			opt_power_farm_3 = g_farmopt->getFarmQingzhou3Power();

			Eigen::VectorXd opt_power_all_eigen = g_farmopt->getTurbinesPower();
			opt_power_turbines = std::vector<double>(opt_power_all_eigen.data(), opt_power_all_eigen.data() + opt_power_all_eigen.size());
			double opt_power_farm_all = g_farmopt->getFarmPower();
			// logger section
			// log time and opt power12/3/all to optlog.txt

			log_hdl << "----------------------------------------" << std::endl;
			// log start time (as date format yy-mm-dd-hh-mm-ss)
			log_hdl << "Optimization Start Time: " << std::put_time(std::localtime(&log_start_datetime), "%Y-%m-%d %H:%M:%S") << std::endl;
			log_hdl << "Optimization Iterations: " << iter_count << ", ";
			log_hdl << "Optimization Time: " << elapsed_time.count() << " seconds, ";
			log_hdl << "Qingzhou12 Power: " << opt_power_farm_12 << " W, ";
			log_hdl << "Qingzhou3 Power: " << opt_power_farm_3 << " W, ";
			log_hdl << "Total Farm Power: " << opt_power_farm_all << " W" << std::endl;
			// log end time (same format)
			log_hdl << "Optimization End Time: " << std::put_time(std::localtime(&log_end_datetime), "%Y-%m-%d %H:%M:%S") << std::endl;
			log_hdl << "----------------------------------------" << std::endl;
			return true;
		}
		else {
			std::cout << "*** 优化失败，状态码: " << status << std::endl;
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
    const int analytic_grad_flag,               // 输入：是否使用解析梯度，int 类型，0-否，1-是
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

	std::vector<double> wind_direction_tbl = {30, 45, 100, 145, 190, 225, 280, 325};
	std::vector<double> wind_speed_tbl = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

	// find the two nearest wind directions and speeds for bilinear interpolation
	double wd = g_farmopt->wind_direction;
	double ws = g_farmopt->wind_speed;
	double wd0 = 0.0, wd1 = 0.0;
	double ws0 = 0.0, ws1 = 0.0;
	int i = 0, j = 0; // save indices
	for (i = 0; i < wind_direction_tbl.size() - 1; i++) {
		if (wd >= wind_direction_tbl[i] && wd <= wind_direction_tbl[i + 1]) {
			wd0 = wind_direction_tbl[i];
			wd1 = wind_direction_tbl[i + 1];
			break;
		}
	}
	for (j = 0; j < wind_speed_tbl.size() - 1; j++) {
		if (ws >= wind_speed_tbl[j] && ws <= wind_speed_tbl[j + 1]) {
			ws0 = wind_speed_tbl[j];
			ws1 = wind_speed_tbl[j + 1];
			break;
		}
	}

	std::string yaw_angle_str_ij = g_csv_doc->GetCell<std::string>(24, j*wind_direction_tbl.size() + i); // row 11 (yaw), col based on ws, wd indices
	std::string yaw_angle_str_i1j = g_csv_doc->GetCell<std::string>(24, j*wind_direction_tbl.size() + (i + 1));
	std::string yaw_angle_str_ij1 = g_csv_doc->GetCell<std::string>(24, (j + 1)*wind_direction_tbl.size() + i);
	std::string yaw_angle_str_i1j1 = g_csv_doc->GetCell<std::string>(24, (j + 1)*wind_direction_tbl.size() + (i + 1));
	// use bilinear interpolation to get init yaw for each turbine
	// convert string to vec
	Eigen::VectorXd yaw_angle_ij = stringToVectorXd(yaw_angle_str_ij);
	Eigen::VectorXd yaw_angle_i1j = stringToVectorXd(yaw_angle_str_i1j);
	Eigen::VectorXd yaw_angle_ij1 = stringToVectorXd(yaw_angle_str_ij1);
	Eigen::VectorXd yaw_angle_i1j1 = stringToVectorXd(yaw_angle_str_i1j1);
	// bilinear interpolation for new init yaw
	Eigen::VectorXd yaw_angle_interp2d = bilinearInterpolation<Eigen::VectorXd>(
		wd, ws,
		wd0, ws0, wd1, ws1,
		yaw_angle_ij, yaw_angle_i1j,
		yaw_angle_ij1, yaw_angle_i1j1
	);

	std::vector<double> yaw_vec_interp2d(yaw_angle_interp2d.data(), yaw_angle_interp2d.data() + yaw_angle_interp2d.size());
	g_farmopt->setYawAngles(yaw_vec_interp2d); // set initial yaw angles from the bilinear interpolation
	// limit the interpolation in the limit
	for (size_t k = 0; k < MaxTurbines; k++) {
		double baseline_yaw = g_farmopt->turbine_chart[k].yaw_angle;
		// restrict to +/- 8 deg from baseline and bounded in global bounds
		if (g_farmopt->turbine_chart[k].yaw_angle < Max(g_farmopt->yaw_lower, baseline_yaw - 8.0))
			g_farmopt->turbine_chart[k].yaw_angle = Max(g_farmopt->yaw_lower, baseline_yaw - 8.0);
		else if (g_farmopt->turbine_chart[k].yaw_angle > Min(g_farmopt->yaw_upper, baseline_yaw + 8.0))
			g_farmopt->turbine_chart[k].yaw_angle = Min(g_farmopt->yaw_upper, baseline_yaw + 8.0);
	}
	// update status and mEER_turbines for all turbines
	for (size_t i = 0; i < MaxTurbines; i++) {
		g_farmopt->turbine_chart[i].status = turbine_status[i];
		g_farmopt->turbine_chart[i].serial_coeff = {mEER_turbines[0][i], mEER_turbines[1][i], mEER_turbines[2][i]};
	}
	g_farmopt->status_all = turbine_status;
	g_farmopt->calculateWake(); // recalculate wake after status update
	g_farmopt->calculate_wake_matrix(); // recalculate wake matrix after status update


	try {
		int pthreads_ = 16;
		SmartPtr<TNLP> mynlp = new MyNLP(g_farmopt, pthreads_, init_yaw_angles, analytic_grad_flag);
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
	// close log handle
	log_hdl.close();

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


template<typename T>
T bilinearInterpolation(double x, double y,
                        double x0, double y0, double x1, double y1,
                        const T& k00, const T& k10, const T& k01, const T& k11) {
    double t = (x - x0) / (x1 - x0);
    double u = (y - y0) / (y1 - y0);
	
    return (1 - t) * (1 - u) * k00
         + t * (1 - u) * k10
         + (1 - t) * u * k01
         + t * u * k11;
}


Eigen::VectorXd stringToVectorXd(const std::string& str) {
    // 去掉首尾的中括号
    std::string s = str;
    if (!s.empty() && s.front() == '[') s.erase(0, 1);
    if (!s.empty() && s.back() == ']') s.pop_back();

    std::istringstream iss(s);
    std::vector<double> values;
    double val;

    // 按空格读取double
    while (iss >> val) {
        values.push_back(val);
    }

    // 构造Eigen::VectorXd
    Eigen::VectorXd vec(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        vec(i) = values[i];
    }

    return vec;
}

// Helper function to find indices of elements in a vector
std::vector<int> find_indices(const std::vector<int> &vec, const std::vector<int> &targets)
{
	std::vector<int> indices;
	for (int target : targets)
	{
		auto it = std::find(vec.begin(), vec.end(), target);
		if (it != vec.end())
		{
			indices.push_back(std::distance(vec.begin(), it));
		}
	}
	return indices;
}