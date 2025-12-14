/*MAIN PROGRAM*/
#include <iostream>
#include "wrappedfunc.hpp"
#include <chrono>
#include "omp.h"
#include "unistd.h"

// DFS 搜索一个连通分量
void dfs(int u, const std::vector<std::vector<int>>& graph, std::vector<int>& visited, std::vector<int>& comp) {
    visited[u] = true;
    comp.push_back(u);
    for (int v : graph[u]) {
        if (!visited[v]) {
            dfs(v, graph, visited, comp);
        }
    }
}

int main(int argc, char **argv)
{
	// initialize windfarm only once
	initializeWindFarm();
	// update windfarm and perform optimization
	std::vector<double> new_wind_speed(5, 9.25);
	std::vector<double> new_wind_direction(5, 300.0);
	std::vector<std::vector<double>> init_yaw_angles = generateRandomPT(1, 159, -15, 15); // all set to 0.0 for testing
	std::vector<double> current_yaw_angles_vec = init_yaw_angles[0];
	std::vector<double> new_yaw_angles(159, 0.0);
	double new_power_12 = 0, new_power_3 = 0;
	std::vector<double> new_power_all(159, 0.0);
	// efficiency: 3 serial process, each process has its efficiency value. mul the 3 values to get final eff.
	std::vector<std::vector<double>> mEER_turbines(3, std::vector<double>(159, 1.0));
	std::vector<int> turbine_start(159, 1); // suppose all turbines on
	turbine_start[2] = 0;					// turbine 2 off for testing
	int analytic_grad_flag = 1;				// use analytic gradient
	g_farmopt->calculateWake(); // initial wake calculation
	Eigen::MatrixXi wakemat = g_farmopt->calculate_wake_matrix(); // wakemat test
	// 连通分量测试
	std::vector<std::pair<int,int>> edges;
    // 对于无向图，只需遍历上三角（i < j），避免重复
    for (int i = 0; i < MaxTurbines; ++i) {
        for (int j = i+1; j < MaxTurbines; ++j) {
            if (wakemat(i, j)) {
                edges.emplace_back(i, j);
            }
        }
    }

    // 构建无向图邻接表
    std::vector<std::vector<int>> graph(MaxTurbines);
    for (auto [u, v] : edges) {
        graph[u].push_back(v);
        graph[v].push_back(u); // 无向图双向加边
    }

    std::vector<int> visited(MaxTurbines, false);
    std::vector<std::vector<int>> components;

	// 寻找所有连通分量
    for (int i = 0; i < MaxTurbines; ++i) {
        if (!visited[i]) {
            std::vector<int> comp;
            dfs(i, graph, visited, comp);
            components.push_back(comp);
        }
    }

    // 输出结果
    std::cout << "Number of connected components: " << components.size() << "\n";
    for (int i = 0; i < (int)components.size(); ++i) {
        std::cout << "Component " << i << ": ";
        for (int v : components[i]) {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }



	//update status and efficiency
	auto start_time = std::chrono::high_resolution_clock::now();
	bool optflag = optimizeWindFarm(new_wind_speed,  // 输入：风速
		new_wind_direction, // 输入：方向
		mEER_turbines, // 输入：各风机效能系数值
		current_yaw_angles_vec, // 输入：各风机初始偏航角度
		turbine_start, // 输入：各风机启停状态
		analytic_grad_flag, // 输入：是否使用解析梯度
		new_yaw_angles, // 输出：各风机偏航角度
		new_power_all, // 输出青州12功率
		new_power_12,  // 输出青州3功率
		new_power_3, 0); // 输出全场功率); // 输入：并行计算线程数
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	std::cout << "Optimization Time: " << elapsed.count() << " seconds" << std::endl;
	if (!optflag)
		return 0;
	// save the result here
	// print original yaw angles (first 5)

	// check averge time for all checks
	// std::vector<double> speed_table = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	// std::vector<double> direction_table = {0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0};
	
	// // use random speed and direction table for testing, use integer as baseline
	// std::vector<double> speed_table(10, 0.0);
	// std::vector<double> direction_table(15, 0.0);
	// std::random_device rd;
    // std::mt19937 gen(rd());
    // // 生成 [0, 1) 范围的随机数
    // std::uniform_real_distribution<> dis(0.0, 1.0);
	// for (int i = 0; i < 10; i++){
	// 	double speed_baseline = 3.0 + i*(24.0-3.0)/9.0;
	// 	speed_baseline += 0.1*dis(gen) - 0.05;
	// 	// limit in 3-25
	// 	speed_baseline = std::max(3.0, std::min(25.0, speed_baseline));
	// 	speed_table[i] = speed_baseline;
	// 	//std::cout << "Speed Table " << i << ": " << speed_table[i] << " m/s" << std::endl;
	// }
	
	// for (int i = 0; i < 15; i++){
	// 	double direction_baseline = i*(360.0/15.0);
	// 	direction_baseline += 5*dis(gen) - 2.5;
	// 	// limit in 0-360
	// 	direction_baseline = std::max(0.0, std::min(360.0, direction_baseline));
	// 	direction_table[i] = direction_baseline;
	// 	//std::cout << "Direction Table " << i << ": " << direction_table[i] << "°" << std::endl;
	// }

	// int n_speed = speed_table.size();
	// int n_direction = direction_table.size();
	// for (int i = 0; i < n_direction; i++)
	// {
	// 	for (int j = 0; j < n_speed; j++)
	// 	{	
	// 		std::vector<double> speed_vec(5, speed_table[j]);
	// 		std::vector<double> direction_vec(5, direction_table[i]);
	// 		for (int mode_ = 0; mode_ < 4; mode_++)
	// 		{
	// 			std::cout << "本次执行结果为: 风向 " << direction_table[i] << "°， 风速: " << speed_table[j] << " m/s，";
	// 			//std::cout << "青州12AGC指令: " << qingzhou12_agcval << "W， 青州3AGC指令: " << qingzhou3_agcval << "W" << std::endl;
	// 			double check_power_12 = 0.0, check_power_3 = 0.0;
	// 			auto start_check = std::chrono::high_resolution_clock::now();
	// 			bool check_flag = optimizeWindFarm(
	// 				speed_vec,
	// 				direction_vec,
	// 				mEER_turbines,
	// 				current_yaw_angles_vec,
	// 				turbine_start,
	// 				analytic_grad_flag,
	// 				new_yaw_angles,
	// 				new_power_all,
	// 				check_power_12,
	// 				check_power_3,
	// 				mode_);
	// 			auto end_check = std::chrono::high_resolution_clock::now();
	// 			std::chrono::duration<double> elapsed_check = end_check - start_check;
	// 			if (!check_flag)
	// 			{
	// 				std::cout << "Optimization check failed at speed " << speed_table[i] << " and direction " << direction_table[j] << std::endl;
	// 				continue;
	// 			}
	// 			std::cout << "本次执行周期为: " << elapsed_check.count() << " s" << std::endl;
	// 			std::cout << "*** 本轮优化结束" << std::endl;
	// 		}
	// 	}
	// }

	std::cout << "Current yaw angles (first 5): ";
	for (size_t i = 0; i < 5; ++i) {
		std::cout << current_yaw_angles_vec[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "yaw angles (first 5): ";
	for (size_t i = 0; i < 5; ++i) {
		std::cout << new_yaw_angles[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "Qingzhou 12 Power: " << new_power_12 << " W" << std::endl;
	std::cout << "Qingzhou 3 Power: " << new_power_3 << " W" << std::endl;
	std::cout << "Final Farm Power: " << g_farmopt->getFarmPower() << " W" << std::endl;

	// // test getFarmObj
	// double final_obj = g_farmopt->getFarmObj();
	// std::cout << "Final Farm Objective: " << final_obj << std::endl;
	cleanupWindFarm();
	return 0;
}