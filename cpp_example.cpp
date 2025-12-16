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

	// test connected components
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

    // // 输出结果
    // std::cout << "Number of connected components: " << components.size() << "\n";
    // for (int i = 0; i < (int)components.size(); ++i) {
    //     std::cout << "Component " << i << ": ";
    //     for (int v : components[i]) {
    //         std::cout << v << " ";
    //     }
    //     std::cout << "\n";
    // }

	//update status and efficiency
	auto start_time = std::chrono::high_resolution_clock::now();
	int mode = 0;
	bool optflag = optimizeWindFarm(new_wind_speed,  // 输入：风速
		new_wind_direction, // 输入：方向
		mEER_turbines, // 输入：各风机效能系数值
		current_yaw_angles_vec, // 输入：各风机初始偏航角度
		turbine_start, // 输入：各风机启停状态
		analytic_grad_flag, // 输入：是否使用解析梯度
		new_yaw_angles, // 输出：各风机偏航角度
		new_power_all, // 输出青州12功率
		new_power_12,  // 输出青州3功率
		new_power_3, mode); // 输出全场功率); // 输入：并行计算线程数
	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end_time - start_time;
	std::cout << "Optimization Time: " << elapsed.count() << " seconds" << std::endl;
	if (!optflag)
		return 0;
	// save the result here

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