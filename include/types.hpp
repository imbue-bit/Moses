#pragma once
#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense>

struct MarketSnapshot {
    std::string asset_id;
    double current_price;
    double timestamp;
    double prev_real_return; // 用于 MWU 回测更新 (P_t - P_{t-1})/P_{t-1}
    std::vector<double> ohlc_window; // 用于 GAF 生成
    double volatility; // 用于风控
};

struct ExpertSignal {
    std::string source_name;
    double raw_value;        // 原始值
    double normalized_value; // [-1, 1]
    double weight;           // 当前 MWU 权
};

struct SystemAction {
    std::string asset_id;
    double final_signal;      // [-1, 1] 聚合信号
    double position_sizing;   // [0, 1] 建议仓位 (PPO控制)
    double risk_penalty;      // 风控扣分
    std::map<std::string, double> debug_weights; // 输出当前权重供监控
};