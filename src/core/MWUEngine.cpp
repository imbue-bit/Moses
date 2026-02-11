#include "../../include/core/MWUEngine.hpp"
#include <cmath>
#include <numeric>

MWUEngine::MWUEngine() : eta(0.05), initialized(false) {}

void MWUEngine::SetParameters(double learning_rate) {
    // 限制学习率范围，防止 PPO 给出极端值导致系统震荡
    this->eta = std::clamp(learning_rate, 0.001, 0.5);
}

void MWUEngine::UpdateWeights(const std::map<std::string, double>& prev_signals, double real_return) {
    if (weights.empty() || prev_signals.empty()) return;

    // 确定真实方向 [-1, 1] (binary for simplicity in loss calculation)
    // 也可以使用连续值，这里使用 Sign 以获得更清晰的后悔界
    double reality = (real_return > 0) ? 1.0 : (real_return < 0 ? -1.0 : 0.0);

    double total_weight = 0.0;

    for (auto& [name, w] : weights) {
        if (prev_signals.find(name) == prev_signals.end()) {
            total_weight += w;
            continue;
        }

        double prediction = prev_signals.at(name); // [-1, 1]
        
        // 0 = Perfect prediction, 1 = Complete opposite
        double loss = std::abs(prediction - reality) / 2.0;
        
        // Multiplicative Update
        w *= (1.0 - eta * loss);
        
        // 设定最小权重阈值，防止专家永久死亡
        if (w < 1e-4) w = 1e-4;
        
        total_weight += w;
    }

    // 归一化权重
    if (total_weight > 0) {
        for (auto& [name, w] : weights) {
            w /= total_weight;
        }
    }
}

double MWUEngine::Aggregate(const std::map<std::string, double>& current_signals) {
    // 初始化新出现的专家
    for (const auto& [name, val] : current_signals) {
        if (weights.find(name) == weights.end()) {
            // 给新专家平均权重或较低的初始权重
            weights[name] = weights.empty() ? 1.0 : (1.0 / weights.size());
        }
    }
    
    // 重新归一化以防新加入导致的Sum!=1
    double total_w_check = 0.0;
    for(const auto& pair : weights) total_w_check += pair.second;
    if(total_w_check > 0) for(auto& pair : weights) weights[pair.first] /= total_w_check;

    // 加权聚合
    double final_sig = 0.0;
    for (const auto& [name, signal] : current_signals) {
        final_sig += signal * weights[name];
    }
    
    return final_sig;
}

std::map<std::string, double> MWUEngine::GetWeights() const {
    return weights;
}