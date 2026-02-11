#include "../../include/risk/RiskManager.hpp"
#include <cmath>
#include <algorithm>

RiskManager::RiskManager(const std::string& ppo_model_path, double dd_limit, double vol) 
    : max_drawdown_limit(dd_limit), vol_target(vol) {
    ppo_agent = std::make_unique<OnnxWrapper>(ppo_model_path, std::vector<int64_t>{1, 3});
}

RiskManager::RiskDecision RiskManager::ConsultPPO(const MarketSnapshot& market, double mwu_entropy) {
    std::vector<float> state;
    state.push_back(static_cast<float>(market.volatility));
    state.push_back(static_cast<float>(mwu_entropy));
    state.push_back(static_cast<float>(market.prev_real_return));

    auto output = ppo_agent->Predict(state);
    
    
    RiskDecision decision;
    if (output.size() >= 2) {
        decision.new_eta = std::clamp(static_cast<double>(output[0]), 0.01, 0.3);
        decision.position_scale = std::clamp(static_cast<double>(output[1]), 0.0, 1.0);
    } else {
        decision.new_eta = 0.05;
        decision.position_scale = 1.0;
    }
    
    return decision;
}

double RiskManager::ApplyHardRiskRules(double raw_signal, const MarketSnapshot& market) {
    double vol_scale = 1.0;
    if (market.volatility > 0 && vol_target > 0) {
        vol_scale = std::clamp(vol_target / market.volatility, 0.1, 1.0);
    }
    
    if (std::abs(raw_signal) < 0.05) return 0.0;
    
    return raw_signal * vol_scale;
}