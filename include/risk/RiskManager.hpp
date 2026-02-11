#pragma once
#include "types.hpp"
#include "ml/OnnxInference.hpp"

class RiskManager {
private:
    std::unique_ptr<OnnxWrapper> ppo_agent;
    double max_drawdown_limit;
    double vol_target;

public:
    RiskManager(const std::string& ppo_model_path, double dd_limit, double vol_target);

    struct RiskDecision {
        double new_eta;
        double position_scale;
    };

    RiskDecision ConsultPPO(const MarketSnapshot& market, double mwu_entropy);
    
    double ApplyHardRiskRules(double raw_signal, const MarketSnapshot& market);
};