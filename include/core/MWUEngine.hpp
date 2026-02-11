#pragma once
#include <map>
#include <string>
#include <vector>
#include "types.hpp"

class MWUEngine {
private:
    std::map<std::string, double> weights;
    double eta;
    bool initialized;

public:
    MWUEngine();
    
    void SetParameters(double learning_rate);
    void UpdateWeights(const std::map<std::string, double>& prev_signals, double real_return);
    double Aggregate(const std::map<std::string, double>& current_signals);
    
    std::map<std::string, double> GetWeights() const;
    bool IsInitialized() const { return initialized; }
};