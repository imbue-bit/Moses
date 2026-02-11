#pragma once
#include <vector>
#include <Eigen/Dense>
#include "ml/OnnxInference.hpp"

class GAFGenerator {
private:
    std::unique_ptr<OnnxWrapper> cnn_model;
    int window_size;

public:
    GAFGenerator(const std::string& model_path, int size);
    
    std::vector<float> Transform(const std::vector<double>& time_series);
    double GetSignal(const std::vector<double>& time_series);
};