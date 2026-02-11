#include "../../include/processors/GAFGenerator.hpp"
#include <cmath>
#include <algorithm>

GAFGenerator::GAFGenerator(const std::string& model_path, int size) : window_size(size) {
    cnn_model = std::make_unique<OnnxWrapper>(model_path, std::vector<int64_t>{1, 1, size, size});
}

std::vector<float> GAFGenerator::Transform(const std::vector<double>& time_series) {
    if (time_series.empty()) return {};
    
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(time_series.data(), time_series.size());
    double min_val = x.minCoeff();
    double max_val = x.maxCoeff();
    
    // 避免除以零
    double range = (max_val - min_val) < 1e-6 ? 1.0 : (max_val - min_val);
    
    Eigen::VectorXd x_scaled = ((x.array() - min_val) / range) * 2.0 - 1.0;
    
    // 限制在 [-1, 1] 避免 NaN
    x_scaled = x_scaled.cwiseMax(-1.0).cwiseMin(1.0); 
    Eigen::VectorXd phi = x_scaled.array().acos();
    
    // 利用三角恒等式: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    // 这里 cos(phi) 就是 x_scaled
    // sin(phi) = sqrt(1 - x_scaled^2)
    Eigen::VectorXd cos_phi = x_scaled;
    Eigen::VectorXd sin_phi = (1.0 - x_scaled.array().square()).sqrt();
    
    // Matrix G = cos_phi * cos_phi^T - sin_phi * sin_phi^T
    Eigen::MatrixXd G = (cos_phi * cos_phi.transpose()) - (sin_phi * sin_phi.transpose());
    
    std::vector<float> result;
    result.reserve(window_size * window_size);
    
    for(int i=0; i<window_size; ++i) {
        for(int j=0; j<window_size; ++j) {
            result.push_back(static_cast<float>(G(i, j)));
        }
    }
    return result;
}

double GAFGenerator::GetSignal(const std::vector<double>& time_series) {
    if (time_series.size() < static_cast<size_t>(window_size)) return 0.0;
    std::vector<double> window(time_series.end() - window_size, time_series.end());
    auto gaf_image = Transform(window);
    auto output = cnn_model->Predict(gaf_image);
    if (output.size() >= 3) {
        return (double)(output[2] - output[0]);
    }
    return 0.0;
}