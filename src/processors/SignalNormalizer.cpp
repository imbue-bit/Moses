#include "../../include/processors/SignalNormalizer.hpp"
#include <cmath>
#include <algorithm>

double SignalNormalizer::Normalize(double raw, Method method, double param1, double param2) {
    switch (method) {
        case Method::TANH:
            // Tanh 压缩到 [-1, 1]
            return std::tanh(raw);
            
        case Method::Z_SCORE:
            // (x - mean) / std_dev
            if (std::abs(param2) < 1e-6) return 0.0;
            return std::clamp((raw - param1) / param2, -1.0, 1.0);
            
        case Method::RANK:
            // 百分位 [0, 1] -> [-1, 1]
            return (std::clamp(raw, 0.0, 1.0) - 0.5) * 2.0;
            
        case Method::CLAMP:
        default:
            return std::clamp(raw, -1.0, 1.0);
    }
}