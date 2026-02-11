#pragma once
#include <string>

class SignalNormalizer {
public:
    enum class Method { TANH, Z_SCORE, RANK, CLAMP };
    
    static double Normalize(double raw, Method method, double param1 = 0.0, double param2 = 1.0);
};