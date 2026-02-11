#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

class OnnxWrapper {
private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<int64_t> input_shape;

public:
    OnnxWrapper(const std::string& model_path, const std::vector<int64_t>& shape);
    
    std::vector<float> Predict(const std::vector<float>& input_data);
};