#include "../../include/ml/OnnxInference.hpp"

#include <stdexcept>
#include <numeric>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN32
static std::wstring StringToWString(const std::string& str) {
    if (str.empty()) return L"";

    int size_needed = MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        static_cast<int>(str.size()),
        NULL,
        0);

    std::wstring wstr(size_needed, 0);

    MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        static_cast<int>(str.size()),
        &wstr[0],
        size_needed);

    return wstr;
}
#endif

OnnxWrapper::OnnxWrapper(const std::string& model_path,
                         const std::vector<int64_t>& shape)
    : env(ORT_LOGGING_LEVEL_WARNING, "QuantEnsemble"),
      input_shape(shape)
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
    std::wstring w_model_path = StringToWString(model_path);
    session = std::make_unique<Ort::Session>(
        env,
        w_model_path.c_str(),
        session_options);
#else
    session = std::make_unique<Ort::Session>(
        env,
        model_path.c_str(),
        session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto name = session->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(strdup(name.get()));
    }

    size_t num_output_nodes = session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(strdup(name.get()));
    }
}

OnnxWrapper::~OnnxWrapper() {
    for (auto p : input_node_names) {
        free(p);
    }
    for (auto p : output_node_names) {
        free(p);
    }
}

std::vector<float> OnnxWrapper::Predict(
    const std::vector<float>& input_data)
{
    size_t input_tensor_size = 1;
    for (auto dim : input_shape) {
        if (dim > 0)
            input_tensor_size *= dim;
    }

    if (input_data.size() != input_tensor_size) {
        throw std::runtime_error(
            "Input data size mismatch with model input shape.");
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                   OrtMemTypeDefault);

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size());

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_node_names.data(),
        &input_tensor,
        1,
        output_node_names.data(),
        1);

    float* floatarr =
        output_tensors.front().GetTensorMutableData<float>();

    size_t output_size =
        output_tensors.front()
            .GetTensorTypeAndShapeInfo()
            .GetElementCount();

    return std::vector<float>(
        floatarr,
        floatarr + output_size);
}
