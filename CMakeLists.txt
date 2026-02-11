cmake_minimum_required(VERSION 3.17)
project(QuantEnsembleCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(nlohmann_json REQUIRED)
find_package(OnnxRuntime REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

set(SOURCE_FILES
    src/main.cpp
    src/core/MWUEngine.cpp
    src/core/AssetManager.cpp
    src/processors/SignalNormalizer.cpp
    src/processors/GAFGenerator.cpp
    src/ml/OnnxInference.cpp
    src/risk/RiskManager.cpp
    src/net/AsyncServer.cpp
)

add_executable(QuantEnsembleCore ${SOURCE_FILES})

target_link_libraries(QuantEnsembleCore
    PRIVATE
    Eigen3::Eigen
    Boost::system
    nlohmann_json::nlohmann_json
    onnxruntime
)