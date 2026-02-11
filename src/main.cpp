#include <iostream>
#include <boost/asio.hpp>
#include "net/AsyncServer.cpp"

int main(int argc, char* argv[]) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: QuantEnsembleCore <port> <path_to_gaf_onnx> <path_to_ppo_onnx>\n";
            return 1;
        }

        int port = std::atoi(argv[1]);
        std::string gaf_model_path = argv[2];
        std::string ppo_model_path = argv[3];

        boost::asio::io_context io_context;

        std::cout << "[Config] Port: " << port << std::endl;
        std::cout << "[Config] Loading GAF Model: " << gaf_model_path << std::endl;
        std::cout << "[Config] Loading PPO Model: " << ppo_model_path << std::endl;

        Server server(io_context, port, gaf_model_path, ppo_model_path);

        std::cout << "Waiting for signals..." << std::endl;
        io_context.run();
    }
    catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}