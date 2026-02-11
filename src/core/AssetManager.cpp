#include "../../include/core/MWUEngine.hpp"
#include <map>
#include <mutex>

class AssetManager {
private:
    std::map<std::string, MWUEngine> engines;
    std::map<std::string, std::map<std::string, double>> history_signals;
    std::mutex mtx;

public:
    MWUEngine& GetEngine(const std::string& asset_id) {
        std::lock_guard<std::mutex> lock(mtx);
        return engines[asset_id]; // 如果不存在则自动创建
    }

    void SaveHistory(const std::string& asset_id, const std::map<std::string, double>& signals) {
        std::lock_guard<std::mutex> lock(mtx);
        history_signals[asset_id] = signals;
    }

    std::map<std::string, double> GetHistory(const std::string& asset_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (history_signals.find(asset_id) != history_signals.end()) {
            return history_signals[asset_id];
        }
        return {};
    }
};