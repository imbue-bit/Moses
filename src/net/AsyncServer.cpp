#include <boost/asio.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <utility>

#include "../../include/core/MWUEngine.hpp"
#include "../../include/core/AssetManager.cpp"
#include "../../include/processors/GAFGenerator.hpp"
#include "../../include/processors/SignalNormalizer.hpp"
#include "../../include/risk/RiskManager.hpp"

using boost::asio::ip::tcp;
using json = nlohmann::json;

class Session : public std::enable_shared_from_this<Session> {
public:
    Session(tcp::socket socket, AssetManager& asset_mgr, GAFGenerator& gaf, RiskManager& risk)
        : socket_(std::move(socket)), asset_mgr_(asset_mgr), gaf_gen_(gaf), risk_mgr_(risk) {}

    void Start() {
        DoRead();
    }

private:
    void DoRead() {
        auto self(shared_from_this());
        socket_.async_read_some(boost::asio::buffer(data_, max_length),
            [this, self](boost::system::error_code ec, std::size_t length) {
                if (!ec) {
                    std::string request_str(data_, length);
                    std::string response = ProcessRequest(request_str);
                    DoWrite(response);
                }
            });
    }

    void DoWrite(std::string response) {
        auto self(shared_from_this());
        boost::asio::async_write(socket_, boost::asio::buffer(response),
            [this, self](boost::system::error_code ec, std::size_t /*length*/) {
                if (!ec) {
                    DoRead(); 
                }
            });
    }

    std::string ProcessRequest(const std::string& req_str) {
        try {
            json req = json::parse(req_str);
            json resp;

            std::string asset = req["asset_id"];
            double prev_ret = req["prev_return"];
            auto input_signals = req["signals"];
            std::vector<double> prices = req["prices"].get<std::vector<double>>();
            double volatility = req["volatility"];

            auto& mwu = asset_mgr_.GetEngine(asset);
            auto history = asset_mgr_.GetHistory(asset);

            // 如果有历史记录，进行权重更新
            if (!history.empty()) {
                mwu.UpdateWeights(history, prev_ret);
            }

            std::map<std::string, double> current_normalized_signals;
            
            for (auto& [key, val] : input_signals.items()) {
                current_normalized_signals[key] = SignalNormalizer::Normalize(val, SignalNormalizer::Method::TANH);
            }

            try {
                double gaf_sig = gaf_gen_.GetSignal(prices);
                current_normalized_signals["INTERNAL_GAF"] = gaf_sig;
            } catch (...) {
                current_normalized_signals["INTERNAL_GAF"] = 0.0;
            }

            MarketSnapshot market{asset, prices.back(), 0.0, prev_ret, prices, volatility};
            
            // 计算权重熵作为系统混乱度指标
            double entropy = 0.0;
            auto weights = mwu.GetWeights();
            for(auto [k, w] : weights) if(w > 0) entropy -= w * std::log(w);

            auto ppo_decision = risk_mgr_.ConsultPPO(market, entropy);
            
            // 应用 PPO 调整学习率
            mwu.SetParameters(ppo_decision.new_eta);

            double agg_signal = mwu.Aggregate(current_normalized_signals);

            // 先硬风控，再叠加 PPO 的仓位建议
            double risk_adj_signal = risk_mgr_.ApplyHardRiskRules(agg_signal, market);
            double final_signal = risk_adj_signal * ppo_decision.position_scale;

            // 保存历史供下一帧训练
            asset_mgr_.SaveHistory(asset, current_normalized_signals);

            resp["asset"] = asset;
            resp["final_signal"] = final_signal;
            resp["weights"] = weights;
            resp["meta_info"] = {
                {"ppo_eta", ppo_decision.new_eta},
                {"ppo_scale", ppo_decision.position_scale}
            };
            
            return resp.dump();

        } catch (const std::exception& e) {
            json err;
            err["error"] = e.what();
            return err.dump();
        }
    }

    tcp::socket socket_;
    AssetManager& asset_mgr_;
    GAFGenerator& gaf_gen_;
    RiskManager& risk_mgr_;
    enum { max_length = 10240 }; // 10KB Buffer
    char data_[max_length];
};

class Server {
public:
    Server(boost::asio::io_context& io_context, short port, 
           const std::string& gaf_model, const std::string& ppo_model)
        : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)),
          gaf_gen_(gaf_model, 64), // GAF 图片大小 64x64
          risk_mgr_(ppo_model, 0.20, 0.02) // MaxDD 20%, TargetVol 2%
    {
        DoAccept();
    }

private:
    void DoAccept() {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket) {
                if (!ec) {
                    std::make_shared<Session>(std::move(socket), asset_mgr_, gaf_gen_, risk_mgr_)->Start();
                }
                DoAccept();
            });
    }

    tcp::acceptor acceptor_;
    AssetManager asset_mgr_;
    GAFGenerator gaf_gen_;
    RiskManager risk_mgr_;
};