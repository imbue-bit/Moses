import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import os

DATA_PATH = 'data/training_data.csv'
MODEL_SAVE_PATH = 'models/ppo_controller.onnx'

class MWUMetaEnv(gym.Env):
    """
    State: [Market_Volatility, MWU_Entropy_Proxy, Recent_Returns]
    Action: [Learning_Rate (eta), Position_Scale]
    Reward: Sharpe Ratio based Reward
    """
    def __init__(self, df):
        super(MWUMetaEnv, self).__init__()
        self.df = df
        self.current_step = 0
        
        # Action[0]: eta (学习率), 映射到 [0.01, 0.5]
        # Action[1]: scale (仓位), 映射到 [0.0, 1.0]
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        # 观察空间：[volatility, trend_strength, recent_return]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.simulated_equity = 1.0
        self.equity_curve = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 20 # Skip initial rolling window
        self.simulated_equity = 1.0
        self.equity_curve = [1.0]
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
        # 用波动率作为代理
        vol = row['volatility']
        ret = row['log_ret']
        
        obs = np.array([vol, vol * 10, ret], dtype=np.float32)
        return obs

    def step(self, action):
        eta_action = (action[0] + 1) / 2 * 0.49 + 0.01 # -> 0.01 ~ 0.5
        pos_scale_action = (action[1] + 1) / 2         # -> 0.0 ~ 1.0
        
        row = self.df.iloc[self.current_step]
        actual_ret = row['log_ret']
        vol = row['volatility']
        
        # 我们假设 MWU 产生的信号是有一定 alpha 的
        mwu_signal_proxy = np.sign(actual_ret) * 0.1 + np.random.normal(0, 0.5) 
        strategy_ret = mwu_signal_proxy * pos_scale_action * actual_ret
        
        self.simulated_equity *= (1 + strategy_ret)
        self.equity_curve.append(self.simulated_equity)
        
        # Reward = Returns - Lambda * Risk
        # 鼓励在高波动时降低仓位，低波动时提高收益
        reward = strategy_ret * 100 - (vol * pos_scale_action * 10) 
        
        # 步进
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

def train_ppo():
    if not os.path.exists('models'): os.makedirs('models')
    
    if not os.path.exists(DATA_PATH):
        print("Data not found. Run fetch_data.py first.")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    env = MWUMetaEnv(df)
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Start Training...")
    model.learn(total_timesteps=20000)
    
    print("Exporting to ONNX...")
    
    class OnnxablePolicy(nn.Module):
        def __init__(self, extractor, action_net, value_net):
            super().__init__()
            self.extractor = extractor
            self.action_net = action_net
            self.value_net = value_net
            
        def forward(self, observation):
            # SB3 PPO forward pass logic
            features = self.extractor(observation)
            action_logits = self.action_net(features)
            # 输出确定的动作 (Deterministic for inference)
            return action_logits # Tanh output directly from actor

    # 提取网络组件
    policy = model.policy
    onnx_policy = OnnxablePolicy(
        policy.mlp_extractor, 
        policy.action_net, 
        policy.value_net
    )
    
    dummy_input = torch.randn(1, 3) # Observation space size is 3
    
    torch.onnx.export(
        onnx_policy,
        dummy_input,
        MODEL_SAVE_PATH,
        opset_version=11,
        input_names=["input"],
        output_names=["action"]
    )
    
    print(f"PPO Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_ppo()