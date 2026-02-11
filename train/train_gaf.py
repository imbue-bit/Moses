import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.onnx

DATA_PATH = 'data/training_data.csv'
WINDOW_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = 'models/gaf_cnn.onnx'

def generate_gaf_image(timeseries):    
    min_val = np.min(timeseries)
    max_val = np.max(timeseries)
    diff = max_val - min_val
    if diff < 1e-6: diff = 1.0
    x_scaled = ((timeseries - min_val) / diff) * 2.0 - 1.0
    x_scaled = np.clip(x_scaled, -1.0, 1.0) 
    phi = np.arccos(x_scaled)
    cos_phi = x_scaled
    sin_phi = np.sqrt(1.0 - x_scaled**2)
    G = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
    return G

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, csv_file, window_size):
        self.data = pd.read_csv(csv_file)
        self.prices = self.data['close'].values
        self.targets = self.data['target_signal'].values
        self.window_size = window_size

    def __len__(self):
        return len(self.prices) - self.window_size

    def __getitem__(self, idx):
        window_data = self.prices[idx : idx + self.window_size]
        target = self.targets[idx + self.window_size - 1]
        
        gaf_img = generate_gaf_image(window_data)
        
        gaf_tensor = torch.tensor(gaf_img, dtype=torch.float32).unsqueeze(0)
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        
        return gaf_tensor, target_tensor

class GAF_CNN(nn.Module):
    def __init__(self):
        super(GAF_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

def train():
    if not os.path.exists('models'): os.makedirs('models')
    
    dataset = FinancialTimeSeriesDataset(DATA_PATH, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = GAF_CNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Start Training with {len(dataset)} samples...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.6f}")

    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 1, WINDOW_SIZE, WINDOW_SIZE)
    torch.onnx.export(
        model, 
        dummy_input, 
        MODEL_SAVE_PATH,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()