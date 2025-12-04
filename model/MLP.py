import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MineNet(nn.Module):
    def __init__(self, dim_x, dim_y, hidden=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )
    def forward(self, x, y):
        xy = torch.cat([x,y], dim=1)
        return self.model(xy)

class MINE:
    def __init__(self, dim_x, dim_y, lr=1e-5, device='cuda'):
        self.net = MineNet(dim_x, dim_y).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.device = device

        # EMA for stability
        self.ma_et = None
        self.ma_rate = 0.01

    @staticmethod
    def shuffle(y):
        idx = torch.randperm(len(y))
        return y[idx]

    def train(self, dataloader, epochs=20):
        self.net.train()
        for epoch in range(epochs):
            running_mi = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.opt.zero_grad()

                # joint
                t = self.net(batch_x, batch_y).mean()

                # marginal
                y_marg = self.shuffle(batch_y)
                et = torch.exp(self.net(batch_x, y_marg)).mean()

                # EMA smoothing
                if self.ma_et is None:
                    self.ma_et = et.detach()
                else:
                    self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * et.detach()

                loss = -(t - torch.log(self.ma_et + 1e-8))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)  
                self.opt.step()

                running_mi += (t - torch.log(self.ma_et)).item()
            print(f"Epoch {epoch+1}/{epochs}, avg MI={running_mi/len(dataloader):.6f}")

    def estimate(self, X, Y, batch_size=65536):
        self.net.eval()
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mi_vals = []
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                t = self.net(batch_x, batch_y).mean()
                et = torch.exp(self.net(batch_x, self.shuffle(batch_y))).mean()
                mi_vals.append((t - torch.log(et)).item())
        return np.mean(mi_vals)