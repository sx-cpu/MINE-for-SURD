import torch
import numpy as np

def generate_gaussian_mi_data(N=200000, rho=0.8, dim=1, device="cuda"):
    """
    生成具有真实可计算互信息的高斯数据。
    I = -0.5 * log(1 - rho^2) * dim
    """
    X = torch.randn(N, dim, device=device)
    eps = torch.randn(N, dim, device=device)
    Y = rho * X + torch.sqrt(torch.tensor(1 - rho*rho, device=device)) * eps

    true_mi = -0.5 * dim * np.log(1 - rho*rho)
    return X, Y, true_mi
