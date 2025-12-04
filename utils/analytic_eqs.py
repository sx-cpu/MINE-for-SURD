import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ===============================
# 1. Mediator generator
# ===============================
def mediator(Nt):
    q1, q2, q3 = np.zeros(Nt), np.zeros(Nt), np.zeros(Nt)
    W1, W2, W3 = np.random.normal(0, 1, Nt), np.random.normal(0, 1, Nt), np.random.normal(0, 1, Nt)
    for n in range(Nt-1):
        q1[n+1] = np.sin(q2[n]) + 0.001*W1[n]
        q2[n+1] = np.cos(q3[n]) + 0.01*W2[n]
        q3[n+1] = 0.5*q3[n] + 0.1*W3[n]
    return q1, q2, q3

