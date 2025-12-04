import numpy as np
import torch
import os
import utils.surd as surd

if __name__ == '__main__':
    
    MI = np.load("MI_results/mediator_MI_results_target_3.npy", allow_pickle=True).item()
    I_R, I_S, MI_out = surd.surd_global(MI, n_vars=3)
    print(I_S)
    