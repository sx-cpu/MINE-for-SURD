import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from model.MLP import MINE
import utils.analytic_eqs as cases  
import utils.datasets as proc
import datetime


# ----------------------------------------
# main 
# ----------------------------------------
if __name__ == "__main__":
    Nt = 2 * 10**5
    lag = 1
    transient = 10000
    samples = Nt - transient

    target_var = 3                 
    input_vars = [1, 2, 3]            

    os.makedirs('./data', exist_ok=True)
    formatted_Nt = "{:.0e}".format(Nt).replace("+0","").replace("+","")
    filepath = os.path.join('./data', f"mediator_Nt_{formatted_Nt}.npy")

    # -----------------------------
    # 1. datasets
    # -----------------------------
    if os.path.isfile(filepath):
        print("Loading saved mediator data ...")
        q1, q2, q3 = np.load(filepath, allow_pickle=True)
    else:
        print("Generating mediator data ...")
        qs = cases.mediator(Nt)
        q1, q2, q3 = [q[transient:] for q in qs]
        np.save(filepath, [q1, q2, q3])
        print("Saved mediator data to", filepath)

    
    data_map = {1: q1, 2: q2, 3: q3}

    # -----------------------------
    # 2. comb
    # -----------------------------
    subsets = proc.all_subsets(input_vars)   # 例如 [(1,), (2,), (1,2)]
    print("Variable sets to evaluate:", subsets)

    # -----------------------------
    # 3. all results 
    # -----------------------------
    MI_results = {}

    batch_size = 65536
    epochs = 100
    lr = 1e-5

    for subset in subsets:
        print(f"\n=== Training MINE for inputs {subset} → target {target_var}[+{lag}] ===")

        # build full X,Y tensors first (you already do this)
        X_list = [data_map[v][:-lag] for v in subset]
        X = np.vstack(X_list).T
        X = torch.tensor(X, dtype=torch.float32)

        Y = data_map[target_var][lag:]
        Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

        # 1) fit normalization on full training data BEFORE creating dataloader (or do it now)
        mine = MINE(dim_x=X.shape[1], dim_y=1, device='cuda')
        mine.fit_normalization(X, Y)   # <-- IMPORTANT: full-data fit

        # 2) now create dataset/loader and train
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mine.train(loader, epochs=epochs)


        # estimate MI
        MI_value = mine.estimate(X, Y, batch_size=65536)
        MI_results[subset] = float(MI_value)

        print(f"MI{subset} = {MI_value:.6f}")

    # -----------------------------
    # 4. save results
    # -----------------------------
        
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    savepath = f"./MI_results/mediator_MI_results_target_{target_var}_{ts}.npy"
    np.save(savepath, MI_results)
    print(f"\nSaved MI results to {savepath}")

    print("\nFinal MI results:")
    for k, v in MI_results.items():
        print(k, ":", v)

    

    
