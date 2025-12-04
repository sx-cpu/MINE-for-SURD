import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from model.MLP import MINE
import utils.analytic_eqs as cases  
import utils.datasets as proc


# ----------------------------------------
# main 
# ----------------------------------------
if __name__ == "__main__":
    Nt = 5 * 10**6
    lag = 1
    transient = 10000
    samples = Nt - transient

    target_var = 1                 
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

    batch_size = 4096

    for subset in subsets:
        print(f"\n=== Training MINE for inputs {subset} → target {target_var}[+{lag}] ===")

        # input: X
        X_list = [data_map[v][:-lag] for v in subset]
        X = np.vstack(X_list).T
        X = torch.tensor(X, dtype=torch.float32)
        print(f"X.shape:{X.shape}")

        # target: Y
        Y = data_map[target_var][lag:]
        Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)
        print(f"Y.shape:{Y.shape}")
        # dataloader
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # init MINE
        mine = MINE(dim_x=X.shape[1], dim_y=1, device='cuda')
        mine.train(loader, epochs=10)

        # estimate MI
        MI_value = mine.estimate(X, Y, batch_size=batch_size)
        MI_results[subset] = float(MI_value)

        print(f"MI{subset} = {MI_value:.6f}")

    # -----------------------------
    # 4. save results
    # -----------------------------
    
    np.save(f"./MI_results/mediator_MI_results_target_{target_var}.npy", MI_results)
    print(f"\nSaved MI results to ./data/mediator_MI_results_target_{target_var}.npy")

    print("\nFinal MI results:")
    for k, v in MI_results.items():
        print(k, ":", v)

    

    
