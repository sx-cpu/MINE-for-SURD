from utils.diagnose.train_mine import train_mine
from utils.diagnose.diagnose_mine import diagnose_mine
from utils.diagnose.ground_truth import generate_gaussian_mi_data
from utils.diagnose.diagnose_mine import plot_mi_curve
from model.MLP import MINE

if __name__ == "__main__":
    X, Y, true_mi = generate_gaussian_mi_data(N=200000, rho=0.8)

    mi_list, loss_list, ma_list, mine = train_mine(
        model_cls=MINE,
        X=X,
        Y=Y,
        batch_size=65536,
        K=5,
        lr=1e-4,
        num_iters=2000
    )

    diagnose_mine(mine, X, Y, batch_size=65536, K=5, true_mi=true_mi)

