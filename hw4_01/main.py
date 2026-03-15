# main.py

import numpy as np
from dataset_utils import load_PubMed_numpy
from graphSAGE import GraphSAGE
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use_embeddings', action='store_true',
                    help='Use node embeddings')
args = parser.parse_args()

np.random.seed(2025)

def main():
    adj, X, Y_onehot, y_int, train_mask, val_mask, test_mask = load_PubMed_numpy()

    N, d_in = X.shape
    C = Y_onehot.shape[1]

    hidden_dim = 128
    lr = 5e-3
    num_epochs = 100

    # base version (no embeddings)
    model = GraphSAGE(input_dim=d_in, hidden_dim=hidden_dim, output_dim=C)

    for epoch in range(1, num_epochs + 1):
        model.grad_descent_step(X, Y_onehot, adj, train_mask, lr)
        train_loss, train_acc, train_macro_f1 = model.loss_accuracy_macrof1(X, Y_onehot, adj, train_mask)
        val_loss, val_acc, val_macro_f1 = model.loss_accuracy_macrof1(X, Y_onehot, adj, val_mask)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, train_macro_F1 = {train_macro_f1:.4f} | "
                f"val_loss = {val_loss:.4f}, val_acc = {val_acc:.4f}, val_macro_f1 = {val_macro_f1:.4f}"
            )

    test_loss, test_acc, test_macro_f1 = model.loss_accuracy_macrof1(X, Y_onehot, adj, test_mask)
    print(f"Test: loss = {test_loss:.4f}, acc = {test_acc:.4f}, macro_F1 = {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()