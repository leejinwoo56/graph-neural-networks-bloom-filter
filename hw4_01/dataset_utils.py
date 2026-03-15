# dataset_utils.py

import os
import numpy as np
import scipy.sparse as sp


def load_PubMed_numpy(data_dir="data/pubmed"):
    """
    Load preprocessed PubMed data (NumPy/SciPy only).

    Returns:
        adj   : (N x N) csr_matrix, row-normalized adjacency (D^{-1}A)
        X     : (N x d_in) csr_matrix, node features
        Y     : (N x C) ndarray, one-hot labels
        y_int : (N,) ndarray, integer labels
        train_mask, val_mask, test_mask : (N,) bool arrays
    """
    adj = sp.load_npz(os.path.join(data_dir, "adj_norm.npz")).tocsr()
    X   = sp.load_npz(os.path.join(data_dir, "features.npz")).tocsr()
    Y   = np.load(os.path.join(data_dir, "labels_onehot.npy"))
    y_int = np.load(os.path.join(data_dir, "labels_int.npy"))

    train_mask = np.load(os.path.join(data_dir, "train_mask.npy")).astype(bool)
    val_mask   = np.load(os.path.join(data_dir, "val_mask.npy")).astype(bool)
    test_mask  = np.load(os.path.join(data_dir, "test_mask.npy")).astype(bool)

    return adj, X, Y, y_int, train_mask, val_mask, test_mask