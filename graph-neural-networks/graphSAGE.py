import numpy as np
import scipy.sparse as sp

class GraphSAGE:
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes=None, use_embedding=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        limit0 = np.sqrt(1 / hidden_dim)
        size0 = (2 * input_dim, hidden_dim)

        limit1 = np.sqrt(1 / output_dim)
        size1 = (2 * hidden_dim, output_dim)

        self.W0 = np.random.uniform(-limit0, limit0, size=size0)
        self.W1 = np.random.uniform(-limit1, limit1, size=size1)

        self.H0 = None
        self.M0 = None
        self.H0_cat = None
        self.pre_H1 = None
        self.H1 = None
        self.M1 = None
        self.H1_cat = None
        self.pre_Z = None
        self.probs = None

    @staticmethod
    def softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def forward(self, X, A_norm):
        if sp.isspmatrix(X):
            X0 = X.toarray()
        else:
            X0 = np.asarray(X)

        self.H0 = X0
        self.M0 = A_norm @ self.H0
        self.H0_cat = np.hstack([self.H0, self.M0])
        self.pre_H1 = self.H0_cat @ self.W0
        self.H1 = np.maximum(0, self.pre_H1)

        self.M1 = A_norm @ self.H1
        self.H1_cat = np.hstack([self.H1, self.M1])
        self.pre_Z = self.H1_cat @ self.W1
        self.probs = self.softmax(self.pre_Z)                  
        return self.probs

    def loss_accuracy_macrof1(self, X, Y_onehot, A_norm, mask):
        probs = self.forward(X, A_norm)
        N = mask.sum()
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        loss = -np.sum(Y_onehot[mask] * log_probs[mask]) / N

        pred = np.argmax(probs[mask], axis=1)
        true = np.argmax(Y_onehot[mask], axis=1)
        num_classes = Y_onehot.shape[1]
        acc = np.mean(pred == true)

        f1_list = []
        for c in range(num_classes):
            tp = np.sum((pred == c) & (true == c))
            fp = np.sum((pred == c) & (true != c))
            fn = np.sum((pred != c) & (true == c))

            denom_p = tp + fp
            precision = tp / denom_p if denom_p > 0 else 0.0

            denom_r = tp + fn
            recall = tp / denom_r if denom_r > 0 else 0.0

            denom_f1 = precision + recall
            f1 = 2 * (precision * recall) / denom_f1 if denom_f1 > 0 else 0.0
            f1_list.append(f1)
            
        macro_f1 = float(np.mean(f1_list))

        return float(loss), float(acc), macro_f1

    def backprop(self, X, Y_onehot, A_norm, mask):
        N_lab = mask.sum()
        dpre_Z = np.zeros_like(self.probs)
        dpre_Z[mask] = (self.probs[mask] - Y_onehot[mask]) / N_lab

        dW1 = self.H1_cat.T @ dpre_Z
        dH1_cat = dpre_Z @ self.W1.T

        d_h = self.hidden_dim
        dH1_direct = dH1_cat[:, :d_h]
        dM1 = dH1_cat[:, d_h:]
        
        dH1_agg = A_norm.T @ dM1
        dH1_total = dH1_direct + dH1_agg
        
        dpre_H1 = dH1_total * (self.pre_H1 > 0).astype(float)
        dW0 = self.H0_cat.T @ dpre_H1

        return dW0, dW1

    def weight_update(self, dW0, dW1, lr):
        self.W0 -= lr * dW0
        self.W1 -= lr * dW1

    def grad_descent_step(self, X, Y_onehot, A_norm, mask, lr):
        _ = self.forward(X, A_norm)
        dW0, dW1 = self.backprop(X, Y_onehot, A_norm, mask)
        self.weight_update(dW0, dW1, lr)
