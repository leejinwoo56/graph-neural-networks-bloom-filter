# graphSAGE.py

import numpy as np
import scipy.sparse as sp
# DO NOT IMPORT ANY OTHER LIBRARIES


class GraphSAGE:
    """
    Two-layer GraphSAGE with mean aggregation and concatenation.

      H0 = X          
      M0 = A_norm @ H0
      H0_cat = [H0 || M0]
      H1 = ReLU(H0_cat @ W0)

      M1 = A_norm @ H1
      H1_cat = [H1 || M1]
      logits = H1_cat @ W1
      probs  = softmax(logits)
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_nodes=None, use_embedding=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        ###################### TODO ####################
        # (i) Weight Initialization

        limit0 = ...
        size0 = ...         # tuple of 2 ints

        limit1 = ...
        size1 = ...         # tuple of 2 ints
        ################################################

        self.W0 = np.random.uniform(
            -limit0, limit0, size=size0
        )

        self.W1 = np.random.uniform(
            -limit1, limit1, size=size1
        )

        # caches for backprop
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
    def softmax(self, logits):
        pass

    @staticmethod
    def softmax(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def forward(self, X, A_norm):
        """
        X      : (N, d_in) csr or dense
        A_norm: (N, N) csr, row-normalized adjacency
        """
        # dense features
        if sp.isspmatrix(X):
            X0 = X.toarray()
        else:
            X0 = np.asarray(X)

        self.H0 = X0

        ###################### TODO ####################
        # (ii) Two-layer GraphSAGE forward pass
        # ----- graphSAGE Layer 0 -----
        self.M0 = ...               # Average neighbor embeddings
        self.H0_cat = ...           # Concat self and neighbor embeddings
        self.pre_H1 = ...           # Apply a linear transformation
        self.H1 = ...               # Nonlinearity

        # ----- graphSAGE Layer 1 -----
        self.M1 = ...               # Average neighbor embeddings
        self.H1_cat = ...           # Concat self and neighbor embeddings
        self.pre_Z = ...            # Apply a linear transformation
        ################################################
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
            ###################### TODO ####################
            # (iv) Evaluation metric
            tp = ...
            fp = ...
            fn = ...

            precision = ...
            recall    = ...
            f1        = ...
            ################################################
            f1_list.append(f1)
            
        macro_f1 = float(np.mean(f1_list))

        return float(loss), float(acc), macro_f1

    def backprop(self, X, Y_onehot, A_norm, mask):
        """
        Returns:
            dW0, dW1, dEmb  (dEmb is None if use_embedding=False)
        """
        N_lab = mask.sum()

        # ----- gradient wrt logits -----
        dpre_Z = np.zeros_like(self.probs)  # (N, C)
        ###################### TODO ####################
        # (iii) Backpropagation and weight update
        # Use cached:
        #   H0, M0, H0_cat, pre_H1, H1, M1, H1_cat, pre_Z, probs
        dpre_Z[mask] = ... # (Hint) Question (a-i)

        # ----- layer 1 -----
        dW1 = ...          # (Hint) Question (a-ii)                         
        dH1_cat = ...      # (Hint) Question (a-iii)                         

        dH1_direct = ...   # (Hint) Question (a-iv)                    
        dM1 = ...          # (Hint) Question (a-iv)

        dH1_agg = ...      # (Hint) Question (a-iv)                     

        dH1_total = ...    # (Hint) Question (a-iv)
        dpre_H1 = ...      # (Hint) Question (a-v)
        
        # ----- layer 0 -----
        dW0 = ...          # (Hint) Question (a-vi)
        ################################################

        return dW0, dW1

    def weight_update(self, dW0, dW1, lr):
        """
        SGD update for W0, W1 and (optionally) emb.
        """
        ###################### TODO ####################
        # (iii) Backpropagation and weight update
        self.W0 = ...
        self.W1 = ...
        ################################################

        return None

    def grad_descent_step(self, X, Y_onehot, A_norm, mask, lr):
        """
        One full-graph gradient descent step.
        """
        _ = self.forward(X, A_norm)
        dW0, dW1 = self.backprop(X, Y_onehot, A_norm, mask)
        self.weight_update(dW0, dW1, lr)