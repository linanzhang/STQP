""" v1
Sparsity-Promoting Dynamic Mode Decomposition 
via Sequentially Thresholded Quadratic Programming 
"""

import numpy as np
from numpy.linalg import solve
from scipy.sparse import csc_matrix, eye, hstack, vstack
from scipy.sparse.linalg import spsolve

from pydmd import DMD


def compute_P_inv_sqrt(P, epsilon=1e-10):
    P = (P + P.T) / 2 # Ensure P is symmetric (due to floating point errors)
    eig_vals, eig_vecs = np.linalg.eigh(P) # Eigenvalue decomposition
    eig_vals = np.maximum(eig_vals, epsilon) # Regularize eigenvalues to avoid division by zero
    inv_sqrt_eig_vals = 1.0 / np.sqrt(eig_vals) # Compute Λ^{-1/2}
    Lambda_inv_sqrt = np.diag(inv_sqrt_eig_vals)
    P_inv_sqrt = eig_vecs @ Lambda_inv_sqrt @ eig_vecs.T # Reconstruct P^{-1/2}
    return P_inv_sqrt


class stqpDMD(DMD):
    def __init__(
        self,
        svd_rank=0,
        tlsq_rank=0,
        exact=True,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        mu=10,
        verbose=True,
        use_csc=False,
        release_memory=True,
        zero_absolute_tolerance=1.0e-12,
        tikhonov_regularization=None
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
            tikhonov_regularization=tikhonov_regularization
        )
        self.mu = mu
        self._verbose = verbose
        self._release_memory = release_memory
        self._zero_absolute_tolerance = zero_absolute_tolerance
        self._use_csc = use_csc
        self._P = None
        self._q = None
        self._modes_activation_bitmask_proxy = None

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition of the input data.
        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        super().fit(X)
        original_eigs = self.eigs.copy()
        P, q = self._optimal_dmd_matrices()

        # # Normalization
        # S = compute_P_inv_sqrt(P)
        # P = S.T @ P @ S
        # q = S.T @ q
        # self._S = S

        self._P = csc_matrix(P) if self._use_csc else P
        self._q = q
        self._max_iterations = P.shape[0]
        self._b = self._stqp_update_csc() if self._use_csc else self._stqp_update()

        # re-allocate the Proxy to avoid problems due to the fact that we re-computed the amplitudes
        self._allocate_modes_bitmask_proxy()

        # release memory
        if self._release_memory:
            self._P = None
            self._q = None
            self._Plow = None

        return self

    # def compute_objective(self, b):
    #     return ( b.conj().T @ self._P @ b - b.conj().T @ self._q - self._q.conj().T @ b ) / np.linalg.norm(self._P, 2) + self.mu**2 * np.count_nonzero(b)

    def compute_objective(self, b):
        P_norm = np.linalg.norm(self._P.toarray(), 2) if self._use_csc else np.linalg.norm(self._P, 2)
        quadratic = (b.conj().T @ self._P @ b).item()
        linear = (b.conj().T @ self._q + self._q.conj().T @ b).item()
        return (quadratic - linear) / P_norm + self.mu**2 * np.count_nonzero(b)

    def _stqp_update(self):
        # the initial step
        b = np.linalg.solve(self._P, self._q)
        S = np.where(np.abs(b) < self.mu)[0] # set of small indices
        n = self._max_iterations
        I = np.eye(n)

        err = True
        k = 1
        # the iterative step
        while k < self._max_iterations and err:
            m = len(S)
            E = I[:, S]*(np.abs(self._P).max())
            O = np.zeros((m, m))
            
            # Construct the augmented system
            P_augmented = np.vstack((np.hstack((self._P, E)), np.hstack((E.T, O))))
            q_augmented = np.concatenate((self._q, np.zeros(m)))
            y = np.linalg.solve(P_augmented, q_augmented)
            b_new = y[:n]
            S_new = np.where(np.abs(b_new) < self.mu)[0]
        
            # check whether S^{k} = S^{k-1}
            err = set(S) != set(S_new)
            
            # update
            b = b_new
            S = S_new
            k += 1

        if self._verbose:
            print("STQP: {} iterations".format(k))
        b[np.abs(b) < self._zero_absolute_tolerance] = 0

        # # Normalization
        # b = self._S@b
        return b


    def _stqp_update_csc(self):
        # Initial step using sparse solver
        b = spsolve(self._P, self._q)
        S = np.where(np.abs(b) < self.mu)[0]
        n = self._max_iterations
        I = eye(n, format='csc')

        err = True
        k = 1
        
        while k < self._max_iterations and err:
            m = len(S)
            E = I[:, S] * (abs(self._P).max())
            O = csc_matrix((m, m))
            
            # Construct augmented system using sparse matrices
            P_augmented = vstack([
                hstack([self._P, E]),
                hstack([E.T, O])
            ], format='csc')
            
            q_augmented = np.concatenate((self._q, np.zeros(m)))
            y = spsolve(P_augmented, q_augmented)
            b_new = y[:n]
            
            S_new = np.where(np.abs(b_new) < self.mu)[0]
            err = set(S) != set(S_new)
            
            b = b_new
            S = S_new
            k += 1

        if self._verbose:
            print("STQP: {} iterations".format(k))
        b[np.abs(b) < self._zero_absolute_tolerance] = 0
        return b
