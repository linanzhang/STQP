import numpy as np

def STQP_Separation(model, m, H, W, U_full):
    Phi = model.modes
    eigs = model.eigs
    b = model._b
    
    b_abs = np.abs(b)
    idx = np.argsort(b_abs)[::-1]
    cum_b = np.cumsum(b_abs[idx]) / np.sum(b_abs)
    b_threshold = 0.99
    k = np.searchsorted(cum_b, b_threshold) + 1
    
    bg_modes = idx[:k]
    fg_modes = idx[k:]
    
    time_dynamics_bg = np.zeros((len(bg_modes), m), dtype=complex)
    for i, mode_idx in enumerate(bg_modes):
        omega = np.log(eigs[mode_idx])
        time_dynamics_bg[i, :] = b[mode_idx] * np.exp(omega * np.arange(m))
    
    X_bg = (Phi[:, bg_modes] @ time_dynamics_bg).real
    X_fg = U_full[:, :m] - X_bg
    
    fg_video = X_fg.T.reshape(m, H, W)
    fg_video = np.clip(np.abs(fg_video), 0, 1)
    
    bg_video = X_bg.T.reshape(m, H, W)
    bg_video = np.clip(np.abs(bg_video), 0, 1)
    
    return fg_video, bg_video, len(bg_modes), len(fg_modes)

def Frequency_Based_Separation(model, m, H, W, U_full, dt, omega_threshold=0.01):
    Phi = model.modes
    eigs = model.eigs
    b = model.amplitudes
    
    omega = np.log(eigs) / dt
    
    bg_modes = np.where(np.abs(omega) < omega_threshold)[0]
    if len(bg_modes) == 0:
        idx_small = np.argsort(np.abs(omega))
        bg_modes = idx_small[:3]
    
    time_vec = np.arange(m) * dt
    td_bg = np.zeros((len(bg_modes), m), dtype=complex)
    for i, mode_idx in enumerate(bg_modes):
        td_bg[i, :] = b[mode_idx] * np.exp(omega[mode_idx] * time_vec)
    
    X_bg = (Phi[:, bg_modes] @ td_bg).real
    X_fg = U_full[:, :m] - X_bg
    
    fg_video = X_fg.T.reshape(m, H, W)
    fg_video = np.clip(np.abs(fg_video), 0, 1)
    
    bg_video = X_bg.T.reshape(m, H, W)
    bg_video = np.clip(np.abs(bg_video), 0, 1)
    
    return fg_video, bg_video, len(bg_modes)

def proximal_nuclear(A, sigma):
    U_svd, s, Vh = np.linalg.svd(A, full_matrices=False)
    s_thresholded = np.maximum(s - sigma, 0)
    return U_svd @ np.diag(s_thresholded) @ Vh


def proximal_L1(X, sigma):
    return np.sign(X) * np.maximum(np.abs(X) - sigma, 0)


def RPCA(X, lam, max_iter=1000, verbose=False):
    n1, n2 = X.shape
    mu = n1 * n2 / (4 * np.sum(np.abs(X)))
    thresh = 1e-8 * np.linalg.norm(X)

    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)

    for i in range(max_iter):
        L = proximal_nuclear(X - S + Y / mu, 1 / mu)
        S = proximal_L1(X - L + Y / mu, lam / mu)
        Y = Y + mu * (X - L - S)
        residual = np.linalg.norm(X - L - S, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)

        if verbose and i % 50 == 0:
            print(f"[RPCA] Iter {i:4d}: res={residual:.3e}")

        if residual < thresh:
            if verbose:
                print(f"Converged at iter {i}")
            break

    return L, S

