import numpy as np
import scipy.sparse as sp
import time

# ── Neumann series approximation of (I - A)^{-1} v ─────────────────────────
def neumann_apply(A, v, k=20):
    """Compute (I + A + A^2 + ... + A^k) v via Horner's method."""
    # Ensure v is a dense array for efficient sparse @ dense
    v = np.asarray(v)
    result = v.copy()
    term   = v.copy()
    for _ in range(k):
        term   = A @ term
        result = result + term
    return result

# ── Main solver ──────────────────────────────────────────────────────────────
def solve_planner(alpha, A_bar, B, l_tilde, dK, K, L_total, G_vec, gamma=None,
                  C_prev=None,
                  k=20,
                  tol=1e-4,
                  eta_K=0.4, eta_L=0.4,
                  max_iter=2000):

    eps_val = 1e-15
    n       = len(alpha)

    # Default LES subsistence thresholds to zero if not provided
    if gamma is None:
        gamma = np.zeros(n)

    # B operator  Bt(v)  = B * (I-A)^{-1} v
    # Adjoint     BtT(w) = (I-A)^{-T} * B^T w   [B diagonal => B^T = B]
    def Bt(v):
        return B * neumann_apply(A_bar, v, k)

    # Cache transpose for efficiency in sparse context
    A_bar_T = A_bar.transpose()
    def BtT(w):
        return neumann_apply(A_bar_T, B * w, k)

    # Effective resources net of committed investment
    K_eff  = K - Bt(dK)
    L_eff  = L_total - np.dot(l_tilde, dK)

    active     = K_eff > eps_val
    Kv         = K_eff[active]
    alpha_act  = alpha[active]
    gamma_act  = gamma[active]
    l_act      = l_tilde[active]
    n_act      = int(active.sum())

    def Bt_act(v_act):
        v_full = np.zeros(n); v_full[active] = v_act
        return Bt(v_full)[active]

    def BtT_act(w_act):
        w_full = np.zeros(n); w_full[active] = w_act
        return BtT(w_full)[active]

    # Cold / warm start
    if C_prev is not None:
        p_guess = alpha / np.maximum(C_prev, eps_val)
        rhs     = p_guess - A_bar_T @ p_guess
        lam_K   = np.maximum(rhs[active] / np.maximum(B[active], eps_val), eps_val)
    else:
        lam_K = np.full(n_act, (alpha_act.sum() / n_act) / (Kv.sum() / n_act))

    lam_L = (alpha_act.sum() / n_act) / max(L_eff, eps_val)

    local_eta_K = np.ones(n_act)
    prev_s_K    = np.zeros(n_act)
    local_eta_L = 1.0
    prev_s_L    = 0.0

    C_act     = np.zeros(n_act)
    converged = False
    opt_iter  = 0

    for it in range(1, max_iter + 1):
        opt_iter = it

        # LES stationarity -> closed-form C = gamma + alpha / pi
        pi_vec = BtT_act(lam_K) + lam_L * l_act
        C_act  = gamma_act + alpha_act / np.maximum(pi_vec, eps_val)

        # Slacks
        s_K = Kv   - Bt_act(C_act)
        s_L = L_eff - np.dot(l_act, C_act)

        # KKT check
        primal_ok_K = np.all(s_K / (Kv + eps_val) >= -tol)
        primal_ok_L = (s_L / (abs(L_eff) + eps_val)) >= -tol
        comp_ok_K   = np.all(np.abs(lam_K * s_K) <= tol)
        comp_ok_L   = abs(lam_L * s_L) <= tol

        if primal_ok_K and primal_ok_L and comp_ok_K and comp_ok_L:
            converged = True
            break

        # Adaptive step size
        if it > 1:
            flip_K          = s_K * prev_s_K < 0
            local_eta_K     = np.where(flip_K,
                                       np.maximum(local_eta_K * 0.5, 0.1),
                                       np.minimum(local_eta_K * 1.1, 5.0))
            local_eta_L     = (local_eta_L * 0.5 if s_L * prev_s_L < 0
                               else min(local_eta_L * 1.1, 5.0))

        prev_s_K = s_K.copy()
        prev_s_L = s_L

        exp_K = np.clip(eta_K * local_eta_K * (-s_K) / (Kv + eps_val),        -20, 20)
        exp_L = np.clip(eta_L * local_eta_L * (-s_L) / (abs(L_eff) + eps_val), -20, 20)
        lam_K = np.maximum(lam_K * np.exp(exp_K), eps_val)
        lam_L = max(lam_L * np.exp(exp_L), eps_val)

    # Recovery
    lam_K_full          = np.zeros(n)
    lam_K_full[active]  = lam_K
    pi_star             = BtT(lam_K_full) + lam_L * l_tilde

    C_star          = np.zeros(n)
    C_star[active]  = C_act
    X_star          = neumann_apply(A_bar, C_star + dK + G_vec, k)

    return dict(C_star=C_star, X_star=X_star, pi_star=pi_star,
                success=converged, lambda_K=lam_K_full, lambda_L=lam_L,
                iterations=opt_iter)


# ── Synthetic data generator ─────────────────────────────────────────────────
def make_synthetic(n, sparsity=0.15, seed=42):
    rng = np.random.default_rng(seed)

    # Sparse A with spectral radius < 0.65
    mask = rng.random((n, n)) < sparsity
    A_raw = rng.uniform(0, 1, (n, n)) * mask
    A_raw_sp = sp.csr_matrix(A_raw)
    
    # Sparse spectral radius estimation
    if n > 1:
        vals = sp.linalg.eigs(A_raw_sp, k=1, which='LM', return_eigenvectors=False)
        sr = np.abs(vals[0])
    else:
        sr = A_raw[0,0]
        
    A = A_raw_sp * (0.60 / max(sr, 1e-10))

    # Depreciation-augmented matrix
    delta = 0.0125
    B     = rng.uniform(0.05, 0.3, n)    # diagonal capital coefficients
    
    # Create sparse A_bar: CSR is efficient for matrix-vector multiplication
    A = sp.csr_matrix(A)
    A_bar = A + delta * sp.diags(B)

    alpha  = rng.dirichlet(np.ones(n))
    l_vec  = rng.uniform(0.1, 0.5, n)
    L_total = n * 0.3                    # aggregate labour endowment

    # l_tilde = (I - A_bar)^{-T} l
    # For small synthetic n, we can use sparse solve, but for benchmark 
    # we usually hold l_tilde as dense.
    I_sparse = sp.eye(n, format='csr')
    l_tilde = sp.linalg.spsolve((I_sparse - A_bar).T, l_vec)

    # Capital stock: K = B * X_0, X_0 ~ uniform
    X0  = rng.uniform(1, 3, n)
    K   = B * X0 * 1.2                  # 20% buffer

    dK  = rng.uniform(0.01, 0.05, n)
    G   = rng.uniform(0.05, 0.15, n)

    return dict(alpha=alpha, A_bar=A_bar, B=B, l_tilde=l_tilde,
                dK=dK, K=K, L_total=L_total, G_vec=G)


# ── Benchmark ────────────────────────────────────────────────────────────────
def run_benchmark(n, n_trials=5):
    print(f"\n{'='*55}")
    print(f"  Benchmark: n = {n} sectors, {n_trials} trials")
    print(f"{'='*55}")

    data = make_synthetic(n)
    times      = []
    iterations = []
    successes  = []

    C_prev = None
    for t in range(n_trials):
        t0  = time.perf_counter()
        res = solve_planner(
            data['alpha'], data['A_bar'], data['B'],
            data['l_tilde'], data['dK'], data['K'],
            data['L_total'], data['G_vec'],
            C_prev=C_prev
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        iterations.append(res['iterations'])
        successes.append(res['success'])
        C_prev = res['C_star']   # warm start next trial

    print(f"\n  {'Trial':<8} {'Time (s)':<12} {'Iterations':<12} {'Converged'}")
    print(f"  {'-'*45}")
    for i, (t, it, s) in enumerate(zip(times, iterations, successes)):
        warm = " (warm)" if i > 0 else " (cold)"
        print(f"  {i+1:<8} {t:<12.4f} {it:<12} {s}{warm}")

    print(f"\n  Mean time      : {np.mean(times):.4f}s")
    print(f"  Min  time      : {np.min(times):.4f}s")
    print(f"  Mean iterations: {np.mean(iterations):.1f}")
    print(f"  Min  iterations: {np.min(iterations)}")
    print(f"  All converged  : {all(successes)}")

    # Verify material balance
    r = res
    I_sparse = sp.eye(n, format='csr')
    net_output = (I_sparse - data['A_bar']) @ r['X_star']
    final_demand = r['C_star'] + data['dK'] + data['G_vec']
    balance_err  = np.max(np.abs(net_output - final_demand))
    print(f"  Material balance error : {balance_err:.2e}")

    return times, iterations


if __name__ == "__main__":
    run_benchmark(10000, n_trials=3)