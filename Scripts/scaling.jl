using LinearAlgebra
using SparseArrays
using Random
using Printf

# ── Neumann series approximation of (I - A)^{-1} v ─────────────────────────
function neumann_apply(A_bar, v, k::Int=20)
    result = copy(v)
    term   = copy(v)
    term_next = similar(v)
    for _ in 1:k
        mul!(term_next, A_bar, term)
        result .+= term_next
        # Swap term and term_next to avoid allocating
        term, term_next = term_next, term
    end
    return result
end

# ── Main solver ──────────────────────────────────────────────────────────────
function solve_planner(alpha, A_bar, B, l_tilde, dK, K, L_total, G_vec, gamma=nothing, C_prev=nothing;
                       k::Int=20, tol_p::Float64=1e-3, tol_d::Float64=1e-4, eta_K::Float64=0.4, eta_L::Float64=0.4, max_iter::Int=2000)

    eps_val = 1e-15
    n = length(alpha)

    if gamma === nothing
        gamma = zeros(n)
    end

    function Bt(v)
        return B .* neumann_apply(A_bar, v, k)
    end

    A_bar_T = sparse(A_bar')
    function BtT(w)
        return neumann_apply(A_bar_T, B .* w, k)
    end

    K_eff = K .- Bt(dK)
    L_eff = L_total - dot(l_tilde, dK)

    active = K_eff .> eps_val
    Kv = K_eff[active]
    alpha_act = alpha[active]
    gamma_act = gamma[active]
    l_act = l_tilde[active]
    n_act = sum(active)

    function Bt_act(v_act)
        v_full = zeros(n)
        v_full[active] .= v_act
        return Bt(v_full)[active]
    end

    function BtT_act(w_act)
        w_full = zeros(n)
        w_full[active] .= w_act
        return BtT(w_full)[active]
    end

    if C_prev !== nothing
        p_guess = alpha_act ./ max.(C_prev[active] .- gamma_act, eps_val)
        w_guess = zeros(n)
        w_guess[active] .= p_guess
        w_res = w_guess .- A_bar_T * w_guess
        lam_K = max.(w_res[active], eps_val)
    else
        lam_K = fill((sum(alpha_act) / n_act) / max((sum(Kv) / n_act), eps_val), n_act)
        lam_K = max.(lam_K, eps_val)
    end

    lam_L = (sum(alpha_act) / n_act) / max(L_eff, eps_val)

    log_lam_K = log.(lam_K)
    log_lam_L = log(lam_L)

    log_y_K = copy(log_lam_K)
    log_y_L = log_lam_L

    prev_grad_K = zeros(n_act)
    prev_grad_L = 0.0
    prev_dtheta_K = zeros(n_act)
    prev_dtheta_L = 0.0

    eta_K_cur = eta_K
    eta_L_cur = eta_L

    C_act = zeros(n_act)
    converged = false
    opt_iter = 0

    for it in 1:max_iter
        opt_iter = it
        
        y_K = exp.(log_y_K)
        y_L = exp(log_y_L)

        pi_vec = BtT_act(y_K) .+ y_L .* l_act
        C_act = gamma_act .+ alpha_act ./ max.(pi_vec, eps_val)

        s_K = Kv .- Bt_act(C_act)
        s_L = L_eff - dot(l_act, C_act)

        # Primal violations: |slack/constraint| <= tol_p (only for negative slacks)
        primal_ok_K = all(abs.(min.(0.0, s_K)) ./ (Kv .+ eps_val) .<= tol_p)
        primal_ok_L = (abs(min(0.0, s_L)) / (abs(L_eff) + eps_val)) <= tol_p

        # Dual complementarity violations: exactly \lambda^(t) * s <= tol_d
        lam_K_cur = exp.(log_lam_K)
        lam_L_cur = exp.(log_lam_L)
        
        comp_ok_K = all(abs.(lam_K_cur .* s_K) .<= tol_d)
        comp_ok_L = abs(lam_L_cur * s_L) <= tol_d

        if primal_ok_K && primal_ok_L && comp_ok_K && comp_ok_L
            converged = true
            break
        end

        grad_K = (.-s_K) ./ (Kv .+ eps_val)
        grad_L = (-s_L) / (abs(L_eff) + eps_val)

        if it > 1
            dy_K = grad_K .- prev_grad_K
            dot_ss = dot(prev_dtheta_K, prev_dtheta_K) + prev_dtheta_L^2
            dot_sg = dot(prev_dtheta_K, dy_K) + prev_dtheta_L * (grad_L - prev_grad_L)
            if abs(dot_sg) > 1e-30
                bb_eta = clamp(abs(dot_ss / dot_sg), 0.01, 0.5)
                eta_K_cur = bb_eta
                eta_L_cur = bb_eta
            end
        end

        prev_grad_K .= grad_K
        prev_grad_L = grad_L

        step_K = eta_K_cur .* grad_K
        step_L = eta_L_cur * grad_L

        prev_dtheta_K .= step_K
        prev_dtheta_L = step_L

        log_lam_K_new = log_y_K .+ step_K
        log_lam_L_new = log_y_L + step_L

        # Nesterov momentum update
        beta_t = (it - 1.0) / (it + 2.0)
        log_y_K = log_lam_K_new .+ beta_t .* (log_lam_K_new .- log_lam_K)
        log_y_L = log_lam_L_new + beta_t * (log_lam_L_new - log_lam_L)

        log_lam_K .= log_lam_K_new
        log_lam_L = log_lam_L_new
    end
    
    lam_K = exp.(log_lam_K)
    lam_L = exp(log_lam_L)

    lam_K_full = zeros(n)
    lam_K_full[active] .= lam_K
    pi_star = BtT(lam_K_full) .+ lam_L .* l_tilde

    C_star = zeros(n)
    C_star[active] .= C_act
    X_star = neumann_apply(A_bar, C_star .+ dK .+ G_vec, k)

    return (C_star=C_star, X_star=X_star, pi_star=pi_star,
            success=converged, lambda_K=lam_K_full, lambda_L=lam_L,
            iterations=opt_iter)
end

# ── Synthetic data generator ─────────────────────────────────────────────────
function power_iteration(A::SparseMatrixCSC, max_iter=20)
    n = size(A, 1)
    b_k = rand(n)
    for _ in 1:max_iter
        b_k1 = A * b_k
        b_k = b_k1 ./ norm(b_k1)
    end
    return dot(b_k, A * b_k) / dot(b_k, b_k)
end

function make_synthetic(n::Int; sparsity=0.15, seed=42)
    Random.seed!(seed)

    # Sparse A with spectral radius ~ 0.60
    A_raw = sprand(n, n, sparsity)
    
    sr = abs(power_iteration(A_raw))
    A = A_raw * (0.60 / max(sr, 1e-10))

    delta = 0.0125
    B = rand(n) .* 0.25 .+ 0.05
    A_bar = A + delta * sparse(Diagonal(B))

    alpha = rand(n)
    alpha ./= sum(alpha)

    l_vec = rand(n) .* 0.4 .+ 0.1
    L_total = n * 0.3

    I_sparse = sparse(I, n, n)
    l_tilde = (I_sparse - A_bar)' \ l_vec

    X0 = rand(n) .* 2.0 .+ 1.0
    K = B .* X0 .* 1.2

    dK = rand(n) .* 0.04 .+ 0.01
    G = rand(n) .* 0.10 .+ 0.05

    return (alpha=alpha, A_bar=A_bar, B=B, l_tilde=l_tilde,
            dK=dK, K=K, L_total=L_total, G_vec=G)
end

# ── Benchmark ────────────────────────────────────────────────────────────────
function run_benchmark(n::Int; n_trials::Int=3)
    println("\n=======================================================")
    @printf("  Benchmark: n = %d sectors, %d trials\n", n, n_trials)
    println("=======================================================")

    data = make_synthetic(n)
    times = Float64[]
    iterations = Int[]
    successes = Bool[]

    C_prev = nothing
    local final_res = nothing
    
    # Warm-up precompilation on a tiny task so it doesn't pollute Trial 1
    println("\n  (Warming up JIT compiler...)")
    warmup_data = make_synthetic(20)
    solve_planner(warmup_data.alpha, warmup_data.A_bar, warmup_data.B, warmup_data.l_tilde, 
                  warmup_data.dK, warmup_data.K, warmup_data.L_total, warmup_data.G_vec)
    
    println("\n  Trial    Time (s)     Iterations   Converged")
    println("  ---------------------------------------------")

    for t in 1:n_trials
        t0 = time()
        res = solve_planner(data.alpha, data.A_bar, data.B,
                            data.l_tilde, data.dK, data.K,
                            data.L_total, data.G_vec, nothing, C_prev)
        elapsed = time() - t0
        push!(times, elapsed)
        push!(iterations, res.iterations)
        push!(successes, res.success)
        C_prev = res.C_star
        final_res = res
        
        warm_str = t > 1 ? "(warm)" : "(cold)"
        @printf("  %-8d %-12.4f %-12d %s %s\n", t, elapsed, res.iterations, res.success, warm_str)
    end

    @printf("\n  Mean time      : %.4fs\n", sum(times) / n_trials)
    @printf("  Min  time      : %.4fs\n", minimum(times))
    @printf("  Mean iterations: %.1f\n", sum(iterations) / n_trials)
    @printf("  Min  iterations: %d\n", minimum(iterations))
    println("  All converged  : ", all(successes))

    # Verify material balance
    I_sparse = sparse(I, n, n)
    net_output = (I_sparse - data.A_bar) * final_res.X_star
    final_demand = final_res.C_star .+ data.dK .+ data.G_vec
    balance_err = maximum(abs.(net_output .- final_demand))
    @printf("  Material balance error : %.2e\n", balance_err)

    return times, iterations
end

# Automatically run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark(10000, n_trials=3)
end
