using LinearAlgebra
using SparseArrays
using Random
using Printf

# ── Neumann series approximation of (I - A)^{-1} v ─────────────────────────
mutable struct NeumannCache
    result::Vector{Float64}
    term::Vector{Float64}
    term_next::Vector{Float64}
    NeumannCache(n::Int) = new(zeros(n), zeros(n), zeros(n))
end

function neumann_apply!(cache::NeumannCache, A_bar, v, k::Int=20)
    copyto!(cache.result, v)
    copyto!(cache.term, v)
    for _ in 1:k
        mul!(cache.term_next, A_bar, cache.term)
        cache.result .+= cache.term_next
        cache.term, cache.term_next = cache.term_next, cache.term
    end
    return cache.result
end

# ── Main solver ──────────────────────────────────────────────────────────────
function solve_planner(alpha_v, A_m, B_m, l_tilde_v, dK_v, K_v, L_total_f, G_v, gamma_v=nothing, C_prev_v=nothing;
                       k::Int=25, tol_p::Float64=1e-3, tol_d::Float64=1e-5, eta_K::Float64=0.2, eta_L::Float64=0.2, max_iter::Int=2000)

    eps_val = 1e-15
    n = length(alpha_v)
    cache = NeumannCache(n)
    A_m_T = sparse(A_m')

    if gamma_v === nothing; gamma_v = zeros(n); end

    Bt(v)  = B_m .* neumann_apply!(cache, A_m, v, k)
    BtT(w) = neumann_apply!(cache, A_m_T, B_m .* w, k)

    K_eff = K_v .- Bt(dK_v)
    L_eff = L_total_f - dot(l_tilde_v, dK_v)

    log_lam_K = log.(max.(BtT(alpha_v ./ max.(K_eff, eps_val)), eps_val))
    log_lam_L = log(sum(alpha_v) / max(n * L_eff, eps_val))

    log_y_K = copy(log_lam_K)
    log_y_L = log_lam_L

    prev_dtheta_K    = zeros(n)
    prev_dtheta_L    = 0.0
    prev_grad_K_iter = zeros(n)
    prev_grad_L_iter = 0.0

    C_res     = zeros(n)
    converged = false
    opt_iter  = 0
    tk_curr   = 1.0

    for iter in 1:max_iter
        opt_iter = iter
        y_K = exp.(log_y_K); y_L = exp(log_y_L)
        pi_vec = BtT(y_K) .+ y_L .* l_tilde_v
        C_res = gamma_v .+ alpha_v ./ max.(pi_vec, eps_val)

        s_K = K_eff .- Bt(C_res)
        s_L = L_eff  - dot(l_tilde_v, C_res)

        # 1. Preconditioning
        diag_H = (A_m .^ 2)' * (alpha_v ./ max.(pi_vec .^ 2, 1e-6))
        prec_K = max.(K_eff, 1e-3) .+ 0.1 .* diag_H
        grad_K_raw = .-s_K
        grad_L_raw = -s_L
        grad_K = grad_K_raw ./ prec_K
        grad_L = grad_L_raw / (abs(L_eff) + eps_val)

        # 2. Convergence Check
        lam_K_cur = exp.(log_lam_K); lam_L_cur = exp(log_lam_L)
        if all(abs.(lam_K_cur .* s_K) .<= tol_d) && abs(lam_L_cur * s_L) <= tol_d
            converged = true; break
        end

        # 3. Spectral Step
        bb_eta = 1.0
        if iter > 1
            dy_K_raw = grad_K_raw .- prev_grad_K_iter
            dy_L_raw = grad_L_raw  - prev_grad_L_iter
            dot_ss = dot(prev_dtheta_K, prec_K .* prev_dtheta_K) + prev_dtheta_L^2 * abs(L_eff)
            dot_sy = dot(prev_dtheta_K, dy_K_raw) + prev_dtheta_L * dy_L_raw
            if abs(dot_sy) > 1e-25
                bb_eta = 1.0 / (1.0 + abs(dot_sy / dot_ss))
            end
        end

        copyto!(prev_grad_K_iter, grad_K_raw)
        prev_grad_L_iter = grad_L_raw

        step_K = bb_eta .* grad_K
        step_L = bb_eta * grad_L
        copyto!(prev_dtheta_K, step_K)
        prev_dtheta_L = step_L

        log_lam_K_new = log_y_K .+ step_K
        log_lam_L_new = log_y_L  + step_L
        tk_next = (1.0 + sqrt(1.0 + 4.0 * tk_curr^2)) / 2.0
        beta_t  = (tk_curr - 1.0) / tk_next
        tk_curr = tk_next

        log_y_K .= log_lam_K_new .+ beta_t .* (log_lam_K_new .- log_lam_K)
        log_y_L  = log_lam_L_new  + beta_t  * (log_lam_L_new  - log_lam_L)
        log_lam_K .= log_lam_K_new
        log_lam_L  = log_lam_L_new
    end

    lam_K = exp.(log_lam_K)
    lam_L = exp(log_lam_L)
    pi_star = BtT(lam_K) .+ lam_L .* l_tilde_v
    X_star = neumann_apply!(cache, A_m, C_res .+ dK_v .+ G_v, k)

    return (C_star=C_res, X_star=X_star, pi_star=pi_star,
            success=converged, lambda_K=lam_K, lambda_L=lam_L,
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
