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
# --- CRRA Dual Solver (FISTA + Backtracking) ---
function solve_planner(alpha_v, A_m, B_m, l_tilde_v, dK_v, K_v, L_total_f, G_v, sigma_v, C_prev_v;
                       k::Int=25, tol_p::Float64=1e-3, tol_d::Float64=1e-4, 
                       eta_K::Float64=0.4, eta_L::Float64=0.5, max_iter::Int=2000,
                       L0::Float64=1.0, L_scale_up::Float64=1.5, L_scale_dn::Float64=1.05)

    eps_val = 1e-15
    n = length(alpha_v)
    cache = NeumannCache(n)
    A_m_T = sparse(A_m')

    Bt(v)  = B_m .* neumann_apply!(cache, A_m, v, k)
    BtT(w) = neumann_apply!(cache, A_m_T, B_m .* w, k)

    K_eff = K_v .- Bt(dK_v)
    L_eff = L_total_f - dot(l_tilde_v, dK_v)

    # Initialization
    pi_init = alpha_v ./ max.(C_prev_v, 1e-4).^sigma_v
    log_lam_K = log.(max.(BtT(pi_init), 1e-8))
    log_lam_L = log(sum(alpha_v) / max(n * L_eff, eps_val))

    log_y_K = copy(log_lam_K); log_y_L = log_lam_L
    log_x_K = copy(log_lam_K); log_x_L = log_lam_L
    log_x_K_new = copy(log_lam_K)
    
    tk_curr = 1.0; L_curr = L0
    converged = false; opt_iter = max_iter

    for iter in 1:max_iter
        opt_iter = iter
        y_K = exp.(log_y_K); y_L = exp(log_y_L)
        pi_vec = BtT(y_K) .+ y_L .* l_tilde_v
        C_res = (max.(alpha_v, 1e-15) ./ max.(pi_vec, eps_val)).^(1.0 ./ sigma_v)

        s_K = K_eff .- Bt(C_res)
        s_L = L_eff  - dot(l_tilde_v, C_res)

        # Grad check
        grad_K = .-s_K ./ max.(K_eff, 1e-12)
        grad_L = -s_L / max(abs(L_total_f), 1e-12)

        if all(abs.(y_K .* s_K) .<= tol_d) && abs(y_L * s_L) <= tol_d
            converged = true; break
        end

        # Dual objective evaluation
        obj_y = dot(y_K, K_eff) + y_L * L_eff
        for i in 1:n
            sig = sigma_v[i]; al = alpha_v[i]; p = max(pi_vec[i], eps_val)
            if abs(sig - 1.0) < 1e-6
                obj_y += al * log(max(al, eps_val) / p) - al
            else
                obj_y += (sig / (1.0 - sig)) * al^(1.0/sig) * p^(1.0 - 1.0/sig)
            end
        end

        L_curr /= L_scale_dn
        local log_x_L_new
        while true
            log_x_K_new .= log_y_K .+ (eta_K / L_curr) .* grad_K
            log_x_L_new  = log_y_L  + (eta_L / L_curr)  * grad_L

            trial_K = exp.(log_x_K_new); trial_L = exp(log_x_L_new)
            pi_trial = BtT(trial_K) .+ trial_L .* l_tilde_v
            
            obj_trial = dot(trial_K, K_eff) + trial_L * L_eff
            for i in 1:n
                sig = sigma_v[i]; al = alpha_v[i]; p = max(pi_trial[i], eps_val)
                if abs(sig - 1.0) < 1e-6
                    obj_trial += al * log(max(al, eps_val) / p) - al
                else
                    obj_trial += (sig / (1.0 - sig)) * al^(1.0/sig) * p^(1.0 - 1.0/sig)
                end
            end

            if obj_trial <= obj_y || L_curr > 1e6; break; end
            L_curr *= L_scale_up
        end

        # Adaptive momentum
        if dot(log_y_K .- log_x_K, grad_K) + (log_y_L - log_x_L)*grad_L < 0
            tk_curr = 1.0
        end

        tk_next = (1.0 + sqrt(1.0 + 4.0 * tk_curr^2)) / 2.0
        beta_t  = (tk_curr - 1.0) / tk_next
        log_y_K .= log_x_K_new .+ beta_t .* (log_x_K_new .- log_x_K)
        log_y_L  = log_x_L_new  + beta_t  * (log_x_L_new  - log_x_L)
        log_x_K .= log_x_K_new; log_x_L = log_x_L_new
        tk_curr = tk_next
    end

    lam_K = exp.(log_x_K); lam_L = exp(log_x_L)
    pi_star = BtT(lam_K) .+ lam_L .* l_tilde_v
    C_star = (max.(alpha_v, 1e-15) ./ max.(pi_star, eps_val)).^(1.0 ./ sigma_v)
    X_star = neumann_apply!(cache, A_m, C_star .+ dK_v .+ G_v, k)

    return (C_star=C_star, X_star=X_star, pi_star=pi_star, success=converged, iterations=opt_iter)
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
    
    sigma = rand(n) .* 0.5 .+ 0.75 # sigma in [0.75, 1.25]
    C_init = rand(n) .* 1.0 .+ 0.5

    return (alpha=alpha, A_bar=A_bar, B=B, l_tilde=l_tilde,
            dK=dK, K=K, L_total=L_total, G_vec=G, sigma=sigma, C_init=C_init)
end

# ── Benchmark ────────────────────────────────────────────────────────────────
# ── Final Benchmarking Loop & Plotting ────────────────────────────────────────
using Plots

function run_scaling_study()
    scales = 1000:1000:10000
    n_trials = 5
    avg_times = Float64[]

    println("Starting Scaling Study (CRRA Solver)...")
    println("Trials per scale: $n_trials")
    println("\nN Sectors | Avg Time (s)")
    println("-----------------------")

    for n in scales
        data = make_synthetic(n)
        # Warmup for this N
        solve_planner(data.alpha, data.A_bar, data.B, data.l_tilde, 
                      data.dK, data.K, data.L_total, data.G_vec, data.sigma, data.C_init)
        
        t_total = 0.0
        for _ in 1:n_trials
            t0 = time()
            solve_planner(data.alpha, data.A_bar, data.B, data.l_tilde, 
                          data.dK, data.K, data.L_total, data.G_vec, data.sigma, data.C_init)
            t_total += (time() - t0)
        end
        avg_t = t_total / n_trials
        push!(avg_times, avg_t)
        @printf("%-10d | %-12.4f\n", n, avg_t)
    end

    # Plotting
    p = plot(scales, avg_times, marker=(:circle, 5), lw=2,
             title="Dual Solver Complexity Scaling (CRRA)",
             xlabel="Number of Sectors (n)",
             ylabel="Wall-clock Time (s)",
             label="Avg Time (5 trials)",
             grid=true, legend=:topleft,
             color=:blue, dpi=300)
    
    # Quadratic Regression: y = a*n^2 + b*n + c
    # Form design matrix X
    X = [ones(length(scales)) collect(scales) collect(scales).^2]
    y = avg_times
    # Solve via normal equations: beta = (X'X) \ (X'y)
    beta = (X' * X) \ (X' * y)
    c, b, a = beta

    plot!(scales, c .+ b .* scales .+ a .* scales.^2, 
          ls=:dash, color=:orange, lw=2, label="Quadratic O(n^2) fit")

    @printf("\nQuadratic Fit: t(n) = %.2e n^2 + %.2e n + %.2e\n", a, b, c)

    out_path = "scaling_crra.png"
    savefig(p, out_path)
    println("Plot saved to $out_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_scaling_study()
end
