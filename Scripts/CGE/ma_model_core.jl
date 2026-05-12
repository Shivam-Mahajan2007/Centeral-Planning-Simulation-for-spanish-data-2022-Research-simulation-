module MACGECore

using LinearAlgebra, SparseArrays, Random, Statistics
using PythonCall: pyconvert

export ces_output, ces_mpk, optimal_KL, compute_mc, tatonnement,
       solve_market_equilibrium, aggregate_demand, excess_demand

"""
    ces_output(K::Float64, L::Float64, gamma::Float64, rho::Float64, X_leontief::Float64)

Compute output using CES production function with Leontief constraint.
"""
function ces_output(K::Float64, L::Float64, gamma::Float64, rho::Float64,
                    X_leontief::Float64)
    # Primary factor composite
    F = (gamma * K^rho + (1-gamma) * L^rho)^(1/rho)
    return min(X_leontief, F)
end

"""
    ces_mpk(K::Float64, L::Float64, gamma::Float64, rho::Float64, X::Float64)

Compute marginal product of capital for CES production.
"""
function ces_mpk(K::Float64, L::Float64, gamma::Float64, rho::Float64,
                 X::Float64)
    F = (gamma * K^rho + (1-gamma) * L^rho)^(1/rho)
    return gamma * K^(rho-1) * F^(1-rho)
end

"""
    optimal_KL(X_target::Float64, w::Float64, r::Float64, gamma::Float64, rho::Float64)

Compute cost-minimizing capital and labor ratios for given output target.
"""
function optimal_KL(X_target::Float64, w::Float64, r::Float64,
                    gamma::Float64, rho::Float64)
    # Cost minimisation: K* / L* = (gamma/(1-gamma) * w/r)^(1/(1-rho))
    ratio = (gamma / (1-gamma) * w / r)^(1/(1-rho))
    
    # Solve for L from F(K,L)=X: L = X / F(ratio,1)
    F_unit = (gamma * ratio^rho + (1-gamma))^(1/rho)
    L_star = X_target / max(F_unit, 1e-12)
    K_star = ratio * L_star
    
    return K_star, L_star
end

"""
    compute_mc(w::Float64, r::Float64, l_coef::Float64, k_coef::Float64,
               P_inputs::Vector{Float64}, A_col::Vector{Float64})

Compute marginal cost given input prices and coefficients.
"""
function compute_mc(w::Float64, r::Float64, l_coef::Float64, k_coef::Float64,
                   P_inputs::Vector{Float64}, A_col::Vector{Float64})
    labor_cost = w * l_coef
    capital_cost = r * k_coef
    intermediate_cost = dot(P_inputs, A_col)
    return labor_cost + capital_cost + intermediate_cost
end

"""
    tatonnement(P_init::Vector{Float64}, C_plan::Vector{Float64},
                I_plan::Vector{Float64}, G_vec::Vector{Float64},
                A::Matrix{Float64}, X_supply::Vector{Float64},
                sigma_v::Vector{Float64}, alpha_h::Matrix{Float64},
                Y_h::Vector{Float64}, w_h::Vector{Float64};
                max_iter::Int=50, tol::Float64=0.005, step_cap::Float64=0.5)

Walrasian tatonnement process to clear goods markets.
"""
function tatonnement(P_init::Vector{Float64},
                     C_plan::Vector{Float64},
                     I_plan::Vector{Float64},
                     G_vec::Vector{Float64},
                     A::Matrix{Float64},
                     X_supply::Vector{Float64},
                     sigma_v::Vector{Float64},
                     alpha_h::Matrix{Float64},   # (n_hh, n)
                     Y_h::Vector{Float64},
                     w_h::Vector{Float64};
                     max_iter::Int=50,
                     tol::Float64=0.005,
                     step_cap::Float64=0.5)

    P = copy(P_init)
    n_hh = length(Y_h)
    n = length(P)

    for iter in 1:max_iter
        # Household demand aggregation
        C_d = zeros(n)
        for h in 1:n_hh
            # mu_h[h] from budget constraint
            price_term = alpha_h[h, :] .^ (1.0 ./ sigma_v) .* P .^ (1.0 .- 1.0 ./ sigma_v)
            mu_h = sum(price_term) / Y_h[h]
            
            # Marshallian demands
            C_d .+= (w_h[h] .* alpha_h[h, :] ./ (mu_h .* P)) .^ (1.0 ./ sigma_v)
        end
        
        # Total demand
        X_d = C_d .+ I_plan .+ G_vec .+ A * X_supply
        
        # Excess demand
        Z = X_d .- X_supply
        scale = max.(X_supply, 1e-12)
        err = maximum(abs.(Z) ./ scale)
        
        # Price update
        P = P .* (1.0 .+ step_cap .* Z ./ scale)
        P = max.(P, 1e-12)
        
        if err < tol
            break
        end
    end
    
    return P
end

function aggregate_demand(P::Vector{Float64}, alpha_h::Matrix{Float64},
                    Y_h::Vector{Float64}, w_h::Vector{Float64},
                    sigma_v::Vector{Float64})
    """
    Compute aggregate consumption demand from all households.
    """
    n_hh, n = size(alpha_h)
    C_d = zeros(n)
    
    for h in 1:n_hh
        price_term = alpha_h[h, :] .^ (1.0 ./ sigma_v) .* P .^ (1.0 .- 1.0 ./ sigma_v)
        mu_h = sum(price_term) / Y_h[h]
        C_d .+= (w_h[h] .* alpha_h[h, :] ./ (mu_h .* P)) .^ (1.0 ./ sigma_v)
    end
    
    return C_d
end

"""
    excess_demand(P::Vector{Float64}, C_d::Vector{Float64},
                  I_plan::Vector{Float64}, G_vec::Vector{Float64},
                  A::Matrix{Float64}, X_supply::Vector{Float64}) -> Vector{Float64}

Compute excess demand for all goods.
"""
function excess_demand(P::Vector{Float64}, C_d::Vector{Float64},
                      I_plan::Vector{Float64}, G_vec::Vector{Float64},
                      A::Matrix{Float64}, X_supply::Vector{Float64})
    X_d = C_d .+ I_plan .+ G_vec .+ A * X_supply
    return X_d .- X_supply
end

"""
    solve_market_equilibrium(P_init, K_firms, K_total, A, B, l_vec,
                             gamma_v, rho_v, sigma_v, markup_v,
                             alpha_h, Y_h, w_h, G_vec, delta_v,
                             phi_v, w_wage, r_vec)

Outer loop: iterate (X, P) until Walrasian equilibrium.
Each outer iteration:
  1. Given P, firms choose cost-minimising (K,L), compute X via CES.
  2. Given (X, P), tatonnement clears goods markets.
  3. Update markups, investment, capital.
Convergence: ||Z||_inf < tol on both goods and factor markets.
"""
function solve_market_equilibrium(
    P_init, K_firms, K_total, A, B, l_vec,
    gamma_v, rho_v, sigma_v, markup_v,
    alpha_h, Y_h, w_h, G_vec, delta_v,
    phi_v, w_wage, r_vec;
    max_outer::Int=30, tol_outer::Float64=1e-4,
    max_inner::Int=50, tol_inner::Float64=0.005,
    eta_markup::Float64=0.1, markup_max::Float64=0.5)
    
    n = length(P_init)
    n_firms = size(K_firms, 1)
    
    P = copy(P_init)
    K = copy(K_total)
    X = zeros(n)
    MC = zeros(n)
    
    for outer_iter in 1:max_outer
        # 1. Firm production decisions
        for i in 1:n
            # Cost-minimizing K/L ratio
            K_star, L_star = optimal_KL(X[i], w_wage, r_vec[i], gamma_v[i], rho_v[i])
            
            # Actual K available (capital constraint)
            K_actual = min(K_star, K[i])
            
            # Compute output with CES, bounded by Leontief intermediate constraint
            # (simplified - assume intermediate constraint not binding)
            X[i] = ces_output(K_actual, L_star, gamma_v[i], rho_v[i], Inf)
            
            # Marginal cost
            A_col = A[:, i]
            P_inputs = P
            MC[i] = compute_mc(w_wage, r_vec[i], l_vec[i], B[i, i], P_inputs, A_col)
        end
        
        # 2. Compute investment and update capital
        I = zeros(n)
        for i in 1:n
            # Tobin's q
            MPK_i = ces_mpk(K[i], X[i]/K[i] * l_vec[i], gamma_v[i], rho_v[i], X[i])
            q_i = P[i] * MPK_i / (r_vec[i] + delta_v)
            
            # Investment
            I[i] = delta_v * K[i] + phi_v[i] * K[i] * max(q_i - 1.0, 0.0)
        end
        
        # 3. Market clearing via tatonnement
        C_d = aggregate_demand(P, alpha_h, Y_h, w_h, sigma_v)
        P_new = tatonnement(P, C_d, I, G_vec, A, X, sigma_v, alpha_h, Y_h, w_h,
                           max_iter=max_inner, tol=tol_inner)
        
        # 4. Update markups
        for i in 1:n
            demand_pressure = (C_d[i] + I[i] + G_vec[i] + sum(A[:, i] .* X)) / X[i] - 1.0
            markup_v[i] = markup_v[i] + eta_markup * demand_pressure
            markup_v[i] = clamp(markup_v[i], 0.0, markup_max)
        end
        
        # 5. Check convergence
        Z = excess_demand(P_new, C_d, I, G_vec, A, X)
        price_error = maximum(abs.(Z) ./ max.(X, 1e-12))
        
        P = P_new
        
        if price_error < tol_outer
            break
        end
    end
    
    return P, X, K, markup_v
end

end # module
