export ZetaMeanLogModel1

using BenchmarkTools, LinearAlgebra, SpecialFunctions, Distributions, Plots, Printf, DataFrames



function ZetaLogLik1(θ, z1)
    return -sum(z1 .* θ .+ log.(zeta.(θ)))
end

# Function to find the root of a certain function related to theta
function Rootθ(t, tol = 1e-8)
    f = t -> log(zeta(t))
    (s, eps) = (3.0, 1.0e-10) 
    df = -First(f, s) - t
    while df <= 0.0
        s = 1 / 2 + s / 2
        df = -First(f, s) - t
    end
    for iter in 1:100
        d2f = -Second(f, s)
        s = s - df / d2f
        df = -First(f, s) - t
        if abs(df) < tol
            break
        end
    end
    return s
end

function First(f::Function, s) # first derivative of f
    d = 1.0e-9
    return imag(f(s + d * im)) / d
end

function Second(f::Function, s) # second derivative of f
    (c, d) = ((1.0 + 1.0im) / sqrt(2.0), 1.0e-4)
    return imag(f(s + c * d) + f(s - c * d)) / d^2
end

# Function to calculate the gradient and fisher information matrix
function derivGradFisher1(μ, θ, z1, X)
    f = t -> log(zeta(t))
    s_vec = (z1 .- μ) .* μ ./ Second.(f, θ)  # Scoring vector
    Var1 = μ.^2 ./ Second.(f, θ)
    grad = X' * s_vec
    hess = X' * diagm(Var1) * X
    return grad, hess
end

# Main function for the Zeta Mean Log model
function ZetaMeanLogModel1(
        z::Vector, 
        W::Matrix; 
        max_iter::Int = 1000, 
        ∇tol::Float64 = 1e-8)
    

    W = StandardizeColumns(W)
    n = length(z)                # numbers of rows and columns
    W = hcat(repeat([1],n), W)   # add ones
    β = zeros(size(W)[2])
    log_lik = Float64[]  # Initialize the log - likelihood array
    iters = 0
    z1 = log.(z)
    μ = exp.(W * β)
    θ = Rootθ.(μ)
    log_likelihood = ZetaLogLik1(θ, z1)

    for iter in 1:max_iter
        iters += 1
        log_likelihood_old = log_likelihood
        β_old = β
        grad, hess = derivGradFisher1(μ, θ, z1, W)
        δ = pinv(hess) * grad  # generalized inverse matrix\
        # Line search
        flag_true = true
        while flag_true
            β = β_old + δ
            μ = exp.(W * β)
            θ = Rootθ.(μ)
            log_likelihood = ZetaLogLik1(θ, z1)
            if log_likelihood >= log_likelihood_old
                flag_true = false
            else
                δ *= 0.5
            end
        end
        # Check for convergence
        if norm(β - β_old) < 1e-8
            break
        end
        push!(log_lik, log_likelihood)
    end
    grad, hess = derivGradFisher1(μ, θ, z, W)
    β_0 = zeros(length(β))
    Wald_stat, std = wald_statistic(β, β_0, hess)
    p_values = p_value(Wald_stat)
    lower_bound, upper_bound = confidence_interval(β, std)
    function format_number(x)
        if abs(x) < 1e-4
            return @sprintf("%8.2e", x)  # Use scientific notation and ensure alignment for negative signs, width is 10
        else
            return @sprintf("%8.4f", x)  # Keep 4 decimal places and ensure alignment for negative signs, width is 10
        end
    end
    
    df = DataFrame(
        Parameter = 1:length(β),
        Estimate = [format_number(x) for x in β],
        StandardError = [format_number(x) for x in std],
        WaldStatistic = [format_number(x) for x in Wald_stat],
        CI_95 = ["($(format_number(l)), $(format_number(u)))" for (l, u) in zip(lower_bound, upper_bound)],
        p_value = [format_number(x) for x in p_values]
    )

    
    io = IOBuffer()
    pretty_table(io, df; backend = Val(:latex), tf = tf_latex_booktabs)
    latex_output = String(take!(io))
    
    println("Zeta Mean Log Model Summary:")
    println(latex_output)
    
    # Calculate BIC
    BIC = calculate_BIC(log_likelihood, β, z)

    return β, θ, iters, log_lik, BIC, df
end

