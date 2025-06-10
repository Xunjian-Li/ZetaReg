using BenchmarkTools, LinearAlgebra, SpecialFunctions, Distributions, Random, Plots

function StandardizeColumns(A::AbstractMatrix)
    col_means = mean(A, dims=1)
    col_std = std(A, dims=1)
    return (A .- col_means) ./ col_std
end

function ZetaLogLik(θ, z_1, n_1)
    return -sum(z_1 .* θ .+ n_1 .* log.(zeta.(θ)))
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
function derivGradFisher(μ, θ, z1, n_1, X)
    f = t -> log(zeta(t))
    s_vec = (z1 .- n_1 .* μ) .* μ ./ Second.(f, θ)  # Scoring vector
    Var1 = n_1 .* μ.^2 ./ Second.(f, θ)
    grad = X' * s_vec
    hess = X' * diagm(Var1) * X
    return grad, hess
end

# Main function for the Zeta Mean Log model with fisher scoring method
function ZetaMeanLogModel(
        Z::Matrix, 
        W::Matrix; 
        max_iter::Int = 1000, 
        ∇tol::Float64 = 1e-8)
    
    W = StandardizeColumns(W)
    
    nn1, nn2 = size(Z)          # numbers of rows and columns
    W = hcat(repeat([1],nn2), W) # add ones
    z = [i for i in 1:nn1, _ in 1:nn2]
    z1 = vec(sum(Z.*log.(z), dims=1))
    n = vec(sum(Z, dims=1))
    β = zeros(size(W)[2])
    
    log_lik = Float64[]  # Initialize the log - likelihood array
    iters = 0
    
    μ = exp.(W * β)
    θ = Rootθ.(μ)
    log_likelihood = ZetaLogLik(θ, z1, n)

    for iter in 1:max_iter
        iters += 1
        log_likelihood_old = log_likelihood
        β_old = β

        grad, hess = derivGradFisher(μ, θ, z1, n, W)
        δ = pinv(hess) * grad  # pinv(.) is used to calculate the generalized inverse matrix

        # Line search
        flag_true = true
        while flag_true
            β = β_old + δ
            μ = exp.(W * β)
            θ = Rootθ.(μ)
            log_likelihood = ZetaLogLik(θ, z1, n)

            if log_likelihood >= log_likelihood_old
                flag_true = false
            else
                δ *= 0.5
            end
        end

        # Check for convergence
        if norm(β - β_old) < ∇tol
            break
        end
        push!(log_lik, log_likelihood)
    end
    
    return β, θ, iters, log_lik
end


Random.seed!(42)

struct Insurance
    num_insurances::Matrix{Int}
    age::Matrix{Int}
end

InsuranceData = Insurance(
    [94 342 590 433 177 59;
     6 34 66 59 30 12;
     0 5 16 11 6 8;
     0 3 8 7 2 2;
     0 0 0 3 4 2;
     0 2 3 3 0 0;
     0 0 2 2 0 0;
     0 0 1 1 1 0;
     0 0 1 0 0 0;
     0 0 0 1 0 0;
     0 0 2 0 0 0;
     0 0 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 0],
    reshape([20, 30, 40, 50, 60, 70], 6, 1)
)

Z = InsuranceData.num_insurances
W = InsuranceData.age

(rho3, θ3, iters3, loglikelihoods3) = ZetaMeanLogModel(Z, W)