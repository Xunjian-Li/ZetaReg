
# module ZetaMeanLog
export ZetaMeanLogModel

using LinearAlgebra, SpecialFunctions, Random, Plots, Printf, PrettyTables
using LoopVectorization, TimerOutputs
import Base: eltype


"""
    struct zetastruct{T, A}

A structured container for Zeta-based model data and intermediate computations.

# Fields
- `y`         : Response vector or matrix
- `W`         : Design matrix
- `ys`        : Column-wise sum of y (for matrix input)
- `z1`        : Log-transformed y or column-sum log(z)
- `β`         : Coefficients (parameter vector)
- `M_storagep`: Intermediate storage (n × p)
- `hess`      : Hessian matrix (p × p)
- `μ`, `θ`    : Fitted mean and natural parameter vectors
- `s_vec`     : Score vector
- `var`       : Variance vector
- `grad`      : Gradient vector (p × 1)
"""
mutable struct zetastruct{T <: AbstractFloat, A <: AbstractArray{T}}
    y :: A
    W :: AbstractMatrix{T}
    ys
    z1 :: A
    β :: A
    M_storagep :: A
    M_storagen :: A
    hess :: AbstractMatrix{T}
    μ :: AbstractVector{T}
    θ :: AbstractVector{T}
    s_vec :: A
    var :: A
    grad :: A
end

"""
    zetastruct(W::Matrix{T}, y::Union{Vector{T}, Matrix{T}}; intercept::Bool=true, matrix::Bool=false)

Construct a `zetastruct` object for scalar or matrix response y.
"""
function zetastruct(
    y::Union{AbstractVector{T}, AbstractMatrix{T}},
    W::AbstractMatrix{T};
    intercept::Bool = true,
    matrix::Bool = false
) where T <: AbstractFloat

    n, p = size(W)
    W = standardize_columns(W)

    if matrix
        @assert isa(y, AbstractMatrix) "y should be a matrix when `matrix=true`"
        nn1, nn2 = size(y)
        z1 = zeros(T, nn2)
        ys = vec(sum(y, dims=1))
        compute_z1!(z1, y)  # User-defined function expected

        if intercept
            W = hcat(ones(T, nn1), W)
        end
        p = size(W, 2)
    else
        @assert isa(y, AbstractVector) "y should be a vector when `matrix=false`"

        if intercept
            W = hcat(ones(T, length(y)), W)
        end

        z1 = log.(y)
        ys = one(T)
        p = size(W, 2)
    end

    β = zeros(T, p)
    M_storagep = zeros(T, p)
    M_storagen = zeros(T, n)
    hess = zeros(T, p, p)
    μ = ones(T, n)
    θ = ones(T, n)
    s_vec = zeros(T, n)
    var = zeros(T, n)
    grad = zeros(T, p)

    return zetastruct(y, W, ys, z1, β, M_storagep, M_storagen, hess, μ, θ, s_vec, var, grad)
end

"""
    eltype(::zetastruct{T, A})

Return the element type T of the zetastruct.
"""
eltype(::zetastruct{T, A}) where {T, A} = T


"""
    standardize_columns(X::Matrix{T})

Return a matrix with columns standardized to zero mean and unit variance.
"""
function standardize_columns(X::AbstractArray{T}) where T <: AbstractFloat
    X_standardized = copy(X)
    for j in 1:size(X, 2)
        col = X[:, j]
        μ = mean(col)
        σ = std(col)
        X_standardized[:, j] .= (col .- μ) ./ σ
    end
    return X_standardized
end

γ1 = -0.07282
γ2 = -0.00969

function zeta_prime(s)
    return -1/(s-1)^2 - γ1
end

function zeta_primeprime(s)
    return 2/(s-1)^3 + γ2
end

function logzeta_prime(s)
    return zeta_prime(s)/zeta(s)
end

function logzeta_primeprime(s)
    a = logzeta_prime(s)
    return zeta_primeprime(s)/zeta(s) - a^2
end

"""
    First(θ::T)
Compute the first derivative (imaginary part) of log(zeta(θ)) using central difference.
"""
function First(θ::T) where T <: AbstractFloat
    dm = T(1.001)
    de1 = T(1.0e-10)
    if θ > dm
        return imag(log(zeta(θ + de1 * im))) / de1
    else
        return logzeta_prime(θ)
    end
end

"""
    Second(θ::T)

Compute the second derivative (imaginary) of log(zeta(θ)) using complex step method.
"""
function Second(θ::T) where T <: AbstractFloat
    dm = T(1.001)
    ce = (one(T) + one(T)*im) / sqrt(T(2.0))
    de2 = T(1.0e-3)
    if θ > dm
        return imag(log(zeta(θ + ce * de2)) + log(zeta(θ - ce * de2))) / de2^2
    else
        return logzeta_primeprime(θ)
    end
end

function Rootθ(t::T, tol ::T = T(1e-8)) where T <: AbstractFloat
    (s, eps) = (T(3.0), T(1.0e-10)) 
    f = -First(s) - t
    while f <= 0.0
        s = 1 / T(2) + s / T(2)
        f = -First(s) - t
    end
    for iter in 1:100
        df = -Second(s)
        s = s - f / df
        f = -First(s) - t
        if abs(f) < tol
            break
        end
    end
    return s
end

"""
    BMatrix(zetas)

Compute the gradient and Hessian for the optimization step.
"""
function BMatrix(
        zetas::zetastruct{T, A}
        )  where {T <: AbstractFloat, A <: AbstractArray{T}}
    
    n, p = size(zetas.W)
    logzeta2 = Second.(zetas.θ)
    @. zetas.s_vec .= (zetas.z1 - zetas.ys * zetas.μ) * zetas.μ / logzeta2
    @. zetas.var  .= zetas.ys * zetas.μ^2 / logzeta2
    mul!(zetas.grad, zetas.W', zetas.s_vec)              # grad = X' * s_vec
    tmp = zetas.var .* zetas.W
    mul!(zetas.hess, transpose(zetas.W), tmp)

end

"""
    compute_z1!(z1, ys)

Compute vector z1 with elementwise sum: `z1[j] = ∑ ys[i, j] * log(i)`
"""
function compute_z1!(z1, zetas_y)
    nn1, nn2 = size(zetas_y)
    @inbounds @turbo for j in 1:nn2
        acc = 0.0
        for i in 1:nn1
            acc += zetas_y[i, j] * log(i)
        end
        z1[j] = acc
    end
    return z1
end

function ZetaMeanLogModel(
        data::DataFrame;
        matrix::Bool = false, 
        intercept::Bool = true, 
        verbose::Bool = false, 
        max_iter::Int = 100, 
        tol::Union{Float64, Float32, BigFloat} = 1e-6
    )
    
    numeric_cols = names(data, eltype.(eachcol(data)) .<: AbstractFloat)
    mat = Matrix(select(data, numeric_cols))
    T = eltype(mat)
    
    W = Matrix(data[:,2:end])
    y = data[:,1]
    
    to = TimerOutput()  # recording time
    n, p = size(W)
    zetas = zetastruct(y, W; intercept = intercept, matrix = matrix)
    
    β_old = similar(zetas.β)
    log_lik = T[]
    iters = T(0)
    
    θ0 = Rootθ(T(1))
    @. zetas.θ .= zetas.θ .* θ0
    log_likelihood = -Inf
    log_likelihood_old = -Inf

    iters = max_iter
    for iter in 1:max_iter
        
        log_likelihood_old = log_likelihood
        copyto!(β_old, zetas.β)
        
        # update gradient and hessian matrix
        @timeit to "computing gradient and hessian" BMatrix(zetas)
        @timeit to "computing update direction" zetas.M_storagep .= zetas.hess \ zetas.grad # update direction
        
        # linear search
        flag_true = true
        while flag_true
            zetas.β .= β_old + zetas.M_storagep
            @timeit to "computing Wβ" mul!(zetas.M_storagen, zetas.W, zetas.β)
            @timeit to "computing μ" @. zetas.μ .= exp(zetas.M_storagen)
            @timeit to "computing θ" @. zetas.θ .= Rootθ(zetas.μ)
            @timeit to "computing log-likelihood" log_likelihood = ZetaLogLik(zetas.θ, zetas.z1, zetas.ys)
            flag_true = false
            if log_likelihood >= log_likelihood_old
                flag_true = false
            else
                zetas.M_storagep .= 0.5*zetas.M_storagep
            end
        end
        
        if verbose
            println("The $iter steps with infinity norm: ", norm(zetas.grad, Inf),
                " Log-likelihood: ", log_likelihood)
        end
        
        # checking convergence
#         if abs(log_likelihood - log_likelihood_old)/(abs(log_likelihood)+1) < tol
        if norm(zetas.grad, Inf) < tol*n
            iters = iter
            println("Converged at the $iter steps with infinity norm: ", norm(zetas.grad, Inf))
            break
        end
    end
    
    BMatrix(zetas)
    β_0 = zeros(length(zetas.β))
    Wald_stat, std = wald_statistic(zetas.β, β_0, zetas.hess)
    p_values = p_value(Wald_stat)
    lower_bound, upper_bound = confidence_interval(zetas.β, std)

    function format_number(x)
        if abs(x) < 1e-4
            return @sprintf("%8.2e", x)  
        else
            return @sprintf("%8.4f", x)  
        end
    end
    
    new_names = ["Intercept", names(data)[2:end]...]
    
    df = DataFrame(
        Parameter = new_names,
        Estimate = [format_number(x) for x in zetas.β],
        StandardError = [format_number(x) for x in std],
        WaldStatistic = [format_number(x) for x in Wald_stat],
        CI_95 = ["($(format_number(l)), $(format_number(u)))" for (l, u) in zip(lower_bound, upper_bound)],
        p_value = [format_number(x) for x in p_values]
    )

    io = IOBuffer()
    pretty_table(io, df; backend = Val(:latex), tf = tf_latex_booktabs)
    latex_output = String(take!(io))
    BIC = calculate_BIC(log_likelihood, zetas.β, y)
    
    if verbose
        println("Zeta Mean Log Model Summary:")
        println(latex_output)
        println(df)
        show(to)
    end
    
    return zetas.β, zetas.θ, iters, log_likelihood, BIC, df
end

