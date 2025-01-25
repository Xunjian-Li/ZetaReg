# module ZetaMean

export ZetaMeanModel, OptimizeZetaMean

using LinearAlgebra, SpecialFunctions, Random, BenchmarkTools

# 定义 ZetaMeanModel 结构体
struct ZetaMeanModel
    Z::Matrix  # 观测矩阵 Z
    W::Matrix  # 协变量矩阵 W
#     β::Vector{Float64}  # 模型参数
    max_iter::Int       # 最大迭代次数
    tol::Float64        # 收敛容差
end

# 默认构造函数
function ZetaMeanModel(Z::Matrix, W::Matrix; max_iter::Int = 1000, tol::Float64 = 1e-8)
    return ZetaMeanModel(Z, W, max_iter, tol)
end


# Rootθ 和 LogMean 函数
function Rootθ1(t, tol = 1e-8)
    if t > 20.
        return exp(-t) / zeta(2) + 2
    end

    (s, eps) = (3.0, 1.0e-10) 
    f = Real(LogMean(s)) - t
    while f <= 0.0
        s = 1 + s / 2
        f = Real(LogMean(s)) - t
    end

    for _ in 1:100
        df = imag(LogMean(s + eps * im)) / eps
        s = s - f / df
        f = Real(LogMean(s)) - t
        if abs(f) < tol
            break
        end
    end
    return s
end

function LogMean(s)
    if abs(s - 2) < 1.0e-5
        return -log(s - 2) - log(Complex(zeta(s)))
    else 
        return log(Complex(zeta(s - 1) / zeta(s)))
    end
end

# 计算 B 矩阵
function BMat(θ, z_1, n_1, X)
    zetaprime1_fun = θ -> imag(zeta(θ + 1.0e-10 * im)) / 1.0e-10
    zetaprime2_fun = θ -> imag(zetaprime1_fun(θ + 1.0e-10 * im)) / 1.0e-10

    zetaprime1_1 = zetaprime1_fun.(θ)
    zetaprime1_2 = zetaprime2_fun.(θ)
    zetaprime2_1 = zetaprime1_fun.(θ .- 1)
    zetaprime2_2 = zetaprime2_fun.(θ .- 1)

    zetaprime1 = zeta.(θ)
    zetaprime2 = zeta.(θ .- 1)

    f1 = zetaprime2 .- zetaprime1
    f2 = zetaprime2 .* zetaprime1_1 .- zetaprime2_1 .* zetaprime1
    f3 = zetaprime1_2 .* zetaprime1 .- zetaprime1_1.^2
    f4 = 2 .* zetaprime1 .* zetaprime1_1 .- zetaprime2_1 .* zetaprime1 .- zetaprime2 .* zetaprime1_1
    f5 = zetaprime2_2 .* zetaprime1 .- zetaprime2 .* zetaprime1_2

    s_vec = -(z_1 .* zetaprime1 .+ n_1 .* zetaprime1_1) .* f1 ./ f2
    a_vec = n_1 .* f1.^2 ./ f2.^2 .* ((f4 ./ f1 .- zetaprime1 .* f5 ./ f2) .* zetaprime1_1 .* (zetaprime1 .- 1) .+ f3)

    return -X' * s_vec, X' * diagm(a_vec) * X
end


# 模型优化函数
function OptimizeZetaMean(model::ZetaMeanModel)
    W = StandardizeColumns(model.W)
    nn1, nn2 = size(model.Z)
    W = hcat(ones(nn2), W)  # 添加偏置项
    z = [i for i in 1:nn1, _ in 1:nn2]
    z_1 = vec(sum(model.Z .* log.(z), dims=1))
    n = vec(sum(model.Z, dims=1))
    β = zeros(size(W)[2])
    log_lik = Float64[]

    μ = exp.(W * β) .+ 1
    θ = Rootθ1.(log.(μ))
    log_likelihood = ZetaLogLik(θ, z_1, n)
    iters = 0

    for iter in 1:model.max_iter
        
        iters = iters+1
        log_likelihood_old = log_likelihood
        β_old = β

        grad1, Hess1 = BMat(θ, z_1, n, W)
        δ = Hess1 \ grad1

        flag_true = true
        while flag_true
            β = β_old - δ
            μ = exp.(W * β) .+ 1
            θ = Rootθ1.(log.(μ))
            log_likelihood = ZetaLogLik(θ, z_1, n)
            flag_true = log_likelihood < log_likelihood_old
            δ *= 0.5
        end

        if norm(β - β_old) < model.tol
            break
        end

        push!(log_lik, log_likelihood)
    end

    return β, iters, log_lik
end

# end  # module ZetaMeanReg