# module ZetaMeanLog
export ZetaMeanLogModel, Rootθ

using BenchmarkTools, LinearAlgebra, SpecialFunctions, Random, Plots, Printf

# 定义模型的相关方法
function Rootθ(t, tol = 1e-8)
    (s, eps) = (3.0, 1.0e-10) 
    f = -LogZetaFirst(s) - t
    while f <= 0.0
        s = 1 / 2 + s / 2
        f = -LogZetaFirst(s) - t
    end
    for iter in 1:100
        df = -LogZetaSecond(s)
        s = s - f / df
        f = -LogZetaFirst(s) - t
        if abs(f) < tol
            break
        end
    end
    return s
end


# 定义计算 LogZeta 的方法
function LogZetaSecond(θ)
    ce = (1.0 + 1.0im) / sqrt(2.0)
    de = 1.0e-5
    return imag(log(zeta(θ + ce * de)) + log(zeta(θ - ce * de))) / de^2
end

function LogZetaFirst(θ)
    de = 1.0e-10
    return imag(log(zeta(θ + de * im))) / de
end

# 定义 Fisher 信息的计算方法
function BMatrix(μ, θ, z_1, n_1, X)
    s_vec = (z_1 .- n_1 .* μ) .* μ ./ LogZetaSecond.(θ)  # Scoring vector
    Var1 = n_1 .* μ.^2 ./ LogZetaSecond.(θ)
    grad = X' * s_vec
    hess = X' * diagm(Var1) * X
    return grad, hess
end


# 定义优化函数
function ZetaMeanLogModel(Z::Matrix, W::Matrix; max_iter::Int = 1000, tol::Float64 = 1e-8)
    
    
    W = StandardizeColumns(W)
    
    nn1, nn2 = size(Z)          # numbers of rows and columns
    W = hcat(repeat([1],nn2), W) # add ones
    z = [i for i in 1:nn1, _ in 1:nn2]
    z_1 = vec(sum(Z.*log.(z), dims=1))
    n = vec(sum(Z, dims=1))
    β = zeros(size(W)[2])
    
    
    log_lik = Float64[]  # 初始化 log-likelihood 数组
    iters = 0
    
    μ = exp.(W * β)
    θ = Rootθ.(μ)
    log_likelihood = ZetaLogLik(θ, z_1, n)

    for iter in 1:max_iter
        iters += 1
        log_likelihood_old = log_likelihood
        β_old = β

        grad, hess = BMatrix(μ, θ, z_1, n, W)
        δ = pinv(hess) * grad

        # 线性搜索
        flag_true = true
        while flag_true
            β = β_old + δ
            μ = exp.(W * β)
            θ = Rootθ.(μ)
            log_likelihood = ZetaLogLik(θ, z_1, n)

            if log_likelihood >= log_likelihood_old
                flag_true = false
            else
                δ *= 0.5
            end
        end

        # 检查收敛
        if norm(β - β_old) < tol
            break
        end
        push!(log_lik, log_likelihood)
    end
    
    grad, hess = BMatrix(μ, θ, z_1, n, W)
    β_0 = zeros(length(β))
    Wald_stat, std = wald_statistic(β, β_0, hess)
    p_values = p_value(Wald_stat)
    lower_bound, upper_bound = confidence_interval(β, std)

    function format_number(x)
        if abs(x) < 1e-4
            return @sprintf("%8.2e", x)  # 使用科学计数法并确保负号对齐，宽度为10
        else
            return @sprintf("%8.4f", x)  # 保留4位小数并确保负号对齐，宽度为10
        end
    end
    
    df = DataFrame(
        Parameter = 1:length(β),
        Estimate = [format_number(x) for x in β],
        StandardError = [format_number(x) for x in std],
        WaldStatistic = [format_number(x) for x in Wald_stat],
        p_value = [format_number(x) for x in p_values],
        Lower95CI = [format_number(x) for x in lower_bound],
        Upper95CI = [format_number(x) for x in upper_bound]
    )

    println("Zeta Mean Log Model Summary:")
    println(df)
    
    # 计算BIC
    BIC = calculate_BIC(log_likelihood, β, Z)

    return β, θ, iters, log_lik, BIC, df
end


# end  # module ZetaMeanLogTest