# module ZetaMean

export ZetaMeanModel

using LinearAlgebra, SpecialFunctions, Random, BenchmarkTools, Printf, DataFrames, PrettyTables, TimerOutputs


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
function BMat(θ, z_1, n, X)
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

    s_vec = -(z_1 .* zetaprime1 .+ n .* zetaprime1_1) .* f1 ./ f2
    a_vec = n .* f1.^2 ./ f2.^2 .* ((f4 ./ f1 .- zetaprime1 .* f5 ./ f2) .* zetaprime1_1 .* (zetaprime1 .- 1) .+ f3)

    return -X' * s_vec, X' * diagm(a_vec) * X
end

function ZetaMeanModel(
        data::DataFrame; 
        matrix::Bool = false, 
        intercept::Bool = true, 
        verbose::Bool = false, 
        max_iter::Int = 1000, 
        tol::Union{Float64, Float32, BigFloat} = 1e-6
    )
    
    numeric_cols = names(data, eltype.(eachcol(data)) .<: AbstractFloat)
    mat = Matrix(select(data, numeric_cols))
    T = eltype(mat)
    
    W = Matrix(data[:,2:end])
    Z = data[:,1]
    W = StandardizeColumns(W)
    
    to = TimerOutput()  # recording time
    
    n0 = length(Z)
    W = hcat(ones(n0), W)  
    n = 1
    z1 = log.(Z)
    
    β = zeros(size(W)[2])
    log_lik = Float64[]
    
    Wβ = W * β
    μ = exp.(Wβ) .+ 1
    log_μ = log.(μ)
    θ = Rootθ1.(log.(μ))
    log_likelihood = ZetaLogLik(θ, z1, n)
    iters = 0

    for iter in 1:max_iter
        
        iters = iters+1
        log_likelihood_old = log_likelihood
        β_old = β

        @timeit to "computing gradient and hessian" grad, Hess = BMat(θ, z1, n, W)
        
        if any(.!isreal.(grad) .| isnan.(grad) .| .!isreal.(Hess) .| isnan.(Hess))
            println("There exists at least a non-real or NaN value. Stopping execution.")
            break
            return β, θ, iters, log_lik, NaN, NaN
        end
        
        @timeit to "computing update direction" δ = Hess \ grad

        flag_true = true
        while flag_true
            β = β_old - δ
            @timeit to "computing Wβ"   mul!(Wβ, W, β)
            @timeit to "computing μ"     @. μ = exp(Wβ) + 1
            @timeit to "computing log(μ)" @. log_μ = log(μ)
            @timeit to "computing θ"     @. θ = Rootθ1(log_μ)
            @timeit to "computing log-likelihood" log_likelihood = ZetaLogLik(θ, z1, n)
            flag_true = log_likelihood < log_likelihood_old  || minimum(θ) < 2.0 + 1e-6
            δ *= 0.5
        end
        
        if verbose
            println("The $iter steps with infinity norm: ", norm(grad, Inf),
                " Log-likelihood: ", log_likelihood)
        end
        
        if abs(log_likelihood - log_likelihood_old)/(abs(log_likelihood)+1) < tol
            iters = iter
            println("Converged at the $iter steps with infinity norm: ", norm(grad, Inf))
            break
        end

    end
    
    grad, hess = BMat(θ, z1, n, W)
    β_0 = zeros(length(β))
    Wald_stat, std = wald_statistic(β, β_0, - hess)
    p_values = p_value(Wald_stat)
    lower_bound, upper_bound = confidence_interval(β, std)
    
    function format_number(x)
        if abs(x) < 1e-4
            return @sprintf("%8.2e", x)  # 使用科学计数法并确保负号对齐，宽度为10
        else
            return @sprintf("%8.4f", x)  # 保留4位小数并确保负号对齐，宽度为10
        end
    end
    
    new_names = ["Intercept", names(data)[2:end]...]
    
    df = DataFrame(
        Parameter = new_names,
        Estimate = [format_number(x) for x in β],
        StandardError = [format_number(x) for x in std],
        WaldStatistic = [format_number(x) for x in Wald_stat],
        CI_95 = ["($(format_number(l)), $(format_number(u)))" for (l, u) in zip(lower_bound, upper_bound)],
        p_value = [format_number(x) for x in p_values]
    )

    io = IOBuffer()
    pretty_table(io, df; backend = Val(:latex), tf = tf_latex_booktabs)
    latex_output = String(take!(io))
    BIC = calculate_BIC(log_likelihood, β, Z)
    
    if verbose
        println("Zeta Mean Model Summary:")
        println(latex_output)
        println(df)
        show(to)
    end
    
    return β, θ, iters, log_likelihood, BIC, df
end

# end  # module ZetaMeanReg