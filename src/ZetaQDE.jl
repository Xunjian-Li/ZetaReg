# module ZetaQDE

export QDE  # 导出主函数 QDE

using LinearAlgebra, SpecialFunctions, Printf, DataFrames, PrettyTables

# 定义 QDE 函数
function QDE(Z, W, max_iter = 1000, tol = 1e-8)
    # 初始化存储结果的矩阵
    log_values = []  # 存储 log(x_{i+1} / x_i) 的值
    log_indices = []  # 存储编号的 log(i / j)
    log_indices1 = []  # 存储编号的 log(i / j)
    
    nn1, nn2 = size(Z)          # numbers of rows and columns
    z = [i for i in 1:nn1, _ in 1:nn2]
    z_1 = vec(sum(Z.*log.(z), dims=1))
    n = vec(sum(Z, dims=1))
    log_lik = Float64[]  # 初始化 log-likelihood 数组
    

    iters = 0
    
    
    W0 = hcat(ones(length(z_1)), W)
    NElement = []
    # 遍历每一列
    for col in 1:size(Z, 2)
        col_values = Z[:, col]  # 当前列
        id11 = W[col, :]
        n1 = 0
        for i in 1:(length(col_values) - 1)
            # 检查相邻两个元素是否都非 0
            if col_values[i] != 0 && col_values[i + 1] != 0
                # 计算 log(x_{i+1} / x_i)
                log_val = log(col_values[i + 1] / col_values[i])
                push!(log_values, log_val)

                # 记录编号关系 log(i / j)
                log_idx = log(i / (i + 1))
                push!(log_indices, log_idx)
                push!(log_indices1, id11)
                n1 = n1 + 1
            else
                break
            end
        end
        push!(NElement, n1)
    end

    # 转换数据类型
    Y = Float64.(log_values)
    result = hcat(log_indices1...)  # 将向量中的每个元素水平拼接成行
    result = transpose(result)
    W = hcat(log_indices, repeat(log_indices, 1, size(W)[2]) .* result)
    X_mat1 = hcat(ones(length(log_indices)), result)
    W = Float64.(W)  # 将矩阵元素转换为 Float64 类型

    # 计算 β 和 θ 的初始值
    β0 = pinv(W' * W) * W' * Y  ## 广义逆
    β = β0
    θ0 = X_mat1 * β0
    
    θ = W0 * β0
    
    if any(θ .< 1)
        log_likelihood = NaN
    else
        log_likelihood = ZetaLogLik(θ, z_1, n)
    end
    push!(log_lik, log_likelihood)
    

    # 初始化协方差矩阵 Sig
    Sig = zeros(length(θ0), length(θ0))
    w1 = sum(Z, dims=1)

    for _ in 1:max_iter
        
        iters = iters+1
        num1 = 0
        for col in 1:size(Z, 2)
            col_values = Z[:, col]  # 当前列
            rows_and_cols = NElement[col]  # 获取第 col 列的相邻元素个数
            Sig1 = zeros(rows_and_cols, rows_and_cols)  # 创建全零矩阵

            flag = true
            for i in 1:(length(col_values) - 1)
                if col_values[i] != 0 && col_values[i + 1] != 0
                    num1 = num1 + 1
                    if flag
                        p1 = i^(-θ0[num1]) / zeta(θ0[num1])
                        p2 = (i + 1)^(-θ0[num1]) / zeta(θ0[num1])
                        Sig[num1, num1] = (p1 + p2) / (p1 * p2) / w1[col]
                        flag = false
                    else
                        p1 = i^(-θ0[num1]) / zeta(θ0[num1])
                        p2 = (i + 1)^(-θ0[num1]) / zeta(θ0[num1])
                        Sig[num1, num1] = (p1 + p2) / (p1 * p2) / w1[col]
                        Sig[num1 - 1, num1] = -1 / p1 / w1[col]
                        Sig[num1, num1 - 1] = -1 / p1 / w1[col]
                    end
                else
                    break
                end
            end
        end
        
        if any(.!isreal.(Sig) .| isnan.(Sig))
            println("There exists at least a non-real or NaN value. Stopping execution.")
            θ = W0 * β
            return β, θ, iters, log_lik, NaN, NaN
        end
        
        # 更新 β 和 θ
        β = pinv(W' * inv(Sig) * W) * W' * inv(Sig) * Y
        θ0 = X_mat1 * β
        
        # 检查收敛
        if norm(β - β0) < tol
            break
        end
        
        θ = W0 * β0
        
        
        if any(θ .< 1)
            log_likelihood = NaN
        else
            log_likelihood = ZetaLogLik(θ, z_1, n)
        end
        push!(log_lik, log_likelihood)
        
        β0 = β
        
    end
    
    
    hess = W' * inv(Sig) * W
    β_0 = zeros(length(β))
    Wald_stat, std = wald_statistic(β, β_0, hess)
    p_values = p_value(Wald_stat)
    lower_bound, upper_bound = confidence_interval(β, std)

    
#     df = DataFrame(
#         Parameter = 1:length(β),
#         Estimate = [format_number(x) for x in β],
#         StandardError = [format_number(x) for x in std],
#         WaldStatistic = [format_number(x) for x in Wald_stat],
#         Lower95CI = [format_number(x) for x in lower_bound],
#         Upper95CI = [format_number(x) for x in upper_bound],
#         p_value = [format_number(x) for x in p_values]
#     )
    
    df = DataFrame(
        Parameter = 1:length(β),
        Estimate = [format_number(x) for x in β],
        StandardError = [format_number(x) for x in std],
        WaldStatistic = [format_number(x) for x in Wald_stat],
        CI_95 = ["($(format_number(l)), $(format_number(u)))" for (l, u) in zip(lower_bound, upper_bound)],
        p_value = [format_number(x) for x in p_values]
    )

    
    # 输出为 LaTeX 表格
    io = IOBuffer()
    pretty_table(io, df; backend = Val(:latex), tf = tf_latex_booktabs)
    latex_output = String(take!(io))
    
    println("QDE Model Summary:")
    println(latex_output)
    
    # 计算BIC
    BIC = calculate_BIC(log_likelihood, β, Z)

    return β0, θ, iters, log_lik, BIC, df
end

# end  # module ZetaQDE