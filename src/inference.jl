export wald_statistic, p_value, confidence_interval, calculate_BIC, format_number


using LinearAlgebra, SpecialFunctions, Distributions, Printf

function wald_statistic(β, β_0, hess)
    delta_β = β - β_0
    std = 1 ./ sqrt.(diag(hess))  # 标准误差是协方差矩阵的对角元素的平方根
    Wald_stat = (delta_β ./ std) .^ 2  # Wald统计量
    return Wald_stat, std
end

# 计算p值
function p_value(Wald_stat)
    p_values = 2 * (1 .- cdf(Normal(0, 1), sqrt.(abs.(Wald_stat))))
    return p_values
end

# 计算95%置信区间
function confidence_interval(β, std)
    z_value = quantile(Normal(0, 1), 0.975)
    lower_bound = β .- z_value .* std
    upper_bound = β .+ z_value .* std
    return lower_bound, upper_bound
end

# 计算BIC (贝叶斯信息准则)
function calculate_BIC(log_likelihood, β, Z)
    n = sum(Z)  # 样本数
    k = length(β)   # 参数数量
    bic = -2 * log_likelihood + k * log(n)
    return bic
end


function format_number(x)
    if abs(x) < 1e-4
        return @sprintf("%8.2e", x)  # 使用科学计数法并确保负号对齐，宽度为10
    else
        return @sprintf("%8.4f", x)  # 保留4位小数并确保负号对齐，宽度为10
    end
end