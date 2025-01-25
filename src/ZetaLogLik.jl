export ZetaLogLik

# 对数似然函数
function ZetaLogLik(θ, z_1, n_1)
    return -sum(z_1 .* θ .+ n_1 .* log.(zeta.(θ)))
end