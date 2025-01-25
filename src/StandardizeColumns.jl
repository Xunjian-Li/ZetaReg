export StandardizeColumns

# 标准化列
function StandardizeColumns(A::AbstractMatrix)
    col_means = mean(A, dims=1)
    col_std = std(A, dims=1)
    return (A .- col_means) ./ col_std
end