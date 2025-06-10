export GenerateMatrixAndVector


function GenerateMatrixAndVector(rows::Int, cols::Int)
    matrix = ones(Float64, rows, cols)
    for j in 2:cols
        for i in 1:rows
            matrix[i, j] = rand()
        end
    end
    vector = 0.05 .* rand(cols)
    return matrix, vector
end