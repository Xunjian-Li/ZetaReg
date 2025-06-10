module ZetaReg

# using LinearAlgebra, Random, SpecialFunctions

include("ZetaMean.jl")
include("ZetaMeanLog.jl")
include("RandZeta.jl")
include("ZetaQDE.jl")
include("ZetaLogLik.jl")
include("GenerateMatrixAndVector.jl")
include("StandardizeColumns.jl")
include("inference.jl")
# include("zetaRegression_sophisticated.jl")

include("datasets/MyPackageData.jl")



# include("datasets/census2000.csv")
# include("datasets/census2010.csv")
# include("datasets/census2020.csv")



end # module ZetaReg
