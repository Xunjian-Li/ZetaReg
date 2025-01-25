module ZetaReg

# using LinearAlgebra, Random, SpecialFunctions

include("ZetaMean.jl")
include("ZetaMeanLog.jl")
include("RandZeta.jl")
include("ZetaQDE.jl")
include("ZetaLogLik.jl")
include("GenerateMatrixAndVector.jl")
include("StandardizeColumns.jl")


include("datasets/MyPackageData.jl")
include("datasets/census2000.jl")
include("datasets/census2010.jl")
include("datasets/census2020.jl")
# export ZetaMeanModel, OptimizeZetaMean



end # module ZetaReg
