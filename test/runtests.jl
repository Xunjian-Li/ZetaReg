using Test, ZetaReg, Random

@testset "ZetaReg Tests" begin
    # 生成随机测试数据
    Random.seed!(42)

    # 测试回归函数
    ze1 = RandZeta(3.1)
    
    println("ze1: ", ze1)
end