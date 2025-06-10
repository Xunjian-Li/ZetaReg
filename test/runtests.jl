using Test, ZetaReg, Random, Printf, DataFrames

@testset "ZetaReg Tests for InsuranceData" begin
    # 生成随机测试数据
    Random.seed!(42)

    Z = InsuranceData.num_insurances
    W = InsuranceData.age
    
    # 打印 InsuranceData 数据
    println("Initial Data from InsuranceData:")
    println("==============================================")

    # 打印 num_insurances 矩阵
    println("num_insurances (Z):")
    println("====================")
    for row in eachrow(InsuranceData.num_insurances)
        println(join(row, ", "))
    end

    println("\nage (W):")
    println("====================")
    println(join(InsuranceData.age, ", "))

    println("==============================================\n")
    
    # 假设 Z 和 W 是已定义的矩阵
    (rho1, θ1, iters1, loglikelihoods1, BIC1, _) = QDE(Z, W)
    (rho2, θ2, iters2, loglikelihoods2, BIC2, _) = ZetaMeanModel(Z, W, matrix = true)
    (rho3, θ3, iters3, loglikelihoods3, BIC3, _) = ZetaMeanLogModel(Z, W, matrix = true)

    
    # 打印结果
    println("Comparison of Results:")
    println("====================================================================")
    println("Method           | β (first)   | θ (min)   | iters | log-likelihood (last)")
    println("====================================================================")
    @printf("%-17s | %-10.4f | %-10.4f | %-5d | %-10.4f\n", "QDE", rho1[1], minimum(θ1), iters1, loglikelihoods1[end])
    @printf("%-17s | %-10.4f | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanModel", rho2[1], minimum(θ2), iters2, loglikelihoods2[end])
    @printf("%-17s | %-10.4f | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanLogModel", rho3[1], minimum(θ3), iters3, loglikelihoods3[end])
    println("====================================================================")

    # 打印表头
    println("Detailed Vectors (β and θ):")
    println("===============================================================================================================")
    println("Index | QDE β      | ZetaMean β      | ZetaMeanLog β   || QDE θ      | ZetaMean θ      | ZetaMeanLog θ   ")
    println("===============================================================================================================")

    # 获取最大长度，确保遍历所有向量
    max_len = maximum([length(rho1), length(rho2), length(rho3), length(θ1), length(θ2), length(θ3)])

    # 按行打印
    for i in 1:max_len
        β_qde = i <= length(rho1) ? @sprintf("%-10.4f", rho1[i]) : " " * " "
        β_zeta_mean = i <= length(rho2) ? @sprintf("%-15.4f", rho2[i]) : " " * " "
        β_zeta_log = i <= length(rho3) ? @sprintf("%-15.4f", rho3[i]) : " " * " "
        θ_qde = i <= length(θ1) ? @sprintf("%-10.4f", θ1[i]) : " " * " "
        θ_zeta_mean = i <= length(θ2) ? @sprintf("%-15.4f", θ2[i]) : " " * " "
        θ_zeta_log = i <= length(θ3) ? @sprintf("%-15.4f", θ3[i]) : " " * " "

        println(@sprintf("%-5d | %-10s | %-15s | %-15s || %-10s | %-15s | %-15s", 
                         i, β_qde, β_zeta_mean, β_zeta_log, θ_qde, θ_zeta_mean, θ_zeta_log))
    end

    println("===============================================================================================================")


end

# 插入一个空行
println()
println()
println()

@testset "ZetaReg Tests for CitiesData" begin
    # 生成随机测试数据
    Random.seed!(42)

    Z = CitiesData.freq_cities
    W = CitiesData.years
    
    # 打印 InsuranceData 数据
    println("Initial Data from CitiesData:")
    println("==============================================")

    # 打印 num_insurances 矩阵
    println("freq_cities (Z):")
    println("====================")
    for row in eachrow(CitiesData.freq_cities)
        println(join(row, ", "))
    end

    println("years (W):")
    println("====================")
    println(join(CitiesData.years, ", "))

    println("==============================================\n")
    
    # 假设 Z 和 W 是已定义的矩阵
    (rho1, θ1, iters1, loglikelihoods1, BIC1, _) = QDE(Z, W)
    (rho2, θ2, iters2, loglikelihoods2, BIC2, _) = ZetaMeanModel(Z, W, matrix = true)
    (rho3, θ3, iters3, loglikelihoods3, BIC3, _) = ZetaMeanLogModel(Z, W, matrix = true)

    
    # 打印结果
    println("Comparison of Results:")
    println("====================================================================")
    println("Method           | β (first)   | θ (min)   | iters | log-likelihood (last)")
    println("====================================================================")
    @printf("%-17s | %-10.4f | %-10.4f | %-5d | %-10.4f\n", "QDE", rho1[1], minimum(θ1), iters1, loglikelihoods1[end])
    @printf("%-17s | %-10.4f | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanModel", rho2[1], minimum(θ2), iters2, loglikelihoods2[end])
    @printf("%-17s | %-10.4f | %-10.4f | %-5d | %-10.4f\n", "ZetaMeanLogModel", rho3[1], minimum(θ3), iters3, loglikelihoods3[end])
    println("====================================================================")

    # 打印表头
    println("Detailed Vectors (β and θ):")
    println("===============================================================================================================")
    println("Index | QDE β      | ZetaMean β      | ZetaMeanLog β   || QDE θ      | ZetaMean θ      | ZetaMeanLog θ   ")
    println("===============================================================================================================")

    # 获取最大长度，确保遍历所有向量
    max_len = maximum([length(rho1), length(rho2), length(rho3), length(θ1), length(θ2), length(θ3)])

    # 按行打印
    for i in 1:max_len
        β_qde = i <= length(rho1) ? @sprintf("%-10.4f", rho1[i]) : " " * " "
        β_zeta_mean = i <= length(rho2) ? @sprintf("%-15.4f", rho2[i]) : " " * " "
        β_zeta_log = i <= length(rho3) ? @sprintf("%-15.4f", rho3[i]) : " " * " "
        θ_qde = i <= length(θ1) ? @sprintf("%-10.4f", θ1[i]) : " " * " "
        θ_zeta_mean = i <= length(θ2) ? @sprintf("%-15.4f", θ2[i]) : " " * " "
        θ_zeta_log = i <= length(θ3) ? @sprintf("%-15.4f", θ3[i]) : " " * " "

        println(@sprintf("%-5d | %-10s | %-15s | %-15s || %-10s | %-15s | %-15s", 
                         i, β_qde, β_zeta_mean, β_zeta_log, θ_qde, θ_zeta_mean, θ_zeta_log))
    end

    println("===============================================================================================================")


end