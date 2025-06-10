# module MyPackageData

# 导出需要公开的常量和类型
export InsuranceData, CitiesData, Insurance, Cities

# 定义 Insurance 结构体
struct Insurance
    num_insurances::Matrix{Int}
    age::Matrix{Int}
end

# 定义 Cities 结构体
struct Cities
    freq_cities::Matrix{Int}
    years::Matrix{Int}
end

# 定义常量 InsuranceData 和 CitiesData
const InsuranceData = Insurance(
    [94 342 590 433 177 59;
     6 34 66 59 30 12;
     0 5 16 11 6 8;
     0 3 8 7 2 2;
     0 0 0 3 4 2;
     0 2 3 3 0 0;
     0 0 2 2 0 0;
     0 0 1 1 1 0;
     0 0 1 0 0 0;
     0 0 0 1 0 0;
     0 0 2 0 0 0;
     0 0 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 0],
    reshape([20, 30, 40, 50, 60, 70], 6, 1)
)

const CitiesData = Cities(
    [614 787 905 955;
     187 287 383 447;
     67 114 152 187;
     26 45 67 92;
     10 18 27 34;
     5 10 12 14;
     2 2 4 6;
     2 2 2 2;
     0 1 1 1],
    reshape([1970, 1980, 1991, 2000], 4, 1) 
)

# end