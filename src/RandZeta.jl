# module RandZeta

export RandZeta

using SpecialFunctions, Random

# 生成 Zipf 分布随机数
function InvB(u1, s) 
    if u1 < 1.0
        return u1
    else
        return (s - (s - 1) * u1)^(1 / (1 - s))
    end
end

function RandZeta(s)
    while true
        u1 = s / (s - 1) * rand()
        u = rand()
        x = InvB(u1, s)
        n = ceil(x)
        if u < (s - 1) / (n^s * ((n - 1)^(1 - s) - n^(1 - s))) || n == 1
            return n
        end
    end
end

# end  # module ZetaMeanReg