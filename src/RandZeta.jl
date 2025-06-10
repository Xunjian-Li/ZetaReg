# module RandZeta

export RandZeta

using SpecialFunctions, Random

# sampling from Zipf 


function InvB(u, s) 
    if u < 1.0
        return u
    else
        return (s - (s - 1) * u)^(1 / (1 - s))
    end
end

function RandZeta(s)
    while true
        u1 = s / (s - 1) * rand()
        u2 = rand()
        x = InvB(u1, s)
        y = ceil(x)
        if u2 < (s - 1) / (y^s * ((y - 1)^(1 - s) - y^(1 - s))) || y == 1
            return y
        end
    end
end

# end  # module ZetaMeanReg