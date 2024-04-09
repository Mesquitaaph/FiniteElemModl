include("./v3Module.jl")
import .v3Module

function main()
    alpha = 1; beta = 1; a = 0; b = 1; ne = 2^1
    f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

    C, X, EQoLG = v3Module.solve(alpha, beta, ne, a, b, f, u)

    C = X = EQoLG = nothing;
    GC.gc()
    # (GC.gc(true);GC.gc(true);GC.gc(true);GC.gc(true);GC.gc(true);GC.gc(true);GC.gc(true);GC.gc(true);)
end

@time main()