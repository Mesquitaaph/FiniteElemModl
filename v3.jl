using GaussQuadrature, SparseArrays, StatsBase, BenchmarkTools

function montaLG(ne)
    LG1 = 1:1:ne

    LG = Matrix{Int64}(undef, ne, 2)

    LG[:,1] .= LG1
    LG[:,2] .= LG1.+1

    return LG'
end

function montaEQ(ne, neq)
    EQ = zeros(Int64, ne+1, 1) .+ ne
    EQ[2:ne] .= 1:1:neq

    return EQ
end

function PHI(P)
    return [(-1*P.+1)./2, (P.+1)./2]
end

function dPHI(P)
    return [-1/2*(P^0); 1/2*(P^0)]
end

function montaK!(ne, neq, dx, alpha, beta, EQoLG::Matrix{Int64})
    npg = 2; P, W = legendre(npg)
    
    phiP = reduce(vcat, PHI(P)'); dphiP = hcat(dPHI.(P)...)
    
    Ke = 2*alpha/dx .* (W.*dphiP) * dphiP' + beta*dx/2 .* (W.*phiP) * phiP'
    
    # I = EQoLG[:, [1, 2]]'
    # I = reshape(I, ne*2)
    
    # K = spzeros(neq+1, neq+1)

    # S = repeat(Ke, ne, ne)
    
    # K[I,I] .+= S[I,I]

    I = vec(EQoLG[[1,1,2,2], 1:1:ne])
    J = vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
    S = repeat(reshape(Ke, 4), outer=ne)
    
    K = sparse(I, J, S)[1:neq, 1:neq]

    return K
end

function montaxPTne(dx, X, P)
    return (dx ./ 2) .* (P .+ 1) .+ X
end

function montaF(ne, neq, X, f, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = reduce(vcat, PHI(P)')#; dphiP = hcat(dPHI.(P)...)
    
    dx = 1/ne

    xPTne = montaxPTne(dx, X[1:end-1]', P)

    Fe = (dx/2 .* W'.*phiP) * f(xPTne)
    
    Fe = vec(Fe'); Fe = Weights(Fe)

    I = vec(EQoLG')
    F = StatsBase.counts(I, Fe)

    return F[1:neq], xPTne
end

function erroVet(ne, xPTne, EQoLG, C, u)
    npg = 5; P, W = legendre(npg)

    phiP = reduce(vcat, PHI(P)')
    h = 1/ne

    d = hcat(C, 0)

    E = sqrt(h/2 * sum(W' * ((u.(xPTne)' - (phiP' * d[EQoLG])).^2)))

    return E
end

function solve(alpha, beta, ne, a, b, f, u)
    dx = 1/ne; neq = ne-1

    X = a:dx:b
    
    EQ = montaEQ(ne, neq)
    LG = montaLG(ne)
    EQoLG = EQ[LG]

    K = montaK!(ne, neq, dx, alpha, beta, EQoLG)
    
    F, xPTne = montaF(ne, neq, X, f, EQoLG)

    C = F'/K

    F = 1; K = 1;
    # GC.gc()

    return C#, X, xPTne, EQoLG
end

alpha = 1; beta = 1; a = 0; b = 1; ne = 2^15
f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))
@btime solve(alpha, beta, ne, a, b, f, u)

function convergence_test(errsize)
    alpha = 1; beta = 1; a = 0; b = 1;
    f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

    NE = 2 .^ [2:1:errsize;]
    H = 1 ./ NE
    
    E = zeros(length(NE))

    for i = 1:lastindex(NE)
        # println("Iniciando i = ", i)
        solve(alpha, beta, NE[i], a, b, f, u)
        # Ci, Xi, xPTnei, EQoLGi = solve(alpha, beta, NE[i], a, b, f, u)
        # E[i] = erroVet(NE[i], xPTnei, EQoLGi, Ci, u)
    end

    return nothing
end

# @btime convergence_test(23)
# dx = 1/ne; neq = ne - 1; EQ = montaEQ(ne, neq); LG = montaLG(ne); EQoLG = EQ[LG]; EQoLGT = EQoLG'

# function teste(alpha, beta, ne, a, b, f, u)
#     K = montaK!(ne, neq, dx, alpha, beta, EQoLG)
# end

# @btime teste(alpha, beta, ne, a, b, f, u)

# function teste2(alpha, beta, ne, a, b, f, u)
#     I = vec(EQoLG[[1,1,2,2], 1:1:ne])
#     J = vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
#     S = repeat(reshape([1 2; 3 4], 4), outer=ne)
    
#     return nothing
# end
# @btime teste2(alpha, beta, ne, a, b, f, u)

# function teste2!(I, J, S, alpha, beta, ne, a, b, f, u)
#     I .= vec(EQoLG[[1,1,2,2], 1:1:ne])
#     J .= vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
#     S .= repeat(reshape([1 2; 3 4], 4), outer=ne)
    
#     return nothing
# end

# I = zeros(Float64, 4*ne)
# J = zeros(Float64, 4*ne)
# S = zeros(Float64, 4*ne)
# @btime teste2!(I, J, S, alpha, beta, ne, a, b, f, u)


# @btime montaLG(ne)