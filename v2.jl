using GaussQuadrature, SparseArrays, StatsBase

function montaLG(ne)
    LG1 = [1:1:ne;]
    LG = Matrix{Int64}(undef, 2, ne)
    LG[1,:] = LG1[:]
    LG[2,:] = (LG1.+1)[:]

    return LG
end

function montaEQ(ne, neq)
    EQ = zeros(Int64, ne+1, 1) .+ ne
    EQ[2:length(EQ)-1] = [1:1:neq;]

    return EQ
end

function PHI(P)
    return [(-1*P.+1)./2, (P.+1)./2]
end

function dPHI(P)
    return [-1/2*(P^0); 1/2*(P^0)]
end

function montaK(ne, neq, dx, alpha, beta, EQoLG)
    npg = 2; P, W = legendre(npg)
    
    phiP = reduce(vcat, PHI(P)'); dphiP = hcat(dPHI.(P)...)
    
    Ke = 2*alpha/dx * (W.*dphiP) * dphiP' + beta*dx/2 * (W.*phiP) * phiP'
    # println("computou Ke")
    I = vec(EQoLG[repeat([1,2], inner=2), [1:1:ne;]])
    J = vec(EQoLG[repeat([1,2], inner=1), repeat([1:1:ne;], inner=2)])
    # println("computou I,J")
    
    S = repeat(reshape(Ke, 4), outer=ne)
    
    K = sparse(I, J, S)

    return K[1:neq, 1:neq]
end

function montaxPTne(dx, X)
    return f(P) = (dx/2)*(P+1) .+ X
end

function montaF(ne, neq, X, f, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = reduce(vcat, PHI(P)')#; dphiP = hcat(dPHI.(P)...)
    
    dx = 1/ne
    xPTne = hcat(montaxPTne(dx, X[1:end-1]).(P)...)'

    Fe = (dx/2*W'.*phiP) * f(xPTne)
    
    Fe = vec(Fe'); Fe = Weights(Fe)

    I = vec(EQoLG')
    F = StatsBase.counts(I, Fe)

    return F[1:neq]
end

function solve()
    alpha = 1; beta = 1; ne = 2^23; dx = 1/ne; neq = ne-1
    a = 0; b = 1
    f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

    X = [a:dx:b;]

    EQ = montaEQ(ne, neq)
    LG = montaLG(ne)
    EQoLG = EQ[LG]

    K = montaK(ne, neq, dx, alpha, beta, EQoLG)
    # println("K")
    
    F = montaF(ne, neq, X, f, EQoLG)'
    # println("F")
    
    C = F/K
    # println("sistema")

    return C
end

@time begin
    solve()
end