using GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, 
Plots, StaticArrays, MKL, LinearSolve

BLAS.set_num_threads(24)

# KLUFactorization()

const cs = (
    cache1 = zeros(Float64, 2, 2),
    cache2 = zeros(Float64, 2, 2),
    cache3 = zeros(Float64, 2, 2)
)

function montaLG(ne)
    LG1 = 1:1:ne

    LG = zeros(Int64, ne, 2)

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
    return [-1/2*(P.^0), 1/2*(P.^ 0)]
end

function montaK(ne, neq, dx, alpha, beta, EQoLG::Matrix{Int64})
    npg = 2; P, W = legendre(npg)

    phiP = reduce(hcat, PHI(P)); dphiP = reduce(hcat, dPHI(P))

    cs.cache1 .= W.*dphiP'
    mul!(cs.cache2, cs.cache1, dphiP)

    cs.cache1 .= W.*phiP'    
    mul!(cs.cache2, cs.cache1, phiP, beta*dx/2, 2*alpha/dx)

    I = vec(EQoLG[[1,1,2,2], 1:1:ne])
    J = vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
    S = repeat(reshape(cs.cache2, 4), outer=ne)

    K = sparse(I, J, S)[1:neq, 1:neq]

    S = nothing; I = nothing; J = nothing

    return K
end

function montaxPTne(dx, X, P)
    return (dx ./ 2) .* (P .+ 1) .+ X
end

function montaF(ne, neq, X, f, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = reduce(vcat, PHI(P)')
    
    dx = 1/ne

    xPTne = montaxPTne(dx, X[1:end-1]', P)
    fxPTne = f.(xPTne)

    Fe = zeros(Float64, 2, ne)
    mul!(Fe, dx/2 .* W'.*phiP, fxPTne)

    F = zeros(neq+1)

    for (i, fe) in zip(EQoLG, Fe)
        F[i] += fe
    end

    Fe = nothing;
    return F[1:neq], xPTne
end

function erroVet(ne, EQoLG, C, u, u_x, xPTne)
    npg = 5; P, W = legendre(npg)
    
    phiP = reduce(hcat, PHI(P)); dphiP = reduce(hcat, dPHI(P))

    h = 1/ne

    cEQoLG = vcat(C, 0)[EQoLG]

    cache1 = zeros(Float64, npg, ne)
    cache2 = zeros(Float64, 1, ne)

    mul!(cache1, phiP, cEQoLG)
    
    cache1 .-= u.(xPTne)
    cache1 .^= 2
    mul!(cache2, W', cache1)
    EL2::Float64 = sqrt(h/2 * sum(cache2))


    mul!(cache1, dphiP, cEQoLG, 2/h, 1.0)
    cache1 .-= u_x.(xPTne)
    cache1 .^= 2
    mul!(cache2, W', cache1)
    EH01::Float64 = sqrt(h/2 * sum(cache2))

    return EL2, EH01
end

function solveSys(alpha, beta, ne, a, b, f, u)
    dx = 1/ne; neq = ne-1

    X = a:dx:b
    
    EQ = montaEQ(ne, neq); LG = montaLG(ne)
    EQoLG = EQ[LG]

    EQ = nothing; LG = nothing;

    K = montaK(ne, neq, dx, alpha, beta, EQoLG)

    F, xPTne = montaF(ne, neq, X, f, EQoLG)

    C = zeros(Float64, neq)

    C .= Symmetric(K)\F

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

# Montagem do sistema e sua resolução
@btime begin
    alpha = pi; beta = exp(1); a = 0; b = 1; ne = 2^23
    u(x) = sin(pi*x); u_x(x) = cos(pi*x)*pi; u_xx(x) = -sin(pi*x)*pi^2; f(x) = -alpha*u_xx(x) + beta*u(x);
    C, X, EQoLG = solveSys(alpha, beta, ne, a, b, f, u)

    C = nothing; X = nothing; EQoLG = nothing
end


function convergence_test!(NE, E, dE)
    alpha = pi; beta = exp(1.0); a = 0; b = 1;
    u(x) = sin(pi*x); u_x(x) = cos(pi*x)*pi; u_xx(x) = -sin(pi*x)*pi^2; f(x) = -alpha*u_xx(x) + beta*u(x);

    for i = 1:lastindex(NE)
        Ci, EQoLGi, xPTnei = solveSys(alpha, beta, NE[i], a, b, f, u)
        E[i], dE[i] = erroVet(NE[i], EQoLGi, Ci, u, u_x, xPTnei)
    end
end

errsize = 23
NE = 2 .^ [2:1:errsize;]
H = 1 ./NE
E = zeros(length(NE))
dE = similar(E)

# @btime convergence_test!(NE, E, dE)

# plot(H, E, xaxis=:log2, yaxis=:log2)
# plot!(H, H .^2, xaxis=:log2, yaxis=:log2)

GC.gc()