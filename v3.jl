using GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, 
Plots, StaticArrays, MKL, BandedMatrices, LinearSolve

# BLAS.get_config()
# versioninfo()
# BLAS.get_num_threads()
BLAS.set_num_threads(24)

KLUFactorization()

const cs = (
    cache1 = zeros(Float64, 2, 2),
    cache2 = zeros(Float64, 2, 2),
    cache3 = zeros(Float64, 2, 2)
)

function Example(alpha, beta, a, b, u, u_x, f)
    return (alpha = alpha, beta = beta, a = a, b = b, u = u, u_x = u_x, f = f)
end

function example1()
    alpha = 1.0; beta = 1.0; a = 0.0; b = 1.0;
    u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1));
    u_x(x) = 1 - (exp(-x) + exp(x)) *(1/(exp(1.0) - exp(-1)));
    f(x) = x

    return Example(alpha, beta, a, b, u, u_x, f)
end

function example2()
    alpha = pi; beta = exp(1.0); a = 0.0; b = 1.0;
    u(x) = sin(pi*x);
    u_x(x) = cos(pi*x)*pi; u_xx(x) = -sin(pi*x)*pi^2;
    f(x) = -alpha*u_xx(x) + beta*u(x)

    return Example(alpha, beta, a, b, u, u_x, f)
end

function examples(case)
    if case == 1
        return example1()
    elseif case == 2
        return example2()
    end
end

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
    
    phiP = zeros(Float64, 2, 2); dphiP = zeros(Float64, 2, 2);

    phiP = reduce(hcat, PHI(P)); dphiP = reduce(hcat, dPHI(P))

    cs.cache1 .= W.*dphiP'
    mul!(cs.cache2, cs.cache1, dphiP)
    cs.cache1 .= W.*phiP'
    
    mul!(cs.cache2, cs.cache1, phiP, beta*dx/2, 2*alpha/dx)

    I = vec(@view EQoLG[[1,1,2,2], 1:1:ne])
    J = vec(@view EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
    S = repeat(vec(cs.cache2), outer=ne)

    K = BandedMatrix(Zeros(neq, neq), (1,1))

    for (i,j,s) in zip(I,J,S)
        if i <= neq && j <= neq 
            @inbounds K[i,j] += s
        end
    end

    S = nothing; I = nothing; J = nothing

    return K
end

function montaxPTne(dx, X, P)
    return (dx ./ 2) .* (P .+ 1) .+ X
end

function montaF(ne, neq, X, f::Function, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = reduce(vcat, PHI(P)')#; dphiP = hcat(dPHI.(P)...)
    
    dx = 1/ne

    xPTne = montaxPTne(dx, X[1:end-1]', P)
    # fxPTne = f.(xPTne)
    fxPTne = similar(xPTne)

    Threads.@threads for i in eachindex(xPTne)
        fxPTne[i] = f(xPTne[i])
    end

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

    # println("Resolvendo sistema")
    C .= Symmetric(K)\F
    # println("Resolvendo sistema: fim")

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

println("Rodando")
# @btime begin
#     alpha, beta, a, b, u, u_x, f = examples(1); ne = 2^23
#     C, X, EQoLG = solveSys(alpha, beta, ne, a, b, f, u)

#     C = nothing; X = nothing; EQoLG = nothing
# end

function convergence_test!(NE, E, dE, example)
    alpha, beta, a, b, u, u_x, f = examples(example)

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

# @profview convergence_test!(NE, E, dE, 2)
@btime convergence_test!(NE, E, dE, 2)

plot(H, E, xaxis=:log2, yaxis=:log2); plot!(H, H .^2, xaxis=:log2, yaxis=:log2)

GC.gc()

############ TESTES ############

# dx = 1/ne; neq = ne - 1; EQ = montaEQ(ne, neq); LG = montaLG(ne); EQoLG = EQ[LG]; EQoLGT = EQoLG'

function brodcast_seq()
    ne = 2^23
    alpha, beta, a, b, u, u_x, f = examples(2)

    npg = 5; P, W = legendre(npg)
    dx = 1/ne
    X = a:dx:b

    xPTne = montaxPTne(dx, X[1:end-1]', P)
    fxPTne = f.(xPTne)

    nothing
end

function brodcast_prll1()
    ne = 2^23
    alpha, beta, a, b, u, u_x, f = examples(2)

    npg = 5; P, W = legendre(npg)
    dx = 1/ne
    X = a:dx:b

    xPTne = montaxPTne(dx, X[1:end-1]', P)

    Threads.@threads for i in eachindex(xPTne)
        @inbounds xPTne[i] = f(xPTne[i])
    end

    nothing
end

function brodcast_prll2()
    ne = 2^23
    alpha, beta, a, b, u, u_x, f = examples(2)

    npg = 5; P, W = legendre(npg)
    dx = 1/ne
    X = a:dx:b

    xPTne = montaxPTne(dx, X[1:end-1], P')

    # fxPTne = similar(xPTne)

    # Threads.@threads for i in 1:(size(xPTne)[2])
    #     fxPTne[:,i] .= f.(xPTne[:,i])
    # end

    nothing
end

# @btime brodcast_seq()

# @btime brodcast_prll1()

# @btime brodcast_prll2()

# EQoLG
# BandedMatrix(Zeros(neq+1, neq+1), (1,1))

# Threads.@threads for ij in 1:neq^2
#     i = fld(ij-1, neq) + 1
#     j = mod(ij-1, neq) + 1
#     K[i,j] = Ks[i,j]
# end

# function teste1(alpha, beta, ne, a, b, f, u, neq)
#     LG = Matrix{Int64}(undef, ne, 2)

#     LG[:,1] .= 1:1:ne
#     LG[:,2] .= 2:1:(ne+1)

#     return LG'
# end

# @btime teste1(alpha, beta, ne, a, b, f, u, neq)

# K = montaK(ne, neq, dx, alpha, beta, EQoLG)
# b = rand(neq)