# using MUMPS, MPI
using GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, 
Plots, StaticArrays, MKL, BandedMatrices, LinearSolve

# BLAS.get_config()
# versioninfo()
# BLAS.get_num_threads()
BLAS.set_num_threads(24)

# KrylovJL_GMRES()
KLUFactorization()
# KrylovKitJL_GMRES()
# IterativeSolversJL_GMRES()
# UMFPACKFactorization()
# MKLPardisoFactorize()
# MKLPardisoIterate()

# ps = MKLPardisoSolver()
# set_nprocs!(ps, 6)
# get_nprocs(ps)
const cs = (
    cache1 = zeros(Float64, 2, 2),
    cache2 = zeros(Float64, 2, 2),
    cache3 = zeros(Float64, 2, 2)
)

macro show_locals()
    quote 
        locals = Base.@locals
        println("\nIndividual sizes (does not account for overlap):")
        for (name, refval) in locals
            println("\t$name: $(Base.format_bytes(Base.summarysize(refval)))")
        end
        print("Joint size: ")
        println("$(Base.format_bytes(Base.summarysize(values(locals))))\n")
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

function montaK!(ne, neq, dx, alpha, beta, EQoLG::Matrix{Int64})
    npg = 2; P, W = legendre(npg)
    
    phiP = zeros(Float64, 2, 2); dphiP = zeros(Float64, 2, 2);

    phiP = reduce(hcat, PHI(P)); dphiP = reduce(hcat, dPHI(P))

    # println(size(phiP), size(dphiP))
    # Ke = zeros(Float64, 2, 2)
    # cache = similar(Ke)

    cs.cache1 .= W.*dphiP'
    mul!(cs.cache2, cs.cache1, dphiP)
    cs.cache1 .= W.*phiP'
    
    mul!(cs.cache2, cs.cache1, phiP, beta*dx/2, 2*alpha/dx)

    # Ke = 2*alpha/dx .* (W.*dphiP') * dphiP + beta*dx/2 .* (W.*phiP') * phiP
    # Ke = 2*alpha/dx .* cs.cache * dphiP + beta*dx/2 .* (W.*phiP') * phiP
    # Ke = 2*alpha/dx .* Ke + beta*dx/2 .* cs.cache * phiP
    # Ke = beta*dx/2 .* cs.cache * phiP + 2*alpha/dx .* Ke


    I = vec(EQoLG[[1,1,2,2], 1:1:ne])
    J = vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
    # S = repeat(reshape(Ke, 4), outer=ne)
    S = repeat(reshape(cs.cache2, 4), outer=ne)

    K = BandedMatrix(Zeros(neq, neq), (1,1))

    for (i,j,s) in zip(I,J,S)
        if i <= neq && j <= neq 
            K[i,j] += s
        end
    end

    # Ks = sparse(I, J, S)[1:neq, 1:neq]
    # Threads.@threads for coo in findall(!iszero, Ks)
    #     K[coo] = Ks[coo]
    # end

    S = nothing; I = nothing; J = nothing

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
    
    # Fe = vec(Fe'); Fe = Weights(Fe)
    # I = vec(EQoLG')
    # F = StatsBase.counts(I, Fe)

    F = zeros(neq+1)

    for (i, fe) in zip(EQoLG, Fe)
        F[i] += fe
    end

    Fe = nothing; I = nothing;
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

    K = montaK!(ne, neq, dx, alpha, beta, EQoLG)

    F, xPTne = montaF(ne, neq, X, f, EQoLG)

    C = zeros(Float64, neq)

    # println("Resolvendo sistema")    
    C .= Symmetric(K)\F
    # prob = LinearProblem(Symmetric(K), F)
    # C .= solve(prob).u
    # println("Resolvendo sistema: fim")

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

# 2^25 máximo de elementos que meu pc aguenta: 16GB de RAM
alpha = 1; beta = 1; a = 0; b = 1; ne = 2^23
f(x) = x; u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1)); u_x(x) = 1 - (exp(-x) + exp(x)) *(1/(exp(1.0) - exp(-1)));

println("Rodando")
# @profview begin
#     C, X, EQoLG = solveSys(alpha, beta, ne, a, b, f, u)

#     C = nothing; X = nothing; EQoLG = nothing
# end

function convergence_test!(NE, E, dE)
    alpha = 1; beta = 1; a = 0; b = 1;
    f(x) = x; u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1)); u_x(x) = 1 - (exp(-x) + exp(x)) *(1/(exp(1.0) - exp(-1)));

    for i = 1:lastindex(NE)
        # println("Iniciando i = ", i)
        # solveSys(alpha, beta, NE[i], a, b, f, u)
        Ci, EQoLGi, xPTnei = solveSys(alpha, beta, NE[i], a, b, f, u)
        E[i], dE[i] = erroVet(NE[i], EQoLGi, Ci, u, u_x, xPTnei)
    end
end

errsize = 23
NE = 2 .^ [2:1:errsize;]
H = 1 ./NE
E = zeros(length(NE))
dE = similar(E)

# 6.103s directly banded
# 9.299s convert to banded
# 14.438 not banded
@profview begin
    convergence_test!(NE, E, dE)
end

### Tempo no MatLab: 4.25 segundos NE = [2:2^23]

# size(E)
plot(H, E, xaxis=:log2, yaxis=:log2)
plot!(H, H .^2, xaxis=:log2, yaxis=:log2)
# println(E)

# @btime convergence_test(23)

GC.gc()

############ TESTES ############

dx = 1/ne; neq = ne - 1; EQ = montaEQ(ne, neq); LG = montaLG(ne); EQoLG = EQ[LG]; EQoLGT = EQoLG'

function teste(alpha, beta, ne, a, b, f, u, neq)

    Ke = [8.38860800000004e6 -8.38860799999998e6; -8.38860799999998e6 8.38860800000004e6]
    
    I = vec(EQoLG[[1,1,2,2], 1:1:ne])
    J = vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
    
    S = repeat(reshape(Ke, 4), outer=ne)    
    
    # Ks = sparse(I, J, S)[1:neq, 1:neq]

    K = BandedMatrix(Zeros(neq, neq), (1,1)) 

    for (i,j,s) in collect(zip(I,J,S))
        if i <= neq && j <= neq 
            K[i,j] += s
        end
    end

    # K
end

# @btime teste(alpha, beta, ne, a, b, f, u, neq)
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

# K = montaK!(ne, neq, dx, alpha, beta, EQoLG)
# b = rand(2^22-1)
# # prob = LinearProblem(A, b)

# for alg in [UMFPACKFactorization()
#     #, MKLPardisoFactorize(), MKLPardisoIterate()
#     ]
#     println(alg)
#     cholesky(K)
#   end
# A\b