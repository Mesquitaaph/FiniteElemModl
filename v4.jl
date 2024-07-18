using GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, 
Plots, MKL, BandedMatrices, LinearSolve

# BLAS.get_config()
# versioninfo()
# BLAS.get_num_threads()
BLAS.set_num_threads(24)

KLUFactorization()

# struct LocalBase
#     type:: String
#     p:: Integer
#     nB:: Integer
#     neq:: Integer
# end

# struct BaseTypes
#     linearLagrange::LocalBase
#     quadraticLagrange::LocalBase
#     cubicLagrange::LocalBase
#     cubicHermite::LocalBase
# end

# bases = BaseTypes("Lag1", "Lag2", "Lag3", "Her3")

const BaseTypes = (
    linearLagrange = "Lag1",
    quadraticLagrange = "Lag2",
    cubicLagrange = "Lag3",
    cubicHermite = "Her3",
)

const LocalBases = (
    Lag1 = (ne) -> (type = BaseTypes.linearLagrange, p = 1, nB = 2, neq = ne - 1),
    Lag2 = (ne) -> (type = BaseTypes.quadraticLagrange, p = 2, nB = 3, neq = 2*ne - 1),
    Lag3 = (ne) -> (type = BaseTypes.cubicLagrange, p = 3, nB = 4, neq = 3*ne - 1),
    Her3 = (ne) -> (type = BaseTypes.cubicHermite, p = 3, nB = 4, neq = 2*ne)
)

LocalBases[Symbol(BaseTypes.linearLagrange)](4)

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

function montaLG(ne, base)
    LG = zeros(Int64, ne, base)

    LG1 = [(base-1)*i - (base-2) for i in 1:ne]
    
    for a in 1:base
        LG[:,a] .= LG1 .+ (a-1)
    end

    return LG'
end

function montaEQ(ne, neq, base)
    EQ = zeros(Int64, (base-1)*ne+1, 1) .+ (neq+1)
    EQ[2:end-1] .= 1:1:neq

    return EQ
end

function PHI(P, base)
    if base == 2
        return [(-1*P.+1)./2, (P.+1)./2]
    elseif base == 3
        return [(P.-1).*P./2, 
                (1 .-P).*(1 .+P),
                (1 .+P).*P./2]
    elseif base == 4
        return [ (9/16)*(1 .- P).*(P .+ (1/3)).*(P .- (1/3)),
                (27/16)*(1 .+ P).*(P .- (1/3)).*(P .- 1),
                (27/16)*(1 .- P).*(P .+ (1/3)).*(P .+ 1),
                 (9/16)*(1 .+ P).*(P .+ (1/3)).*(P .- (1/3))]
    end
end

function dPHI(P, base)
    if base == 2
        return [-1/2*(P.^0), 1/2*(P.^ 0)]
    elseif base == 3
        return [ P .- 1/2, 
                -2 .*P,
                P .+1/2]
    elseif base == 4
        return [ (1/16)*(9   *(2  .- 3*P).*P .+ 1),
                 (9/16)*(  P.*(9*P.- 2) .- 3),
                (-9/16)*(  P.*(9*P.+ 2) .- 3),
                 (1/16)*(9*P.*(3*P.+ 2) .- 1)]
    end
end

function montaK(base, ne, neq, dx, alpha, beta, EQoLG::Matrix{Int64})
    npg = base; P, W = legendre(npg)

    phiP = reduce(hcat, PHI(P, base))'; dphiP = reduce(hcat, dPHI(P, base))'

    cache1 = zeros(Float64, base, base); cache2 = similar(cache1);

    cache1 .= W'.*dphiP
    mul!(cache2, cache1, dphiP')

    cache1 .= W'.*phiP    
    mul!(cache2, cache1, phiP', beta*dx/2, 2*alpha/dx)

    base_idxs = 1:base

    I = vec(@view EQoLG[repeat(1:base, inner=base), 1:1:ne])
    J = vec(@view EQoLG[base_idxs, repeat(1:1:ne, inner=base)])
    S = repeat(vec(cache2), outer=ne)

    K = BandedMatrix(Zeros(neq, neq), (base-1, base-1))
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

function montaF(base, ne, neq, X, f::Function, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = reduce(vcat, PHI(P, base)')
    
    dx = 1/ne

    xPTne = montaxPTne(dx, X[1:end-1]', P)
    # fxPTne = f.(xPTne)
    fxPTne = similar(xPTne)

    Threads.@threads for i in eachindex(xPTne)
        fxPTne[i] = f(xPTne[i])
    end

    Fe = zeros(Float64, base, ne)
    mul!(Fe, dx/2 .* W'.*phiP, fxPTne)

    F = zeros(neq+1)

    for (i, fe) in zip(EQoLG, Fe)
        F[i] += fe
    end

    Fe = nothing;
    return F[1:neq], xPTne
end

function erroVet(base, ne, EQoLG, C, u, u_x, xPTne)
    npg = 5; P, W = legendre(npg)
    
    phiP = reduce(hcat, PHI(P, base)); dphiP = reduce(hcat, dPHI(P, base))

    h = 1/ne

    cEQoLG = vcat(C, 0)[EQoLG]

    cache1 = zeros(Float64, npg, ne)
    cache2 = zeros(Float64, 1, ne)

    mul!(cache1, phiP, cEQoLG)    # phiP * vcat(C, 0)[EQoLG]
    cache1 .-= u.(xPTne)
    cache1 .^= 2
    mul!(cache2, W', cache1)
    EL2::Float64 = sqrt(h/2 * sum(cache2))

    # E = sqrt(h/2 * sum(W' * ((u.(xPTne) - (phiP' * d[EQoLG])).^2)))

    mul!(cache1, dphiP, cEQoLG, 2/h, 1.0)
    cache1 .-= u_x.(xPTne)
    cache1 .^= 2
    mul!(cache2, W', cache1)
    EH01::Float64 = sqrt(h/2 * sum(cache2))

    return EL2, EH01
end

function solveSys(base, alpha, beta, ne, a, b, f, u)
    dx = 1/ne;
    neq = (base-1)*ne - 1;

    X = a:dx:b
    
    EQ = montaEQ(ne, neq, base); LG = montaLG(ne, base)
    EQoLG = EQ[LG]

    EQ = nothing; LG = nothing;

    K = montaK(base, ne, neq, dx, alpha, beta, EQoLG)

    F, xPTne = montaF(base, ne, neq, X, f, EQoLG)

    C = zeros(Float64, neq)

    # println("Resolvendo sistema")
    C .= Symmetric(K)\F
    # println("Resolvendo sistema: fim")

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

println("Rodando \n")
# let
#     alpha, beta, a, b, u, u_x, f = examples(2); ne = 2^2; base = 3
#     C, EQoLG, xPTne = solveSys(base, alpha, beta, ne, a, b, f, u)
#     X = vcat(a:(1/ne):b)[2:end]
#     # C = nothing; X = nothing; EQoLG = nothing
#     npg = 5; P, W = legendre(npg)

#     phiP = reduce(hcat, PHI(P, base));

#     d = phiP * vcat(C, 0)[EQoLG]

#     plot(xPTne, d); plot!(xPTne, u.(xPTne))
# end

function convergence_test!(base, NE, E, dE, example)
    alpha, beta, a, b, u, u_x, f = examples(example)

    for i = 1:lastindex(NE)
        Ci, EQoLGi, xPTnei = solveSys(base, alpha, beta, NE[i], a, b, f, u)
        E[i], dE[i] = erroVet(base, NE[i], EQoLGi, Ci, u, u_x, xPTnei)
    end
end

errsize = 23
NE = 2 .^ [2:1:errsize;]
H = 1 ./NE
E = zeros(length(NE))
dE = similar(E)
base = 4
exemplo = 2

# @profview convergence_test!(NE, E, dE, 2)
# @btime convergence_test!(base, NE, E, dE, exemplo)

plot(H, E, xaxis=:log10, yaxis=:log10); plot!(H, H .^base, xaxis=:log10, yaxis=:log10)

GC.gc()

# plot(-1:0.01:1, PHI(-1:0.01:1, 4))

############ TESTES ############

ne = 2^2; dx = 1/ne;

# EQoLG
# BandedMatrix(Zeros(neq+1, neq+1), (1,1))

# Threads.@threads for ij in 1:neq^2
#     i = fld(ij-1, neq) + 1
#     j = mod(ij-1, neq) + 1
#     K[i,j] = Ks[i,j]
# end

function teste1(ne)
    base = 3; neq = (base-1)*ne - 1;

    LG = montaLG(ne, base)
    # println(LG)

    EQ = montaEQ(ne, neq, base)
    # println(EQ)

    EQoLG = EQ[LG]
    
    EQoLG1 = [vcat(neq+1, 2:2:neq-1); 1:2:neq; vcat(2:2:neq-1,neq+1)]
    println(EQoLG)
    println(EQoLG1)

    nothing
end

# teste1(ne)

# K = montaK(ne, neq, dx, alpha, beta, EQoLG)
# b = rand(neq)
# base = 3
# [(base-1)*i - (base-2) for i in 1:ne]