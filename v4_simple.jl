using GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, 
Plots, MKL, BandedMatrices, LinearSolve

BLAS.set_num_threads(24)

KLUFactorization()

BaseTypes = (
    linearLagrange = "Lag1",
    quadraticLagrange = "Lag2",
    cubicLagrange = "Lag3",
    cubicHermite = "Her3",
)

LocalBases = (
    Lag1 = (ne) -> (type = BaseTypes.linearLagrange, p = 1, nB = 2, neq = ne - 1),
    Lag2 = (ne) -> (type = BaseTypes.quadraticLagrange, p = 2, nB = 3, neq = 2*ne - 1),
    Lag3 = (ne) -> (type = BaseTypes.cubicLagrange, p = 3, nB = 4, neq = 3*ne - 1),
    Her3 = (ne) -> (type = BaseTypes.cubicHermite, p = 3, nB = 4, neq = 2*ne)
)

function Example(alpha, beta, gamma, a, b, u, u_x, f)
    return (alpha = alpha, beta = beta, gamma = gamma, a = a, b = b, u = u, u_x = u_x, f = f)
end

function example1()
    alpha = 1.0; beta = 1.0; gamma = 0.0; a = 0.0; b = 1.0;
    u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1));
    u_x(x) = 1 - (exp(-x) + exp(x)) *(1/(exp(1.0) - exp(-1)));
    f(x) = x

    return Example(alpha, beta, gamma, a, b, u, u_x, f)
end

function example2()
    alpha = pi; beta = exp(1.0); gamma = 0.0; a = 0.0; b = 1.0;
    u(x) = sin(pi*x);
    u_x(x) = cos(pi*x)*pi; u_xx(x) = -sin(pi*x)*pi^2;
    f(x) = -alpha*u_xx(x) + beta*u(x)

    return Example(alpha, beta, gamma, a, b, u, u_x, f)
end

function example3()
    alpha = 1.0; beta = 1.0; gamma = 1.0; a = 0.0; b = 1.0;
    u(x) = 4*(x - 1/2)^2 - 1;
    u_x(x) = 8*(x - 1/2); u_xx(x) = 8;
    f(x) = -alpha*u_xx(x) + gamma*u_x(x) + beta*u(x)

    return Example(alpha, beta, gamma, a, b, u, u_x, f)
end

function examples(case)
    if case == 1
        return example1()
    elseif case == 2
        return example2()
    elseif case == 3
        return example3()
    end
end

function montaLG(ne, base)
    LG = zeros(Int64, ne, base.nB)

    LG1 = [(base.nB-1)*i - (base.nB-2) for i in 1:ne]
    
    for a in 1:base.nB
        LG[:,a] .= LG1 .+ (a-1)
    end

    return LG'
end

function montaEQ(ne, neq, base)
    EQ = zeros(Int64, (base.nB-1)*ne+1, 1) .+ (neq+1)
    EQ[2:end-1] .= 1:1:neq

    return EQ
end

function PHI(P, base)
    if base.type == BaseTypes.linearLagrange
        return [(-1*P.+1)./2, (P.+1)./2]
    elseif base.type == BaseTypes.quadraticLagrange
        return [(P.-1).*P./2, 
                (1 .-P).*(1 .+P),
                (1 .+P).*P./2]
    elseif base.type == BaseTypes.cubicLagrange
        return [ (9/16)*(1 .- P).*(P .+ (1/3)).*(P .- (1/3)),
                (27/16)*(1 .+ P).*(P .- (1/3)).*(P .- 1),
                (27/16)*(1 .- P).*(P .+ (1/3)).*(P .+ 1),
                 (9/16)*(1 .+ P).*(P .+ (1/3)).*(P .- (1/3))]
    end
end

function dPHI(P, base)
    if base.type == BaseTypes.linearLagrange
        return [-1/2*(P.^0), 1/2*(P.^ 0)]
    elseif base.type == BaseTypes.quadraticLagrange
        return [ P .- 1/2, 
                -2 .*P,
                P .+1/2]
    elseif base.type == BaseTypes.cubicLagrange
        return [ (1/16)*(9   *(2  .- 3*P).*P .+ 1),
                 (9/16)*(  P.*(9*P.- 2) .- 3),
                (-9/16)*(  P.*(9*P.+ 2) .- 3),
                 (1/16)*(9*P.*(3*P.+ 2) .- 1)]
    end
end

function montaK(base, ne, neq, dx, alpha, beta, gamma, EQoLG::Matrix{Int64})
    npg = base.nB; P, W = legendre(npg)

    phiP = reduce(hcat, PHI(P, base))'; dphiP = reduce(hcat, dPHI(P, base))'

    parcelaNormal = beta*dx/2 * (W'.*phiP) * phiP';
    parcelaDerivada1 = gamma * (W'.*dphiP) * phiP';
    parcelaDerivada2 = 2*alpha/dx * (W'.*dphiP) * dphiP';

    Ke = parcelaDerivada2 + parcelaDerivada1 + parcelaNormal

    base_idxs = 1:base.nB

    I = vec(@view EQoLG[repeat(1:base.nB, inner=base.nB), 1:1:ne])
    J = vec(@view EQoLG[base_idxs, repeat(1:1:ne, inner=base.nB)])
    S = repeat(vec(Ke), outer=ne)

    K = BandedMatrix(Zeros(neq, neq), (base.nB-1, base.nB-1))
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
    fxPTne = similar(xPTne)

    Threads.@threads for i in eachindex(xPTne)
        fxPTne[i] = f(xPTne[i])
    end

    Fe = dx/2 .* W'.*phiP * fxPTne

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

    mul!(cache1, phiP, cEQoLG)
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

function solveSys(base, alpha, beta, gamma, ne, a, b, f, u)
    dx = 1/ne;
    neq = base.neq;

    X = a:dx:b
    
    EQ = montaEQ(ne, neq, base); LG = montaLG(ne, base)
    EQoLG = EQ[LG]

    EQ = nothing; LG = nothing;

    K = montaK(base, ne, neq, dx, alpha, beta, gamma, EQoLG)

    F, xPTne = montaF(base, ne, neq, X, f, EQoLG)

    C = zeros(Float64, neq)

    C .= K\F

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

println("Rodando \n")
let
    alpha, beta, gamma, a, b, u, u_x, f = examples(3); ne = 2^15; baseType = BaseTypes.linearLagrange
    base = LocalBases[Symbol(baseType)](ne)
    C, EQoLG, xPTne = solveSys(base, alpha, beta, gamma, ne, a, b, f, u)
    X = vcat(a:(1/ne):b)[2:end-1]
    # C = nothing; X = nothing; EQoLG = nothing

    plot(X, u.(X)); plot!(X, C)
end

function convergence_test!(NE, E, dE, example)
    alpha, beta, gamma, a, b, u, u_x, f = examples(example)

    for i = 1:lastindex(NE)
        base = LocalBases[Symbol(baseType)](NE[i])
        Ci, EQoLGi, xPTnei = solveSys(base, alpha, beta, gamma, NE[i], a, b, f, u)
        E[i], dE[i] = erroVet(base, NE[i], EQoLGi, Ci, u, u_x, xPTnei)
    end
end

errsize = 15
NE = 2 .^ [2:1:errsize;]
H = 1 ./NE
E = zeros(length(NE))
dE = similar(E)
baseType = BaseTypes.linearLagrange
exemplo = 3

@btime convergence_test!(NE, E, dE, exemplo)
plot(H, E, xaxis=:log10, yaxis=:log10); plot!(H, H .^LocalBases[Symbol(baseType)](2).nB, xaxis=:log10, yaxis=:log10)

GC.gc()