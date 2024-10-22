using GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, Plots, MKL, BandedMatrices, LinearSolve, Printf

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

function Example(alpha, beta, gamma, sigma, a, b, u, u_x, f)
    return (alpha = alpha, beta = beta, gamma = gamma, sigma = sigma, a = a, b = b, u = u, u_x = u_x, f = f)
end

function example1()
    alpha = 1.0; beta = 1.0; gamma = 0.0; sigma(x) = 0; a = 0.0; b = 1.0;
    u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1));
    u_x(x) = 1 - (exp(-x) + exp(x)) *(1/(exp(1.0) - exp(-1)));
    f(x) = x

    return Example(alpha, beta, gamma, sigma, a, b, u, u_x, f)
end

function example2()
    alpha = pi; beta = exp(1.0); gamma = 0.0; sigma(x) = 0; a = 0.0; b = 1.0;
    u(x) = sin(pi*x);
    u_x(x) = cos(pi*x)*pi; u_xx(x) = -sin(pi*x)*pi^2;
    f(x) = -alpha*u_xx(x) + beta*u(x)

    return Example(alpha, beta, gamma, sigma, a, b, u, u_x, f)
end

function example3()
    alpha = 1.0; beta = 1.0; gamma = 1.0; sigma(x) = 0; a = 0.0; b = 1.0;
    u(x) = 4*(x - 1/2)^2 - 1;
    u_x(x) = 8*(x - 1/2); u_xx(x) = 8;
    f(x) = -alpha*u_xx(x) + gamma*u_x(x) + beta*u(x)

    return Example(alpha, beta, gamma, sigma, a, b, u, u_x, f)
end

function example4()
    alpha = 1.0; beta = 0.0; gamma = 0.0; sigma(x) = 1.0; a = 0.0; b = 1.0;
    u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1));
    u_x(x) = 1 - (exp(-x) + exp(x)) * (1/(exp(1.0) - exp(-1)));
    u_xx(x) = (1/(exp(1.0) - exp(-1))) * (-exp(x) + exp(-x));
    f(x) = -alpha*u_xx(x) + sigma(x)*u(x)

    return Example(alpha, beta, gamma, sigma, a, b, u, u_x, f)
end

function example5()
    alpha = 1.0; beta = 0.0; gamma = 0.0; sigma(x) = sin(x); a = 0.0; b = 1.0;
    u(x) = x + (exp(-x) - exp(x))/(exp(1.0) - exp(-1));
    u_x(x) = 1 - (exp(-x) + exp(x)) *(1/(exp(1.0) - exp(-1)));
    u_xx(x) = (1/(exp(1.0) - exp(-1))) * (-exp(x) + exp(-x));
    f(x) = -alpha*u_xx(x) + sigma(x)*u(x)

    return Example(alpha, beta, gamma, sigma, a, b, u, u_x, f)
end

function examples(case)
    if case == 1
        return example1()
    elseif case == 2
        return example2()
    elseif case == 3
        return example3()
    elseif case == 4
        return example4()
    elseif case == 5
        return example5()
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

function montaLG_serial(ne, base)
    LG = zeros(Int64, ne, base.nB)

    LG1 = [(base.nB-1)*i - (base.nB-2) for i in 1:ne]
    
    for e in 1:ne
        for a in 1:base.nB
            @inbounds LG[e,a] = LG1[e] + (a-1)
        end
    end

    return LG'
end

function montaEQ_serial(ne, neq, base)
    EQ = zeros(Int64, (base.nB-1)*ne+1, 1)

    EQ[1] = neq+1; EQ[end] = neq+1
    
    for i in 2:(base.nB-1)*ne
        @inbounds EQ[i] = i-1
    end

    return EQ
end

function PHI_serial(P, base)
    if base.type == BaseTypes.linearLagrange
        return [(-P+1)/2, (P+1)/2]
    end
end

function PHI(P, base)
    if base.type == BaseTypes.linearLagrange
        return [(-1*P.+1)./2, (P.+1)./2]
    end
end

function dPHI_serial(P, base)
    if base.type == BaseTypes.linearLagrange
        return [-1/2, 1/2]
    end
end

function dPHI(P, base)
    if base.type == BaseTypes.linearLagrange
        return [-1/2*(P.^0), 1/2*(P.^ 0)]
    end
end

function montaxPTne(dx, X, P)
    return (dx ./ 2) .* (P .+ 1) .+ X
end

function xksi(ksi, e, X)
    h = X[e+1] - X[e]
    return h/2*(ksi+1) + X[e]
end

function montaK(base, ne, neq, dx, alpha, beta, gamma, sigma, EQoLG::Matrix{Int64}, X)
    npg = base.p+1; P, W = legendre(npg)

    phiP = reduce(hcat, PHI(P, base)); dphiP = reduce(hcat, dPHI(P, base))

    Ke = zeros(Float64, 2, 2)

    Ke .+= beta*dx/2 * (W'.*phiP') * phiP; # parcelaNormal
    Ke .+= gamma * (W'.*dphiP') * phiP; # parcelaDerivada1
    Ke .+= 2*alpha/dx * (W'.*dphiP') * dphiP; # parcelaDerivada2

    K = BandedMatrix(Zeros(neq, neq), (base.nB-1, base.nB-1))
    for e in 1:ne
        for b in 1:2
            @inbounds j = EQoLG[b, e]
            for a in 1:2
                @inbounds i = EQoLG[a, e]
                if i <= neq && j <= neq
                    @inbounds K[i,j] += Ke[a,b]
                end
            end
        end
    end

    # base_idxs = 1:base.nB


    # I = vec(@view EQoLG[repeat(1:base.nB, inner=base.nB), 1:1:ne])
    # J = vec(@view EQoLG[base_idxs, repeat(1:1:ne, inner=base.nB)])
    # S = repeat(vec(Ke), outer=ne)

    # K = sparse(I, J, S)
    # for (i,j,s) in zip(I,J,S)
    #     if i <= neq && j <= neq
    #         @inbounds K[i,j] += s
    #     end
    # end

    S = nothing; I = nothing; J = nothing

    return K
end

function montaKSerial(base, ne, neq, dx, alpha, beta, gamma, sigma, EQoLG::Matrix{Int64}, X)
    npg = base.p+1; P, W = legendre(npg)

    phiP(ksi, a) = PHI(ksi, base)[a]; dphiP(ksi, a) = dPHI(ksi, base)[a];

    Ke = zeros(Float64, 2, 2)

    for a in 1:2
        for b in 1:2
            for ksi in 1:npg
                @inbounds parcelaNormal = beta*dx/2 * W[ksi] * phiP(P[ksi], a) * phiP(P[ksi], b);
                @inbounds parcelaDerivada1 = gamma * W[ksi] * phiP(P[ksi], a) * dphiP(P[ksi], b);
                @inbounds parcelaDerivada2 = 2*alpha/dx * W[ksi] * dphiP(P[ksi], a) * dphiP(P[ksi], b);

                @inbounds Ke[a,b] += parcelaDerivada2 + parcelaNormal + parcelaDerivada1
            end
        end
    end

    K = BandedMatrix(Zeros(neq, neq), (base.nB-1, base.nB-1))
    for e in 1:ne
        for b in 1:2
            @inbounds j = EQoLG[b, e]
            for a in 1:2
                @inbounds i = EQoLG[a, e]
                if i <= neq && j <= neq
                    @inbounds K[i,j] += Ke[a,b]
                end
            end
        end
    end

    return K
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

function montaF_serial(base, ne, neq, X, f::Function, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = zeros(npg, base.nB)
    for a in 1:2
        for ksi in 1:npg
            phiP[ksi, a] = PHI(P[ksi], base)[a]
        end
    end
    
    dx = 1/ne
    
    F = zeros(neq+1)
    xPTne = zeros(npg, ne)
    for e in 1:ne
        for ksi in 1:npg
            @inbounds fxptne = f(xksi(P[ksi], e, X))
            @inbounds xPTne[ksi, e] = fxptne
            for a in 1:2
                @inbounds partial = dx/2 * W[ksi] * phiP[ksi, a]
                @inbounds i = EQoLG[a, e]
                @inbounds F[i] += partial * fxptne
            end
        end
    end

    return F[1:neq], xPTne
end

function erroVet(base, ne, EQoLG, C, u, u_x, xPTne)
    npg = 5; P, W = legendre(npg)
    
    phiP = reduce(hcat, PHI(P, base)); dphiP = reduce(hcat, dPHI(P, base))

    h = 1/ne

    cEQoLG = vcat(C, 0)[EQoLG]

    EL2 = sqrt(h/2 * sum(W' * ((u.(xPTne) - (phiP * cEQoLG)).^2)))

    EH01 = sqrt(h/2 * sum(W' * ((u_x.(xPTne) - (2/h .* phiP * cEQoLG)).^2)))

    return EL2, EH01
end

function erroVet_serial(base, ne, EQoLG, C, u, u_x, xPTne)
    npg = 5; P, W = legendre(npg)
    
    phiP(ksi, a) = PHI(ksi, base)[a]; dphiP(ksi, a) = dPHI(ksi, base)[a];

    h = 1/ne

    cEQoLG = vcat(C, 0)[EQoLG]

    EL2 = 0.0
    for e in 1:ne
        @inbounds c1e = cEQoLG[1, e]
        @inbounds c2e = cEQoLG[2, e]
        for ksi in 1:npg
           @inbounds EL2 += W[ksi] * (u(xPTne[ksi, e]) - (phiP(P[ksi], 1) * c1e) - (phiP(P[ksi], 2) * c2e))^2
        end
    end

    EL2 = sqrt(h/2 * EL2)

    EH01 = 0.0
    # EH01 = sqrt(h/2 * sum(W' * ((u_x.(xPTne) - (2/h .* dphiP * cEQoLG)).^2)))

    return EL2, EH01
end

function solveSys(base, alpha, beta, gamma, sigma, ne, a, b, f, u)
    dx = 1/ne;
    neq = base.neq;

    X = a:dx:b
    
    EQ = montaEQ(ne, neq, base); LG = montaLG(ne, base)

    EQoLG = EQ[LG]

    EQ = nothing; LG = nothing;
    
    K = montaK(base, ne, neq, dx, alpha, beta, gamma, sigma, EQoLG, X)

    F, xPTne = montaF(base, ne, neq, X, f, EQoLG)

    C = zeros(Float64, neq)

    C .= K\F

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

function solveSys_serial(base, alpha, beta, gamma, sigma, ne, a, b, f, u)
    dx = 1/ne;
    neq = base.neq;

    X = a:dx:b
    
    EQ = montaEQ_serial(ne, neq, base); LG = montaLG_serial(ne, base)
    EQoLG = EQ[LG]

    EQ = nothing; LG = nothing;
    
    K = montaKSerial(base, ne, neq, dx, alpha, beta, gamma, sigma, EQoLG, X)

    F, xPTne = montaF_serial(base, ne, neq, X, f, EQoLG)

    C = zeros(Float64, neq)

    C .= K\F

    F = nothing; K = nothing;

    return C, EQoLG, xPTne
end

println("Rodando \n")
# let
#     alpha, beta, gamma, sigma, a, b, u, u_x, f = examples(3); ne = 2^3; baseType = BaseTypes.linearLagrange
#     base = LocalBases[Symbol(baseType)](ne)
#     C, EQoLG, xPTne = solveSys(base, alpha, beta, gamma, sigma, ne, a, b, f, u)
#     X = vcat(a:(1/ne):b)[2:end-1]
#     # C = nothing; X = nothing; EQoLG = nothing

#     plot(X, u.(X)); plot!(X, C)
# end

function convergence_test!(NE, E, dE, example, serial)
    alpha, beta, gamma, sigma, a, b, u, u_x, f = examples(example)

    if serial
        for i = 1:lastindex(NE)
            base = LocalBases[Symbol(baseType)](NE[i])
            Ci, EQoLGi, xPTnei = solveSys_serial(base, alpha, beta, gamma, sigma, NE[i], a, b, f, u)
            E[i], dE[i] = erroVet_serial(base, NE[i], EQoLGi, Ci, u, u_x, xPTnei)
        end
    else
        for i = 1:lastindex(NE)
            base = LocalBases[Symbol(baseType)](NE[i])
            Ci, EQoLGi, xPTnei = solveSys(base, alpha, beta, gamma, sigma, NE[i], a, b, f, u)
            E[i], dE[i] = erroVet(base, NE[i], EQoLGi, Ci, u, u_x, xPTnei)
        end
    end
end

errsize = 23
NE = 2 .^ [2:1:errsize;]
H = 1 ./NE
E = zeros(length(NE))
dE = similar(E)
baseType = BaseTypes.linearLagrange
exemplo = 1

@btime convergence_test!(NE, E, dE, exemplo, false)
# plot(H, E, xaxis=:log10, yaxis=:log10); plot!(H, H .^LocalBases[Symbol(baseType)](2).nB, xaxis=:log10, yaxis=:log10)


GC.gc()

# alph, beta, gamma, sigma, a, b, u, u_x, f = examples(exemplo)
# ne = 2^23

# base = LocalBases[Symbol(baseType)](ne)

# dx = 1/ne;
# neq = base.neq;

# X = a:dx:b

# EQ = montaEQ(ne, neq, base); LG = montaLG(ne, base)

# EQoLG = EQ[LG]

# EQ = nothing; LG = nothing;

# @printf("ne = %f, Matrix Type = %s, elapsed time =", ne, "Serial")
# @benchmark montaF_serial(base, ne, neq, X, f, EQoLG)

# @printf("ne = %f, Matrix Type = %s, elapsed time =", ne, "Vetorized")
# @benchmark montaK(base, ne, neq, dx, alph, beta, gamma, sigma, EQoLG, X)

# GC.gc()