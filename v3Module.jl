module v3Module
    using GaussQuadrature, SparseArrays, StatsBase, LinearAlgebra
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

    function montaK(ne, neq, dx, alpha, beta, EQoLG::Matrix{Int64})
        npg = 2; P, W = legendre(npg)
        
        phiP = reduce(vcat, PHI(P)'); dphiP = hcat(dPHI.(P)...)
        
        Ke = 2*alpha/dx .* (W.*dphiP) * dphiP' + beta*dx/2 .* (W.*phiP) * phiP'

        I = vec(EQoLG[[1,1,2,2], 1:1:ne])
        J = vec(EQoLG[[1,2], repeat(1:1:ne, inner=2)])
        
        S = repeat(reshape(Ke, 4), outer=ne)
        
        K = sparse(I, J, S)[1:neq, 1:neq]
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
        
        Fe = vec(Fe'); Fe = Weights(Fe)

        I = vec(EQoLG')
        F = StatsBase.counts(I, Fe)

        xPTne = nothing; Fe = nothing; I = nothing;
        return F[1:neq]
    end

    function erroVet(ne, EQoLG, C, u, X)
        npg = 5; P, W = legendre(npg)
        xPTne = montaxPTne(dx, X[1:end-1]', P)
        phiP = reduce(vcat, PHI(P)')
        h = 1/ne

        d = vcat(C, 0)

        E = sqrt(h/2 * sum(W' * ((u.(xPTne) - (phiP' * d[EQoLG])).^2)))

        return E
    end

    function solve(alpha, beta, ne, a, b, f, u)
        dx = 1/ne; neq = ne-1

        X = a:dx:b
        
        EQ = montaEQ(ne, neq); LG = montaLG(ne)
        EQoLG = EQ[LG]

        EQ = nothing; LG = nothing;

        K = montaK(ne, neq, dx, alpha, beta, EQoLG)

        F = montaF(ne, neq, X, f, EQoLG)

        C = Symmetric(K)\F

        F = nothing; K = nothing;
        GC.gc()
        return C, X, EQoLG
    end

    function convergence_test(errsize)
        alpha = 1; beta = 1; a = 0; b = 1;
        f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

        NE = 2 .^ [2:1:errsize;]
        H = 1 ./ NE
        
        E = zeros(length(NE))

        for i = 1:lastindex(NE)
            Ci, Xi, EQoLGi = solve(alpha, beta, NE[i], a, b, f, u)
            E[i] = erroVet(NE[i], EQoLGi, Ci, u, Xi)
        end

        return E, H
    end

    export solve, convergence_test
end