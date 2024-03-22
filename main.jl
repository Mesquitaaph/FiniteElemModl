using GaussQuadrature, SparseArrays

run(`cmd /c cls`)

function montaLG(ne)
    LG1 = [1:1:ne;]
    LG = Matrix{Int64}(undef, 2, ne)
    LG[1,:] = LG1[:]
    LG[2,:] = (LG1.+1)[:]

    return LG
end

function montaEQ(ne, neq)
    EQ = zeros(Int64, ne+1, 1)
    EQ[2:length(EQ)-1] = [1:1:neq;]

    return EQ
end


function PHI(P)
    return [(1-P)/2, (1+P)/2]
end

function dPHI(P)
    return [-1/2*(P^0); 1/2*(P^0)]
end

function montaK(ne, neq, dx, alpha, beta, EQoLG)
    K = spzeros(neq, neq)
    npg = 2; P, W = legendre(npg)

    phiP = hcat(PHI.(P)...); dphiP = hcat(dPHI.(P)...)

    Ke = 2*alpha/dx * (W.*dphiP) * dphiP' + beta*dx/2 * (W.*phiP) * phiP'

    for e = 1:ne
        for a = 1:2
            i = EQoLG[a,e]
            for b = 1:2
                j = EQoLG[b,e]
                if i != 0 && j != 0
                    K[i,j] += Ke[a,b]
                end
            end
        end
    end

    return K
end

function montaxPTne(dx, X)
    return f(P) = (dx/2)*(P+1) .+ X
end

function montaF(ne, neq, X, f, EQoLG)
    npg = 5; P, W = legendre(npg)

    phiP = hcat(PHI.(P)...); dphiP = hcat(dPHI.(P)...)
    
    dx = 1/ne
    xPTne = hcat(montaxPTne(dx, X[1:end-1]).(P)...)'

    Fe = (dx/2*W'.*phiP) * f(xPTne)

    F = zeros(neq, 1)
    for e = 1:ne
        for a = 1:2
            i = EQoLG[a,e]
            if i != 0
                F[i] += Fe[a,e]
            end
        end
    end

    return F
end

function solve()
    alpha = 1; beta = 1; ne = 2^16+1; dx = 1/ne; neq = ne-1
    a = 0; b = 1
    f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

    X = [a:dx:b;]

    EQ = montaEQ(ne, neq)
    LG = montaLG(ne)
    
    EQoLG = EQ[LG]
    
    K = montaK(ne, neq, dx, alpha, beta, EQoLG)

    F = montaF(ne, neq, X, f, EQoLG)'

    C = F/K

    return C
end

@time begin
    solve()
end