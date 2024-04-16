using GaussQuadrature, SparseArrays, StatsBase, BenchmarkTools, LinearAlgebra, Plots, StaticArrays

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
    return [-1/2*(P^0); 1/2*(P^0)]
end

function montaK!(ne, neq, dx, alpha, beta, EQoLG::Matrix{Int64})
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
    return F[1:neq]#, xPTne
end

function erroVet(ne, EQoLG, C, u, X)
    dx = 1/ne
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

    K = montaK!(ne, neq, dx, alpha, beta, EQoLG)

    F = montaF(ne, neq, X, f, EQoLG)

    # C = K\F
    C = Symmetric(K)\F

    F = nothing; K = nothing;
    # @show_locals
    return C, X, EQoLG
end

# 2^25 máximo de elementos que meu pc aguenta: 16GB de RAM
alpha = 1; beta = 1; a = 0; b = 1; ne = 2^23
f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

# @btime begin
#     C, X, EQoLG = solve(alpha, beta, ne, a, b, f, u)

#     C = nothing; X = nothing; EQoLG = nothing
# end

function convergence_test!(NE, E)
    alpha = 1; beta = 1; a = 0; b = 1;
    f(x) = x; u(x) = x + (ℯ^(-x) - ℯ^x)/(ℯ - ℯ^(-1))

    for i = 1:lastindex(NE)
        # println("Iniciando i = ", i)
        # solve(alpha, beta, NE[i], a, b, f, u)
        Ci, Xi, EQoLGi = solve(alpha, beta, NE[i], a, b, f, u)
        E[i] = erroVet(NE[i], EQoLGi, Ci, u, Xi)
    end
end

errsize = 23
NE = 2 .^ [2:1:errsize;]

E = zeros(length(NE))

@btime begin
    convergence_test!(NE, E)
end

### Tempo no MatLab: 4.25 segundos NE = [2:2^23]

# size(E)
# plot!(H, E, xaxis=:log2, yaxis=:log2)
# plot!(H, H .^2, xaxis=:log2, yaxis=:log2)

# @btime convergence_test(23)

GC.gc()

############ TESTES ############

# dx = 1/ne; neq = ne - 1; EQ = montaEQ(ne, neq); LG = montaLG(ne); EQoLG = EQ[LG]; EQoLGT = EQoLG'

# function teste(alpha, beta, ne, a, b, f, u, neq)
#     LG1 = 1:1:ne
    
#     LG = Matrix{Int64}(undef, ne, 2)

#     LG[:,1] .= LG1
#     LG[:,2] .= LG1.+1

#     return LG'
# end

# @btime teste(alpha, beta, ne, a, b, f, u, neq)

# function teste1(alpha, beta, ne, a, b, f, u, neq)
#     LG = Matrix{Int64}(undef, ne, 2)

#     LG[:,1] .= 1:1:ne
#     LG[:,2] .= 2:1:(ne+1)

#     return LG'
# end

# @btime teste1(alpha, beta, ne, a, b, f, u, neq)

test = 2 .^ [2:1:4;]

test[2]