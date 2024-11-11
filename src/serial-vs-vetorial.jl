using 
    GaussQuadrature, StatsBase, SparseArrays, BenchmarkTools, LinearAlgebra, Plots, 
    MKL, BandedMatrices, LinearSolve, Printf, DataFrames, Latexify, Statistics

include.(["utils.jl", "examples.jl", "convergence.jl", "serial.jl", "vetorial.jl"])

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

function xksi(ksi, e, X)
    h = X[e+1] - X[e]
    return h/2*(ksi+1) + X[e]
end

println("Rodando \n")
# teste = @benchmark begin
#     alpha, beta, gamma, sigma, a, b, u, u_x, f = examples(3); ne = 2^2; baseType = BaseTypes.linearLagrange
#     base = LocalBases[Symbol(baseType)](ne)
#     C, EQoLG, xPTne = solveSys(base, alpha, beta, gamma, sigma, ne, a, b, f, u)
#     # X = vcat(a:(1/ne):b)[2:end-1]
#     # C = nothing; X = nothing; EQoLG = nothing

#     # plot(X, u.(X)); plot!(X, C)
# end

# teste
# maximum(teste.times)/1000_000

errsize = 10
NE = 2 .^ [2:1:errsize;]
H = 1 ./NE
E = zeros(length(NE))
dE = similar(E)
baseType = BaseTypes.linearLagrange
exemplo = 1

b1 = @benchmark convergence_test!(NE, E, dE, exemplo, true)

b2 = @benchmark convergence_test!(NE, E, dE, exemplo, false)

display(b)
display(b1)

plot(H, E, xaxis=:log10, yaxis=:log10); plot!(H, H .^LocalBases[Symbol(baseType)](2).nB, xaxis=:log10, yaxis=:log10)


GC.gc()

alph, beta, gamma, sigma, a, b, u, u_x, f = examples(exemplo)

# bench_vetorial = zeros(Float64, size(NE))
# for i in 1:lastindex(NE)
#     ne = NE[i]; dx = 1/ne
    
#     base = LocalBases[Symbol(baseType)](ne)
    
#     neq = base.neq; X = a:dx:b
    
#     EQ = montaEQ(ne, neq, base); LG = montaLG(ne, base)
    
#     EQoLG = EQ[LG]; EQ = nothing; LG = nothing;

#     bench = measure_func(montaK, (base, ne, neq, dx, alph, beta, gamma, sigma, EQoLG, X))
#     bench_vetorial[i] = bench.mean
# end

# bench_serial = similar(bench_vetorial)
# for i in 1:lastindex(NE)
#     ne = NE[i]; dx = 1/ne
    
#     base = LocalBases[Symbol(baseType)](ne)
    
#     neq = base.neq; X = a:dx:b
    
#     EQ = montaEQ_serial(ne, neq, base); LG = montaLG_serial(ne, base)
    
#     EQoLG = EQ[LG]; EQ = nothing; LG = nothing;

#     bench = measure_func(montaKSerial, (base, ne, neq, dx, alph, beta, gamma, sigma, EQoLG, X))
#     bench_serial[i] = bench.mean
# end

# sum_serial = sum(bench_serial)
# sum_vetorial = sum(bench_vetorial)

# df = DataFrame(
#     ne=vcat(string.("\$2^{", [2:1:errsize;],"}\$"), "sum"), 
#     serial=format_num.(vcat(bench_serial, sum_serial)), 
#     vetorial=format_num.(vcat(bench_vetorial, sum_vetorial)), 
#     speedup=first.(string.(vcat(bench_serial./bench_vetorial, sum_serial/sum_vetorial)), 7)
# )

# latexify(df; env = :table, booktabs = true, latex = false) |> print

# p = plot(
#     [2:1:errsize;], [bench_serial, bench_vetorial], yaxis=:log10, 
#     labels=["Serial" "Vetorial"], xlabel="Número de elementos (\$2^n\$)", ylabel="Tempo de execução (ns)",
#     yticks=10 .^[1:1:10;], fmt = :png
# ); display(p)
# # savefig(p, "Vetorial vs Serial - Nº Elementos x Tempo.png")

# plot(
#     [2:1:errsize;], bench_serial./bench_vetorial, 
#     label="Speedup", xlabel="Número de elementos (\$2^n\$)", ylabel="Speedup",
#     yticks=[0:.125:4;]
# )

GC.gc()