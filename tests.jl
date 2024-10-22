using SparseArrays, BenchmarkTools, LinearSolve, BandedMatrices, SuiteSparse

function fem_solution_sparse!(c,A,b)
  # c .= A\b

  prob = LinearProblem(A, b)
  sol = solve(prob, KLUFactorization())
  c .= sol.u
end

function fem_solution!(c,A,b)
  c .= A\b
end

c = zeros(2^23)
m = length(c)
h = 1.0/(m+1)
α = 2.0/h + 2.0*h/3.0
β = h/6.0 - 1.0/h

A = spdiagm(0 => fill(α, m), 1 => fill(β, m-1), -1 => fill(β, m-1))
b = h^2 * collect(1:m)

S = BandedMatrix(Zeros(m, m), (1, 1))

function fill_banded()
  Threads.@threads for i in eachindex(1:m)
    if i-1>0
      @inbounds S[i,i-1] = A[i,i-1]
    end
    if i+1<=m
      @inbounds S[i,i+1] = A[i,i+1]
    end
    @inbounds S[i,i] = A[i,i]
  end
end

@benchmark fill_banded()

@benchmark S = BandedMatrix(A)

# @benchmark fem_solution!(c,S,b)

# @benchmark fem_solution_sparse!(c,A,b)