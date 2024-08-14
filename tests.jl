using SparseArrays, BenchmarkTools

function fem_solution!(c,A,b)
  c .= A\b
end

c = zeros(2^15)
m = length(c)
h = 1.0/(m+1)
α = 2.0/h + 2.0*h/3.0
β = h/6.0 - 1.0/h

A = spdiagm(0 => fill(α, m), 1 => fill(β, m-1), -1 => fill(β, m-1))
b = h^2 * collect(1:m)

@benchmark fem_solution!(c,$A,$b)

zrs = nnz(A)

sizeof(c)/1024