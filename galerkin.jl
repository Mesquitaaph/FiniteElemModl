using Plots

function exemplo1()
  α = 1; β = 0
  f(x) = 8; u(x) = -4*x*(x-1)

  return α, β, f, u
end

function exemplo2()
  α = 1; β = 1
  f(x) = x; u(x) = x + (exp(-x) - exp(x))/(ℯ - exp(-1))

  return α, β, f, u
end

function exemplo3()
  α = 1; β = 1
  f(x) = -2*π^2*cos(2*π*x) + sin(π*x)^2; u(x) = sin(π*x)^2

  return α, β, f, u
end

function exemplos(exemplo)
  if exemplo == 1
    return exemplo1()
    
  elseif exemplo == 2
    return exemplo2()    

  elseif exemplo == 3
    return exemplo3()

  else
    print("Exemplo não existente")
    return nothing
  end
end

function K_galerkin(α, β, h, m)
  K = zeros(m, m)

  for i in 1:m-1
    K[i,i] = α*(1/h + 1/h) + β/3*(h+h)
    Kij = β*h/6 - α/h
    K[i,i+1] = Kij
    K[i+1, i] = Kij
  end

  K[m, m] = α*(1/h + 1/h) + β/3*(h+h)

  return K
end

function f_gauss(f, a, b)
  function g(ksi)
    dx = (b-a)/2
    x = (b-a)/2 * (ksi + 1) + a
    return f(x)*dx
  end

  return g
end

function quadratura(f, a, b)
  W = [
      [0,   0,    0,    0],
      [1,   1,    0,    0],
      [5/9, 8/9,  5/9,  0],
      [0,   0,    0,    0],
  ]

  KSI = [
      [0,           0,          0,          0],
      [-sqrt(3)/3,  sqrt(3)/3,  0,          0],
      [-sqrt(3/5),  0,          sqrt(3/5),  0],
      [0,           0,          0,          0],
  ]

  n_gauss = 3

  g = f_gauss(f, a, b)

  return sum([W[n_gauss-1][i]*g(KSI[n_gauss-1][i]) for i in 1:n_gauss])
end


function Fi_galerkin(f, i, X, h, m)
  function f1(x)
    return f(x)*(x - X[i-1])
  end

  function f2(x)
    return f(x)*(X[i+1] - x)
  end

  integral_1 = quadratura(f1, X[i-1], X[i]) # integral X[i-1] ate X[i] de f(x)(x - X[i-1])
  integral_2 = quadratura(f2, X[i], X[i+1]) # integral X[i] ate X[i+1] de f(x)(X[i+1] - x)

  return integral_1/h + integral_2/h
end

function F_galerkin(f, X, h, m)
  F = zeros(m)
  for i in 2:m+1
    F[i-1] = Fi_galerkin(f, i, X, h, m)
  end

  return F
end


function solve(exemplo, m)
  h = 1/(m+1)
  X = [(i-1)*h for i in 1:m+2]

  α, β, f, u =  exemplos(exemplo)

  K = K_galerkin(α, β, h, m)
  
  F = F_galerkin(f, X, h, m)
  
  C = K\F

  return X, C
end

exemplo = 3; m = 20

X, C = solve(exemplo, m)

D = vcat(0, C, 0)

α, β, f, u = exemplos(exemplo)
Xtest = 0:0.001:1

plot(X, D); plot!(Xtest, u)