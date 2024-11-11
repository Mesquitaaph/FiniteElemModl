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