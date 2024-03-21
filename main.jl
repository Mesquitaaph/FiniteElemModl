using GaussQuadrature

run(`clear`)

# x, w = legendre(5)

function PHI(P)
    return [(1-P)/2 (1+P)/2]
end

function dPHI(P)
    return [-1/2*(P^0) 1/2*(P^0)]
end

function montaK(ne, dx, alpha, beta)
    npg = 2; P, W = legendre(npg)

    phiP = PHI.(P); dphiP = dPHI.(P)
    println(dphiP)
    # print(dphiP)
    Ke = 2*alpha/dx * (W.*dphiP) * dphiP' + beta*dx/2 * (W.*phiP) * phiP'

    return Ke
end

alpha = 1; beta = 1; ne = 1; dx = 1/4
montaK(ne, dx, alpha, beta)

# tst = [1 -1]