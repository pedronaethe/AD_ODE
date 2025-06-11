using LinearAlgebra
export gdet_func, gcov_func, gcon_func, gcov_bl

function gdet_func(gcov::MMat4)
    """
    Returns the determinant of the covariant metric tensor.

    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    F = lu(gcov)
    U = F.U

    if any(abs(U[i, i]) < 1e-14 for i in 1:size(U, 1))
        @warn "Singular matrix in gdet_func!"
        return -1.0
    end

    gdet = prod(diag(U))
    return sqrt(abs(gdet))
end

function gcov_func(X::MVec4)
    """
    Returns covariant metric tensor in Kerr-Schild coordinates.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r::Float64 = 0;
    th::Float64 = 0;
    r, th = bl_coord(X)
    gcov = zeros(MMat4)
    cth = cos(th)
    sth = abs(sin(th))


    s2 = sth * sth
    rho2 = r * r + a * a * cth * cth

    tfac = 1.
    rfac = r - R0
    hfac = π
    pfac = 1.
    gcov[1, 1] = (-1. + 2. * r / rho2) * tfac * tfac
    gcov[1, 2] = (2. * r / rho2) * tfac * rfac
    gcov[1, 4] = (-2. * a * r * s2 / rho2) * tfac * pfac

    gcov[2, 1] = gcov[1, 2]
    gcov[2, 2] = (1. + 2. * r / rho2) * rfac * rfac
    gcov[2, 4] = (-a * s2 * (1. + 2. * r / rho2)) * rfac * pfac
    
    gcov[3, 3] = rho2 * hfac * hfac
    
    gcov[4, 1] = gcov[1, 4]
    gcov[4, 2] = gcov[2, 4]
    gcov[4, 4] =
        s2 * (rho2 + a * a * s2 * (1. + 2. * r / rho2)) * pfac * pfac
    
        return gcov
end



function gcon_func(gcov::MMat4)
    """
    Returns contravariant metric tensor in Kerr-Schild coordinates through matrix inversion of the covariant tensor.
    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    gcon = inv(gcov)
    if any(isnan.(gcon)) || any(isinf.(gcon))
        @error "Singular gcov encountered in gcon"
        print_matrix("gcov", gcov)
        print_matrix("gcon", gcon)
        error("Singular gcov encountered, cannot compute gcon.")
    end
    return gcon
end


function gcov_bl(r,th)
    """
    Computes the metric tensor in Boyer-Lindquist coordinates.
    Parameters:
    @r: Radial coordinate in Boyer-Lindquist coordinates.
    @th: Angular coordinate in Boyer-Lindquist coordinates.
    """
    gcov = zeros(MMat4)
    sth = sin(th)
    if(sth < 1e-40)
        sth = 10^(-40)
    end
    cth = cos(th)
    s2 = sth * sth
    if(r < 1e-40)
        r = 10^(-40)
    end
    a2 = a * a
    r2 = r * r
    DD = (1.0 - 2.0 / r + a2 / r2)
    mu = 1.0 + a2 * cth * cth / r2

    gcov[1, 1] = -(1.0 - 2.0 / (r * mu))
    gcov[1, 4] = -2.0 * a * s2 / (r * mu)
    gcov[4, 1] = gcov[1, 4]
    gcov[2, 2] = mu / (DD )
    gcov[3, 3] = r2 * mu
    gcov[4, 4] = r2 * sth * sth * (1.0 + a2 / r2 + 2.0 * a2 * s2 / (r2 * r * mu))

    #if any element of the diagonal is zero print variables
    if(gcov[1,1] == 0 || gcov[2,2] == 0 || gcov[3,3] == 0 || gcov[4,4] == 0)   
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $a, r2 $r2, a2 $a2, mu $mu, DD $DD")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
    #if any (isnan.(gcov)) || any(isinf.(gcov))
    if any(isnan.(gcov)) || any(isinf.(gcov))
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $a, r2 $r2, a2 $a2, mu $mu, DD $DD")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
    return gcov
end