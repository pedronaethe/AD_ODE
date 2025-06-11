export get_analytic_jk

function get_fluid_nu(Kcon::MVec4, Ucov::MVec4)
    """
    Computes the fluid frequency from the covariant 4-vector and the 4-velocity.
    Parameters:
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @Ucov: Covariant 4-velocity of the fluid in internal coordinates.
    """
    nu = - (Kcon[1] * Ucov[1] + Kcon[2] * Ucov[2] + Kcon[3] * Ucov[3] + Kcon[4] * Ucov[4]) * ME * CL * CL / HPL
    nu = abs(nu)
    return nu
end

function get_jk(X::MVec4, Kcon::MVec4, freqcgs::Float64)
    if(MODEL == "analytic")
        return get_analytic_jk(X, Kcon, freqcgs)
    elseif(MODEL == "thin_disk")
        return get_thindisk_jk(X, Kcon, freqcgs)
    else
        error("Unknown model: $MODEL")
    end
end

function approximate_solve(Ii, ji, ki, jf, kf, dl)
    """
    Evolves the intensity along the geodesic using an approximate method.

    Parameters:
    @Ii: Initial intensity at the start of the segment.
    @ji: Emissivity at the start of the segment.
    @ki: Absorption coefficient at the start of the segment.
    @jf: Emissivity at the end of the segment.
    @kf: Absorption coefficient at the end of the segment.
    @dl: Length of the segment along the geodesic.
    """


    If = 0.0
    javg = (ji + jf) / 2.
    kavg = (ki + kf) / 2.

    dtau = dl * kavg

    if (dtau < 1.e-3)
    If = Ii + (javg - Ii * kavg) * dl * (1. - (dtau / 2.) * (1. - dtau / 3.))
    else
    efac = exp(-dtau)
    If = Ii * efac + (javg / kavg) * (1. - efac)
    end

    if(isnan(If) || isinf(If))
        @error "Invalid intensity computed" If
        println("Ii = $Ii, ji = $ji, ki = $ki, jf = $jf, kf = $kf, dl = $dl")
        println("dtau = $dtau, javg = $javg, kavg = $kavg")
        println("efac = $(exp(-dtau))")
        error("Invalid intensity computed: $If")
    end

    return If
end
