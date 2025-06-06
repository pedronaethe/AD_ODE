export get_analytic_jk

function get_fluid_nu(Kcon::MVec4, Ucov::MVec4)
    """
    Computes the fluid frequency from the covariant 4-vector and the 4-velocity.
    Parameters:
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @Ucov: Covariant 4-velocity of the fluid in internal coordinates.
    """
    nu = - (Kcon[1] * Ucov[1] + Kcon[2] * Ucov[2] + Kcon[3] * Ucov[3] + Kcon[4] * Ucov[4]) * ME * CL * CL / HPL
    return nu
end




function get_analytic_jk(X, Kcon, freqcgs::Float64)
    """
    Computes the emissivity and absorption coefficient for the model at a given position and frequency.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @freqcgs: pivotal frequency in cgs units.
    """
    Ne = get_model_ne(X)
    
    if (Ne <= 0.)
        return 0, 0
    end

    Ucon, Ucov, Bcon, Bcov = get_model_4vel(X)
    ν::Float64 = get_fluid_nu(Kcon, Ucov)
    if(ν <= 0.)
        println("At X = $X, Kcon = $Kcon")
        error("Frequency must be positive, got ν = $ν")
    end
    jnu_inv = max(Ne * (ν/freqcgs)^(-α)/ν^2, 0.0)
    knu_inv = max((A * Ne * (ν/freqcgs)^(-(α + 2.5)) + 1.e-54) * ν, 0.0)
    return jnu_inv, knu_inv

end

function radiating_region(X::MVec4)
    """
    Checks if the position is within the radiating region.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, _ = bl_coord(X)
    return (r > Rh + 0.0001 && r > 1. && r < 1000.0)
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

    return If
end
