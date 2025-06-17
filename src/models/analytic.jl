include("../metrics.jl")
export get_analytic_jk, radiating_region

#Model parameters (adjust spin in main.jl)
const A = 0.0
const α_analytic = -0.0
const height = (10. /3.)
const l0 = 1.0

function radiating_region(X::MVec4)
    """
    Checks if the position is within the radiating region.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, _ = bl_coord(X)
    return (r > (Rh + 0.0001) && r > 1. && r < 1000.0)
end


function get_model_4vel(X::MVec4)
    """
    Computes the 4-velocity of the model from the position vector in internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.

    Observations:
    - This follows the model described in the paper (https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6/pdf).
    """

    r,th = bl_coord(X)
    R = r * sin(th) 
    #Here, we are considering q = 0.5
    l = l0/(1 + R) * R^(1.5)
    bl_gcov::MMat4 = gcov_bl(r, th)
    bl_gcon::MMat4 = gcon_func(bl_gcov)
    gcov::MMat4 = gcov_func(X)
    gcon::MMat4 = gcon_func(gcov)
    bl_Ucov = MVec4(undef)
    # Get the normal observer velocity for Ucon/Ucov, in BL coordinates
    ubar = sqrt(-1. / (bl_gcon[1,1] - 2. * bl_gcon[1,4] * l+ bl_gcon[4,4] * l * l))
    bl_Ucov[1] = -ubar
    bl_Ucov[2] = 0.0
    bl_Ucov[3] = 0.0
    bl_Ucov[4] = l * ubar
    bl_Ucon::MVec4 = flip_index(bl_Ucov, bl_gcon)

    ks_Ucon::MVec4 = bl_to_ks(X, bl_Ucon)
    Ucon::MVec4 = vec_from_ks(X, ks_Ucon)
    Ucov::MVec4 = flip_index(Ucon, gcov)
    Bcon = zero(MVec4)
    Bcon[3] = 1.0
    Bcov::MVec4 = flip_index(Bcon, gcov)

    return Ucon, Ucov, Bcon, Bcov
end


function get_model_ne(X::MVec4)
    """
    Computes the electron number density from the position vector in internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    
    Observations:
    - This follows the model described in the paper (https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6/pdf).
    """

    r, th = bl_coord(X)
    
    n_exp::Float64 = 0.5 * ((r/10)^2 + (height * cos(th))^2)
    return (n_exp < 200) ? RHO_unit * exp(-n_exp) : 0.0
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

    if(ν <= 0. || any(isnan(ν)) || any(isinf(ν)))
        println("At X = $X\n Kcon = $Kcon")
        println("Ucov = $Ucov")
        println("Kcon $Kcon")
        error("Frequency must be positive, got ν = $ν")
    end
    # if(exp(X[2]) < 100 && exp(X[2]) > 50)
    #     println("Ne = $Ne, ν = $ν, exp(X[2]) = $(exp(X[2]))")
    # end
    jnu_inv = max(Ne * (ν/freqcgs)^(-α_analytic)/ν^2, 0.0)
    knu_inv = max((A * Ne * (ν/freqcgs)^(-(α_analytic + 2.5)) + 1.e-54) * ν, 0.0)

    if(jnu_inv > 1e-30)
        println("r = $(exp(X[2])), th = $(X[3] * π), Ne = $Ne, ν = $ν, jnu_inv = $jnu_inv, knu_inv = $knu_inv")
        error("")
    end

    if(isnan(jnu_inv) || isinf(jnu_inv))
        @error "Invalid jnu_inv computed" jnu_inv
        println("Ne = $Ne, ν = $ν")
        error("Invalid jnu_inv computed: $jnu_inv")
    end
    if(isnan(knu_inv) || isinf(knu_inv))
        @error "Invalid knu_inv computed" knu_inv
        println("Ne = $Ne, ν = $ν")
        error("Invalid knu_inv computed: $knu_inv")
    end
    
    return jnu_inv, knu_inv
end
