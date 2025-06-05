include("metrics.jl")
export get_model_4vel, get_model_ne

#Model parameters (adjust spin in main.jl)
const A = 1.e6
const α = 0.0
const height = (100. /3.)
const l0 = 1.0

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
    ubar = sqrt(-1. / (bl_gcon[1,1] - 2. * bl_gcon[1,4] * l
                    + bl_gcon[4,4] * l * l))
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