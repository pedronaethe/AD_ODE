include("../src/metrics.jl")

export make_camera_tetrad



function make_camera_tetrad(X::MVec4)
    """
    Returns a camera tetrad based on the input position vector X.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.

    Observations:
    The tetrad is constructed such that:
    - e^0 is aligned with the camera's Ucam.
    - e^3 is aligned with the radial direction (outward).
    - e^2 is aligned with the north pole of the coordinate system ("y" in the image plane).
    - e^1 is the remaining direction ("x" in the image plane).

    Points the camera such that the angular momentum at FOV center is 0.
    """

    Gcov = gcov_func(X);
    Gcon = gcon_func(Gcov);


    trial= zero(MVec4)
    trial[1] = -1.0

    Ucam::MVec4 = flip_index(trial, Gcon)
    #@warn("Warning! Two different definitions of Ucam in make_camera_tetrad! One from ipole Ilinois repository and one from Monika's repository.")
    # Ucam[1] = 1.0
    # Ucam[2] = 0.0
    # Ucam[3] = 0.0
    # Ucam[4] = 0.0

    trial = zero(MVec4)
    trial[1] = 1.0
    trial[2] = 1.0

    Kcon::MVec4 = flip_index(trial, Gcon)
    trial = zero(MVec4)
    trial[3] = 1.0
    sing::Int, Econ, Ecov = make_plasma_tetrad(Ucam, Kcon, trial, Gcov)
    return sing, Econ, Ecov;
    
end


function make_plasma_tetrad(Ucon::MVec4, Kcon::MVec4, Bcon::MVec4, Gcov::MMat4)
    """
    Returns a plasma tetrad based on the input vectors Ucon, Kcon, and Bcon.

    Parameters:
    @Ucon: Covariant 4-velocity vector of the plasma.
    @Kcon: Covariant 4-vector in the direction of the camera.
    @Bcon: Covariant 4-vector in the direction of the magnetic field.
    @Gcov: Covariant metric tensor in Kerr-Schild coordinates.


    Observations:
    Econ[k][l]
    - k: index attached to tetrad basis, index down
    - l: index attached to coordinate basis, index up
    
    Ecov switches both indices
    e^0 along U
    e^2 along b
    e^3 along spatial part of K
    """
    Econ = MMat4(undef)
    Ecov = MMat4(undef)
    ones_vector = ones(MVec4)
    Econ[1,:] = set_Econ_from_trial(1, Ucon);
    Econ[2,:] = set_Econ_from_trial(4, ones_vector);
    Econ[3,:] = set_Econ_from_trial(3, Bcon);
    Econ[4,:] = set_Econ_from_trial(4, Kcon);
    Econ[1,:] = normalize_vector(Econ[1,:], Gcov);
    Econ[4,:] = project_out(Econ[4,:], Econ[1,:], Gcov);
    Econ[4,:] = project_out(Econ[4,:], Econ[1,:], Gcov);
    Econ[4,:] = normalize_vector(Econ[4,:], Gcov);

    Econ[3,:] = project_out(Econ[3,:], Econ[1,:], Gcov);
    Econ[3,:] = project_out(Econ[3,:], Econ[4,:], Gcov);
    Econ[3,:] = project_out(Econ[3,:], Econ[1,:], Gcov);
    Econ[3,:] = project_out(Econ[3,:], Econ[4,:], Gcov);
    Econ[3,:] = normalize_vector(Econ[3,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[1,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[3,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[4,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[1,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[3,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[4,:], Gcov);
    Econ[2,:] = normalize_vector(Econ[2,:], Gcov);
    oddflag::Int = 0
    flag::Int, dot_var::Float64 = check_handedness(Econ, Gcov)
    
    if (flag != 0)
        oddflag |= 0x10 
    end

    if (abs(abs(dot_var) - 1) > 1e-10)
        oddflag |= 0x1
    end

    if dot_var < 0
        for k in 1:4
            Econ[2, k] *= -1
        end
    end

    for k in 1:4
        Ecov[k, :] = flip_index(Econ[k, :], Gcov)
    end

    for l in 1:4
        Ecov[1, l] *= -1 
    end

    return oddflag, Econ, Ecov
end


function null_normalize(Kcon::MVec4, fnorm::Float64)
    """
    Normalizes null vector in a tetrad frame.

    Parameters:
    @Kcon: Covariant 4-vector in the tetrad frame.
    @fnorm: Desired norm of the vector.
    """
    Kcon_out = copy(Kcon)
    inorm::Float64 = sqrt(sum(Kcon[2:4] .^ 2))
    Kcon_out[1] = fnorm
    for k in 2:4
        Kcon_out[k] *= fnorm/inorm
    end
    return Kcon_out
end

function tetrad_to_coordinate(Econ::MMat4, Kcon_tetrad::MVec4)
    """
    Returns the covariant 4-vector in the coordinate frame from the tetrad frame.

    Parameters:
    @Econ: Tetrad basis vectors in covariant form.
    @Kcon_tetrad: Covariant 4-vector in the tetrad frame.
    """


    Kcon::MVec4 = MVec4(undef)
    for l in 1:4
        Kcon[l] = Econ[1, l] * Kcon_tetrad[1] + Econ[2, l] * Kcon_tetrad[2]+ Econ[3, l] * Kcon_tetrad[3] + Econ[4, l] * Kcon_tetrad[4]
    end
    return Kcon
end