
export set_Econ_from_trial, normalize_vector

function set_Econ_from_trial(defdir::Int, trial::Vector{Float64})
    """
    Copy the trial vector, however, if the norm of the trial vector is small, set it to default value.

    Parameters:
    @defdir: Direction to define the tetrad vector.
    @trial: Trial vector to set the tetrad vector.
    """
    Econ::Vector{Float64} = zeros(Float64, 4)
    norm = sum(abs.(trial[2:4])) 
    for k in 1:4
        if norm <= SMALL
            Econ[k] = (k == defdir) ? 1.0 : 0.0
        else
            Econ[k] = trial[k]
        end
    end
    return Econ
end

function normalize_vector(vcon::Vector{Float64}, Gcov::Array{Float64,2})
    """
    Forcing the vector to |v.v| = 1.

    Parameters:
    @vcon: Vector to be normalized.
    @Gcov: Covariant metric tensor used for normalization.
    """

    vcon_out = copy(vcon)

    norm = 0.0
    for k in 1:4
        for l in 1:4
            norm += vcon[k] * vcon[l] * Gcov[k, l]
        end
    end

    norm = sqrt(abs(norm))
    for k in 1:4
        vcon_out[k] /= norm
    end
    return vcon_out
end

function project_out(vcona::Vector{Float64}, vconb::Vector{Float64}, Gcov::Array{Float64,2})
    """
    Projects out the component of vcona along vconb using the metric tensor. Output is orthogonal to vconb.

    Parameters:
    @vcona: Vector to be projected.
    @vconb: Vector to project out.
    @Gcov: Covariant metric tensor used for projection.
    """
    vconb_sq = 0.0
    for k in 1:4
        for l in 1:4
            vconb_sq += vconb[k] * vconb[l] * Gcov[k, l]
        end
    end

    adotb = 0.0
    for k in 1:4
        for l in 1:4
            adotb += vcona[k] * vconb[l] * Gcov[k, l]
        end
    end
    vcona_out = copy(vcona)
    for k in 1:4
        vcona_out[k] -= vconb[k] * adotb / vconb_sq
    end
    return vcona_out
end

function levi_civita(i::Int, j::Int, k::Int, l::Int)
    """
    Returns the Levi-Civita symbol for the indices i, j, k, l.

    Parameters:
    @i, @j, @k, @l: Indices for which the Levi-Civita symbol is computed.
    """
    return (i == j || i == k || i == l || j == k || j == l || k == l) ? 0 : sign((i - j) * (k - l))
end



function check_handedness(Econ::Array{Float64,2}, Gcov::Array{Float64,2})
    """
    This will check the handness of the tetrad basis. +1 if right-handed, -1 if left-handed.

    Parameters:
    Econ: Tetrad basis vectors in covariant form.
    Gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """

    g = gdet_func(Gcov)
    if g < 0.0
        @warn "Encountered singular gcov checking handedness!"
        return (1, 0.0)
    end
    dot_var::Float64 = 0.0
    for i in 1:4, j in 1:4, l in 1:4, k in 1:4
        dot_var += g * levi_civita(i-1, j-1, k-1, l-1) * Econ[1, i] * Econ[2, j] * Econ[3, k] * Econ[4, l]
    end

    return (0, dot_var)
end