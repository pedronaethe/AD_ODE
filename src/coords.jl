
export theta_func


function theta_func(X::Vector{Float64})
    """
    Computes the theta coordinate from the internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    _, th = bl_coord(X)
    return th
end




function bl_to_ks(X::Vector{Float64}, ucon_bl::Vector{Float64})
    """
    Converts the 4-velocity from Boyer-Lindquist coordinates to Kerr-Schild coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @ucon_bl: Covariant 4-velocity in Boyer-Lindquist coordinates.
    """
    ucon_ks = zeros(Float64, NDIM)
    r, th = bl_coord(X)
    trans = zeros(Float64, NDIM, NDIM)
    for μ in 1:NDIM
        for ν in 1:NDIM
            trans[μ, ν] = if μ == ν 1.0 else 0.0 end
        end
    end

    trans[1, 2] = 2.0 * r / (r * r - 2.0 * r + a * a)
    trans[4, 2] = a / (r * r - 2.0 * r + a * a)
    for μ in 1:NDIM
        for ν in 1:NDIM
            ucon_ks[μ] += trans[μ, ν] * ucon_bl[ν]
        end
    end

    return ucon_ks
end


function set_dxdX(X::Vector{Float64})
    """
    Computes the Jacobian matrix dxdX for the transformation from Kerr-Schild coordinates to internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    dxdX = zeros(Float64, NDIM, NDIM)
    hslope = 0.0   
    for mu in 1:NDIM
        for nu in 1:NDIM
            if mu == nu
                dxdX[mu, nu] = 1.0
            else
                dxdX[mu, nu] = 0.0
            end
        end
    end

    dxdX[2,2] = exp(X[2])
    dxdX[3,3] = π + (1 - hslope) * π * cos(2 * π * X[3])
    if(dxdX[3,3] <= 0.0)
        println("Warning! dxdX[3,3] is non-positive: ", dxdX[3,3])
        println("X[3] = ", X[3])
        dxdX[3,3] = 1.0e-10  # Set a small positive value to avoid issues
    end

    return dxdX
end

function set_dXdx(X::Vector{Float64})
    """
    Computes the inverse Jacobian matrix dXdx for the transformation from internal coordinates to Kerr-Schild coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    dxdX = zeros(Float64, NDIM, NDIM)
    dxdX = set_dxdX(X)
    #invert matrix to find dXdx from dxdX using linear algebra package
    dXdx = inv(dxdX)

    return dXdx
end


function vec_from_ks(X::Vector{Float64}, v_ks::Vector{Float64})
    """
    Converts a 4-vector from Kerr-Schild coordinates to the natural coordinate system.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @v_ks: 4-vector in Kerr-Schild coordinates.
    """
    v_nat = zeros(Float64, NDIM)
    dXdx = zeros(Float64, NDIM, NDIM)
    dXdx = set_dXdx(X)

    for μ in 1:NDIM
        for ν in 1:NDIM
            v_nat[μ] += dXdx[μ, ν] * v_ks[ν]
        end
    end

    return v_nat
end


function bl_coord(X::Vector{Float64})
    """
    Returns Boyer-Lindquist coordinates (r, th) from internal coordinates (X[2], X[3]).
    Parameters:
    @X: Vector of position coordinates in internal coordinates coordinates.
    """
    r = exp(X[2]) + R0;
    th = π *X[3]
    return r, th
end

function flip_index(vector::Vector{Float64}, metric::Array{Float64,2})
    """
    Returns the flipped index of a vector using the metric tensor.
    
    Parameters:
    @vector: Vector to be flipped.
    @metric: Metric tensor used for flipping.
    """
    flipped_vector = zeros(Float64, NDIM)
    for ν in 1:NDIM
        for μ in 1:NDIM
            flipped_vector[ν] += metric[ν, μ] * vector[μ]
            if (ν == 4)
            end
        end
    end
    return flipped_vector
end