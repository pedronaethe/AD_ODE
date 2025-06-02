export camera_position

include("../src/metrics.jl")

function root_find(x::Vector{Float64})
    """
    Finds the root of the theta function using a bisection method.
    Parameters:
    @x: Vector of position coordinates in internal coordinates.
    """
    th = x[3]

    xa = zeros(Float64, NDIM)
    xb = zeros(Float64, NDIM)
    xc = zeros(Float64, NDIM)

    xa[2] = log(x[2])
    xa[4] = x[4]
    xb[2] = xa[2]
    xb[4] = xa[4]
    xc[2] = xa[2]
    xc[4] = xa[4]
        
    if x[3] < π / 2.
        xa[3] = cstartx[3]
        xb[3] = (cstopx[3] - cstartx[3]) / 2 + SMALL
    else
        xa[3] = (cstopx[3] - cstartx[3]) / 2 - SMALL
        xb[3] = cstopx[3]
    end

    tol:: Float64 = 1.e-9
    tha = theta_func(xa)
    thb = theta_func(xb)

    if(abs(tha - th) < tol)
        return xa[3]
    elseif(abs(thb - th) < tol)
        return xb[3]
    end

    for i in 1:1000
        xc[3] = 0.5 * (xa[3] + xb[3])
        thc = theta_func(xc)

        if((thc - th) * (thb - th) < 0.)
            xa[3] = xc[3]
        else
            xb[3] = xc[3]
        end

        err = thc - th
        if(abs(err) < tol)
            break
        end
    end

  return xc[3];
end

function camera_position(cam_dist::Float64, cam_theta_angle::Float64, cam_phi_angle::Float64)
    """
    Computes the camera position in internal coordinates based on the distance and angles.
    Parameters:
    @cam_dist: Radial distance of the camera.
    @cam_theta_angle: Polar angle of the camera in degrees.
    @cam_phi_angle: Azimuthal angle of the camera in degrees.
    """
    
    X = zeros(Float64, NDIM)
    x = [0.0, cam_dist, cam_theta_angle/180 * π, cam_phi_angle/180 * π]
    X[1] = 0.0
    X[2] = log(cam_dist)
    X[3] = root_find(x)
    X[4] = cam_phi_angle/180 * π
    return X
end

