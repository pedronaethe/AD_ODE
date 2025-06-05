
export init_XK
function init_XK(i::Int, j::Int, Xcam::MVec4, params, fovx::Float64, fovy::Float64)
    """
    Initializes a geodesic from the camera

    Parameters:
    @i: x-index of the pixel in the image plane.
    @j: y-index of the pixel in the image plane.
    @Xcam: Position vector of the camera in internal coordinates.
    @params: Parameters for the camera.
    @fovx: Field of view in the x-direction.
    @fovy: Field of view in the y-direction.
    """

    Econ = MMat4(undef)
    Ecov = MMat4(undef)
    Kcon = MVec4(undef)
    Kcon_tetrad = MVec4(undef)
    X = MVec4(undef)


    _, Econ, Ecov = make_camera_tetrad(Xcam)
    if(i == 0 && j == 0)
        @warn("Warning! Two different definitions of Kcon in init_XK! One from ipole Ilinois repository and one from Monika's repository.")
        @warn("Using Ipole's definition")
    end
    dxoff::Float64 = (i + 0.5 + params.xoff - 0.01) / params.nx - 0.5
    dyoff::Float64 = (j + 0.5 + params.yoff) / params.ny - 0.5

    Kcon_tetrad[1] = 0.0
    Kcon_tetrad[2] = (dxoff * cos(params.rotcam) - dyoff * sin(params.rotcam)) * fovx
    Kcon_tetrad[3] = (dxoff * sin(params.rotcam) + dyoff * cos(params.rotcam)) * fovy
    Kcon_tetrad[4] = 1.0
    # Kcon_tetrad[1] = 0.0
    # Kcon_tetrad[2] = (i/(params.nx) - 0.5) * fovx
    # Kcon_tetrad[3] = (j/(params.ny) - 0.5) * fovy
    # Kcon_tetrad[4] = 1.0

    Kcon_tetrad = null_normalize(Kcon_tetrad, 1.0)  

    Kcon = tetrad_to_coordinate(Econ, Kcon_tetrad)
    for mu in 1:NDIM
        X[mu] = Xcam[mu]
    end
    return X, Kcon
end

function get_connection_analytic(X::MVec4)
    """
    Returns the analytical connection coefficients in Kerr-Schild coordinates.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    lconn = zeros(4, 4, 4)
    
    r1 = exp(X[2]) 
    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r3 * r1

    th = π * X[3]
    dthdx2 = π
    d2thdx22 = 0.0

    dthdx22 = dthdx2 * dthdx2

    sth = sin(th)
    cth = cos(th)
    sth2 = sth * sth
    r1sth2 = r1 * sth2
    sth4 = sth2 * sth2
    cth2 = cth * cth
    cth4 = cth2 * cth2
    s2th = 2.0 * sth * cth
    c2th = 2 * cth2 - 1.0

    a2 = a * a
    a2sth2 = a2 * sth2
    a2cth2 = a2 * cth2
    a3 = a2 * a
    a4 = a3 * a
    a4cth4 = a4 * cth4

    rho2 = r2 + a2cth2
    rho22 = rho2 * rho2
    rho23 = rho22 * rho2
    irho2 = 1.0 / rho2
    irho22 = irho2 * irho2
    irho23 = irho22 * irho2
    irho23_dthdx2 = irho23 / dthdx2

    fac1 = r2 - a2cth2
    fac1_rho23 = fac1 * irho23
    fac2 = a2 + 2 * r2 + a2 * c2th
    fac3 = a2 + r1 * (-2.0 + r1)

    lconn[1, 1, 1] = 2.0 * r1 * fac1_rho23
    lconn[1, 1, 2] = r1 * (2.0 * r1 + rho2) * fac1_rho23
    lconn[1, 1, 3] = -a2 * r1 * s2th * dthdx2 * irho22
    lconn[1, 1, 4] = -2.0 * a * r1sth2 * fac1_rho23

    lconn[1, 2, 1] = lconn[1, 1, 2]
    lconn[1, 2, 2] = 2.0 * r2 * (r4 + r1 * fac1 - a4cth4) * irho23
    lconn[1, 2, 3] = -a2 * r2 * s2th * dthdx2 * irho22
    lconn[1, 2, 4] = a * r1 * (-r1 * (r3 + 2 * fac1) + a4cth4) * sth2 * irho23

    lconn[1, 3, 1] = lconn[1, 1, 3]
    lconn[1, 3, 2] = lconn[1, 2, 3]
    lconn[1, 3, 3] = -2.0 * r2 * dthdx22 * irho2
    lconn[1, 3, 4] = a3 * r1sth2 * s2th * dthdx2 * irho22

    lconn[1, 4, 1] = lconn[1, 1, 4]
    lconn[1, 4, 2] = lconn[1, 2, 4]
    lconn[1, 4, 3] = lconn[1, 3, 4]
    lconn[1, 4, 4] = 2.0 * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23

    lconn[2, 1, 1] = fac3 * fac1 / (r1 * rho23)
    lconn[2, 1, 2] = fac1 * (-2.0 * r1 + a2sth2) * irho23
    lconn[2, 1, 3] = 0.0
    lconn[2, 1, 4] = -a * sth2 * fac3 * fac1 / (r1 * rho23)

    lconn[2, 2, 1] = lconn[2, 1, 2]
    lconn[2, 2, 2] = (r4 * (-2.0 + r1) * (1.0 + r1) + a2 * (a2 * r1 * (1.0 + 3.0 * r1) * cth4 + a4 * cth4 * cth2 + r3 * sth2 + r1 * cth2 * (2.0 * r1 + 3.0 * r3 - a2sth2))) * irho23
    lconn[2, 2, 3] = -a2 * dthdx2 * s2th / fac2
    lconn[2, 2, 4] = a * sth2 * (a4 * r1 * cth4 + r2 * (2 * r1 + r3 - a2sth2) + a2cth2 * (2.0 * r1 * (-1.0 + r2) + a2sth2)) * irho23

    lconn[2, 3, 1] = lconn[2, 1, 3]
    lconn[2, 3, 2] = lconn[2, 2, 3]
    lconn[2, 3, 3] = -fac3 * dthdx22 * irho2
    lconn[2, 3, 4] = 0.0

    lconn[2, 4, 1] = lconn[2, 1, 4]
    lconn[2, 4, 2] = lconn[2, 2, 4]
    lconn[2, 4, 3] = lconn[2, 3, 4]
    lconn[2, 4, 4] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23)

    lconn[3, 1, 1] = -a2 * r1 * s2th * irho23_dthdx2
    lconn[3, 1, 2] = r1 * lconn[3, 1, 1]
    lconn[3, 1, 3] = 0.0
    lconn[3, 1, 4] = a * r1 * (a2 + r2) * s2th * irho23_dthdx2

    lconn[3, 2, 1] = lconn[3, 1, 2]
    lconn[3, 2, 2] = r2 * lconn[3, 1, 1]
    lconn[3, 2, 3] = r2 * irho2
    lconn[3, 2, 4] = (a * r1 * cth * sth * (r3 * (2.0 + r1) + a2 * (2.0 * r1 * (1.0 + r1) * cth2 + a2 * cth4 + 2 * r1sth2))) * irho23_dthdx2

    lconn[3, 3, 1] = lconn[3, 1, 3]
    lconn[3, 3, 2] = lconn[3, 2, 3]
    lconn[3, 3, 3] = -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2
    lconn[3, 3, 4] = 0.0

    lconn[3, 4, 1] = lconn[3, 1, 4]
    lconn[3, 4, 2] = lconn[3, 2, 4]
    lconn[3, 4, 3] = lconn[3, 3, 4]
    lconn[3, 4, 4] = -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4.0 + r1) + a2cth2) + 2.0 * r1 * a4 * sth4) * irho23_dthdx2

    lconn[4, 1, 1] = a * fac1_rho23
    lconn[4, 1, 2] = r1 * lconn[4, 1, 1]
    lconn[4, 1, 3] = -2.0 * a * r1 * cth * dthdx2 / (sth * rho22)
    lconn[4, 1, 4] = -a2sth2 * fac1_rho23

    lconn[4, 2, 1] = lconn[4, 1, 2]
    lconn[4, 2, 2] = a * r2 * fac1_rho23
    lconn[4, 2, 3] = -2 * a * r1 * (a2 + 2 * r1 * (2.0 + r1) + a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2)
    lconn[4, 2, 4] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23

    lconn[4, 3, 1] = lconn[4, 1, 3]
    lconn[4, 3, 2] = lconn[4, 2, 3]
    lconn[4, 3, 3] = -a * r1 * dthdx22 * irho2
    lconn[4, 3, 4] = dthdx2 * (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22

    lconn[4, 4, 1] = lconn[4, 1, 4]
    lconn[4, 4, 2] = lconn[4, 2, 4]
    lconn[4, 4, 3] = lconn[4, 3, 4]
    lconn[4, 4, 4] = (-a * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23

    return lconn
end



function push_photon!(X::MVec4, Kcon::MVec4, dl::Float64, Xhalf::MVec4, Kconhalf::MVec4)
    """
    Pushes the photon geodesic forward/backwards by a step size dl/-dl using the analytic connection coefficients.
    Parameters:
    @X: Position vector of the photon in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @dl: Step size for the geodesic integration.
    @Xhalf: Position vector of the photon at the half-step.
    @Kconhalf: Covariant 4-vector of the photon at the half-step.
    """

    lconn = Tensor3D(undef)

    dKcon = MVec4(undef)
    Xh = MVec4(undef)
    Kconh = MVec4(undef)

    lconn = get_connection_analytic(X)

    for k in 1:NDIM
        for i in 1:NDIM
            for j in 1:NDIM
                dKcon[k] -= 0.5 * dl * lconn[k, i, j] * Kcon[i] * Kcon[j]
            end
        end
    end
    for k in 1:NDIM
        Kconh[k] = Kcon[k] + dKcon[k]
    end
        

    for i in 1:NDIM
        Xh[i] = X[i] + 0.5 * dl * Kcon[i]
    end

    for i in 1:NDIM
        Xhalf[i] = Xh[i]
        Kconhalf[i] = Kconh[i]
    end

    lconn = get_connection_analytic(Xh)

    fill!(dKcon, 0.0)

    for k in 1:NDIM
        for i in 1:NDIM
            for j in 1:NDIM
                dKcon[k] -= dl * lconn[k, i, j] * Kconh[i] * Kconh[j]
            end
        end
    end

    for k in 1:NDIM
        Kcon[k] += dKcon[k]
    end

    for k in 1:NDIM
        X[k] += dl * Kconh[k]
    end
end

const DEL = 1.e-7
function get_connection(X::MVec4)
    """
    Returns the connection coefficients in Kerr-Schild coordinates using finite differences.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    conn = Tensor3D(undef)
    tmp = Tensor3D(undef)
    Xh = copy(X)
    Xl = copy(X)
    gcon = MMat4(undef)
    gcov = MMat4(undef)
    gh = MMat4(undef)
    gl = MMat4(undef)

    gcov = gcov_func(X)
    gcon = gcon_func(gcov)

    for k in 1:NDIM
        Xh .= X
        Xl .= X
        Xh[k] += DEL
        Xl[k] -= DEL

        gh = gcov_func(Xh)
        gl = gcov_func(Xl)



        for i in 1:NDIM
            for j in 1:NDIM
                conn[i, j, k] = (gh[i, j] - gl[i, j]) / (Xh[k] - Xl[k])
            end
        end
    end


    for i in 1:NDIM
        for j in 1:NDIM
            for k in 1:NDIM
                tmp[i, j, k] = 0.5 * (conn[j, i, k] + conn[k, i, j] - conn[k, j, i])
            end
        end
    end

    for i in 1:NDIM
        for j in 1:NDIM
            for k in 1:NDIM
                conn[i, j, k] = 0.0
                for l in 1:NDIM
                    conn[i, j, k] += gcon[i, l] * tmp[l, j, k]
                end
            end
        end
    end

    return conn
end


function stepsize(X::MVec4, Kcon::MVec4, eps::Float64)
    """
    Computes the step size for the geodesic integration based on the position and covariant 4-vector.
    Parameters:
    @X: Position vector of the photon in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @eps: Small constant for controlling the step size.
    """
    dlx2::Float64 = 0.0
    dlx3::Float64 = 0.0
    dlx4::Float64 = 0.0
    idlx2::Float64 = 0.0
    idlx3::Float64 = 0.0
    idlx4::Float64 = 0.0
    dl::Float64 = 0.0

    
    if(true)
        deh::Float64 = min(abs(X[2] - cstartx[2]), 0.1)
        dlx2 = eps * (10 * deh) / (abs(Kcon[2]) + SMALL*SMALL)
        cut::Float64 = 0.02
        lx3::Float64 = cstopx[3] - cstartx[3]
        dpole::Float64 = min(abs(X[3] / lx3), abs((cstopx[3] - X[3]) / lx3))
        d2fac::Float64 = (dpole < cut) ? dpole / 3 : min(cut / 3 + (dpole - cut) * 10., 1)
        dlx3 = eps * d2fac / (abs(Kcon[3]) + SMALL*SMALL)

        dlx4 = eps / (abs(Kcon[4]) + SMALL*SMALL)
        idlx2 = 1.0 / (abs(dlx2) + SMALL*SMALL)
        idlx3 = 1.0 / (abs(dlx3) + SMALL*SMALL)
        idlx4 = 1.0 / (abs(dlx4) + SMALL*SMALL)

        dl = 1.0 / (idlx2 + idlx3 + idlx4)
    else
        dlx2 = eps / (abs(Kcon[2]) + SMALL*SMALL)
        dlx3 = eps * min(X[3], 1. - X[3]) / (abs(Kcon[3]) + SMALL*SMALL)
        dlx4 = eps / (abs(Kcon[4]) + SMALL*SMALL)

        idlx2 = 1.0 / (abs(dlx2) + SMALL*SMALL)
        idlx3 = 1.0 / (abs(dlx3) + SMALL*SMALL)
        idlx4 = 1.0 / (abs(dlx4) + SMALL*SMALL)

        dl = 1.0 / (idlx2 + idlx3 + idlx4)
    end

    return dl
end


function stop_backward_integration(X::MVec4, Kcon::MVec4)
    """
    Checks if the backward integration should stop based on the position and covariant 4-vector.
    Parameters:
    @X: Position vector of the photon in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    """
    if (((X[2] > log(Rstop)) && (Kcon[2] < 0.0)) || (X[2] < log(Rh+ 0.0001)))
        return 1
    end

    return 0
end

function trace_geodesic(Xi::MVec4, Kconi::MVec4, traj::Vector{OfTraj}, eps::Float64, step_max::Int, i::Int, j::Int)
    """
    Function loops through the geodesic integration steps, pushing the photon along the geodesic.
    Parameters:
    @Xi: Initial position vector of the photon in internal coordinates.
    @Kconi: Initial covariant 4-vector of the photon in internal coordinates.
    @traj: Structure to store the trajectory of the photon.
    @eps: Small constant for controlling the step size.
    @step_max: Maximum number of steps for the geodesic integration.
    @i: x-index of the pixel in the image plane (debugging purposes).
    @j: y-index of the pixel in the image plane (debugging purposes).
    """
    
    X = copy(Xi)
    Kcon = copy(Kconi)
    Xhalf = copy(Xi)
    Kconhalf = copy(Kconi)

    push!(traj, OfTraj(
        0,
        copy(Xi),   
        copy(Kconi),   
        copy(Xi),   
        copy(Kconi)    
    ))
    
    nstep = 1
    # @printf("X goin in trace_geodesic: %.15e, %.15e, %.15e, %.15e\n", X[1], X[2], X[3], X[4])
    # @printf("Kcon going in trace_geodesic: %.15e, %.15e, %.15e, %.15e\n", Kcon[1], Kcon[2], Kcon[3], Kcon[4])
    while (stop_backward_integration(X, Kcon) == 0) && (nstep < step_max)
        dl = stepsize(X, Kcon, eps)
        # @printf("Step %d: dl = %.15e\n", nstep, dl)

        traj[nstep].dl = dl * L_unit * HPL / (ME * CL^2)


        push_photon!(X, Kcon, -dl, Xhalf, Kconhalf)
        # if(i == 0 && j == 1)
        #     @printf("At step = %d\n", nstep)
        #     @printf("Radius = %.15e\n", exp(X[2]))
        #     @printf("X after push: %.15e, %.15e, %.15e, %.15e\n", X[1], X[2], X[3], X[4])
        #     @printf("Kcon after push: %.15e, %.15e, %.15e, %.15e\n", Kcon[1], Kcon[2], Kcon[3], Kcon[4])
        #     @printf("dl = %.15e, traj.dl = %.15e\n\n", dl, traj[nstep].dl)
        # end
        nstep += 1
        push!(traj, OfTraj(
            copy(dl),
            copy(X),   
            copy(Kcon),   
            copy(Xhalf),   
            copy(Kconhalf)    
        ))
    end
    pop!(traj)
    nstep -= 1



    return nstep
end

