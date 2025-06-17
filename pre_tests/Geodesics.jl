using LinearAlgebra
using Plots
using Printf


mutable struct Params
    old_centering::Bool
    xoff::Float64
    yoff::Float64
    nx::Int
    ny::Int
    rotcam::Float64
    eps_ipole::Float64
    maxnstep::Int
    dsource::Float64
end

mutable struct OfTraj
    dl::Float64
    X::MVec4
    Kcon::MVec4
    Xhalf::MVec4
    Kconhalf::MVec4
end


function print_vector(name::String, vec::MVec4)
    println("Vector: $name")
    for i in eachindex(vec)
        print("$(vec[i]) ")
    end
    println()
end
function print_matrix(name::String, mat::MMat4)
    println("Matrix: $name")
    for i in axes(mat, 1)
        for j in axes(mat, 2)
            @printf("%.15e ", mat[i, j])
        end
        println()
    end
end

function gcov_func(X::MVec4)
    r::Float64 = 0;
    th::Float64 = 0;
    r, th = bl_coord(X)
    gcov = zero(MMat4)
    cth = cos(th)
    sth = abs(sin(th))
    if(sth < 1e-40)
        sth = 1e-40
    end

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

function gcon_func(gcov::Array{Float64, 2})
    return gconKS(gcov)
end

function gcovKS(r::Float64, th::Float64)
    cth::Float64 = cos(th)
    sth::Float64 = sin(th)
    s2::Float64 = sth * sth
    rho2::Float64 = r * r + a * a * cth * cth

    gcov::Array{Float64, 2} = zero(MMat4)

    gcov[1, 1] = -1. + 2. * r / rho2
    gcov[1, 2] = 2. * r / rho2
    gcov[1, 4] = -2. * a * r * s2 / rho2
    gcov[2, 1] = gcov[1, 2]
    gcov[2, 2] = 1. + 2. * r / rho2
    gcov[2, 4] = -a * s2 * (1. + 2. * r / rho2)
    gcov[3, 3] = rho2
    gcov[4, 1] = gcov[1, 4]
    gcov[4, 2] = gcov[2, 4]
    gcov[4, 4] = s2 * (rho2 + a * a * s2 * (1. + 2. * r / rho2))
    
    return gcov
end

function gconKS(gcov::Array{Float64, 2})
    gcon = inv(gcov)
    if any(isnan.(gcon)) || any(isinf.(gcon))
        @error "Singular gcov encountered in gconKS!"
        @info "gcov = $gcov"
        error("Singular gcov encountered, cannot compute gcon.")
    end
    return gcon
end

function bl_coord(X::MVec4)
    R0 = 0.0;
    r = exp(X[2]) + R0;
    th = π *X[3]
    return r, th
end

function flip_index(vector::MVec4, metric::Array{Float64,2})
    flipped_vector = zeros(Float64, NDIM)
    for ν in 1:NDIM
        for μ in 1:NDIM
            flipped_vector[ν] += metric[ν, μ] * vector[μ]
        end
    end
    return flipped_vector
end

function set_Econ_from_trial(defdir::Int, trial::MVec4)
    Econ = MVec4(undef)
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

function normalize(vcon::MVec4, Gcov::Array{Float64,2})

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

function project_out(vcona::MVec4, vconb::MVec4, Gcov::Array{Float64,2})
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
    return (i == j || i == k || i == l || j == k || j == l || k == l) ? 0 : sign((i - j) * (k - l))
end

function gdet_func(gcov::Matrix{Float64})
    F = lu(gcov)
    U = F.U

    if any(abs(U[i, i]) < 1e-14 for i in 1:size(U, 1))
        @warn "Singular matrix in gdet_func!"
        return -1.0
    end

    gdet = prod(diag(U))
    return sqrt(abs(gdet))
end


function check_handedness(Econ::Array{Float64,2}, Gcov::Array{Float64,2})

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

function make_plasma_tetrad(Ucon::MVec4, Kcon::MVec4, Bcon::MVec4, Gcov::Array{Float64, 2})

    Econ = MMat4(undef)
    Ecov = MMat4(undef)
    ones_vector = ones(MVec4)
    Econ[1,:] = set_Econ_from_trial(1, Ucon);
    Econ[2,:] = set_Econ_from_trial(4, ones_vector);
    Econ[3,:] = set_Econ_from_trial(3, Bcon);
    Econ[4,:] = set_Econ_from_trial(4, Kcon);
    Econ[1,:] = normalize(Econ[1,:], Gcov);
    Econ[4,:] = project_out(Econ[4,:], Econ[1,:], Gcov);
    Econ[4,:] = project_out(Econ[4,:], Econ[1,:], Gcov);
    Econ[4,:] = normalize(Econ[4,:], Gcov);

    Econ[3,:] = project_out(Econ[3,:], Econ[1,:], Gcov);
    Econ[3,:] = project_out(Econ[3,:], Econ[4,:], Gcov);
    Econ[3,:] = project_out(Econ[3,:], Econ[1,:], Gcov);
    Econ[3,:] = project_out(Econ[3,:], Econ[4,:], Gcov);
    Econ[3,:] = normalize(Econ[3,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[1,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[3,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[4,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[1,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[3,:], Gcov);
    Econ[2,:] = project_out(Econ[2,:], Econ[4,:], Gcov);
    Econ[2,:] = normalize(Econ[2,:], Gcov);
    oddflag::Int = 0
    flag::Int, dot_var::Float64 = check_handedness(Econ, Gcov)
    
    if (flag != 0)
        oddflag |= 0x10 
    end

    if (abs(abs(dot_var) - 1) > 1e-10 && use_eKS_internal == 0) || (abs(abs(dot_var) - 1) > 1e-7  && use_eKS_internal == 1)
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

function make_camera_tetrad(X::MVec4)

    Gcov = gcov_func(X);
    Gcon = gcon_func(Gcov);


    trial = zero(MVec4)
    trial[1] = -1.0

    Ucam::MVec4 = flip_index(trial, Gcon)
    #println("Warning! Two different definitions of Ucam in make_camera_tetrad! One from ipole Ilinois repository and one from Monika's repository.")
    Ucam[1] = 1.0
    Ucam[2] = 0.0
    Ucam[3] = 0.0
    Ucam[4] = 0.0

    trial = zero(MVec4)
    trial[1] = 1.0
    trial[2] = 1.0

    Kcon::MVec4 = flip_index(trial, Gcon)
    trial = zero(MVec4)
    trial[3] = 1.0
    sing::Int, Econ, Ecov = make_plasma_tetrad(Ucam, Kcon, trial, Gcov)
    return sing, Econ, Ecov;
    
end

function null_normalize(Kcon::MVec4, fnorm::Float64)
    Kcon_out = copy(Kcon)
    inorm::Float64 = sqrt(sum(Kcon[2:4] .^ 2))
    Kcon_out[1] = fnorm
    for k in 2:4
        Kcon_out[k] *= fnorm/inorm
    end
    return Kcon_out
end

function tetrad_to_coordinate(Econ::Array{Float64, 2}, Kcon_tetrad::MVec4)
    Kcon = MVec4(undef)
    for l in 1:4
        Kcon[l] = Econ[1, l] * Kcon_tetrad[1] + Econ[2, l] * Kcon_tetrad[2]+ Econ[3, l] * Kcon_tetrad[3] + Econ[4, l] * Kcon_tetrad[4]
    end
    return Kcon
end

function init_XK(i::Int, j::Int, Xcam::MVec4, params, fovx::Float64, fovy::Float64)

    Econ = MMat4(undef)
    Ecov = MMat4(undef)
    Kcon = MMVec4(undef)
    Kcon_tetrad =  MMVec4(undef)
    X = MMVec4(undef)


    _, Econ, Ecov = make_camera_tetrad(Xcam)
    if(i == 0 && j == 0)
        @warn("Warning! Two different definitions of Kcon in init_XK! One from ipole Ilinois repository and one from Monika's repository.")
    end
    #dxoff::Float64 = (i + 0.5 + params.xoff - 0.01) / params.nx - 0.5
    #dyoff::Float64 = (j + 0.5 + params.yoff) / params.ny - 0.5

    #Kcon_tetrad[1] = 0.0
    #Kcon_tetrad[2] = (dxoff * cos(params.rotcam) - dyoff * sin(params.rotcam)) * fovx
    #Kcon_tetrad[3] = (dxoff * sin(params.rotcam) + dyoff * cos(params.rotcam)) * fovy
    #Kcon_tetrad[4] = 1.0

    Kcon_tetrad[1] = 0.0
    Kcon_tetrad[2] = (i/(params.nx) - 0.5) * fovx
    Kcon_tetrad[3] = (j/(params.ny) - 0.5) * fovy
    Kcon_tetrad[4] = 1.0

    Kcon_tetrad = null_normalize(Kcon_tetrad, 1.0)  

    Kcon = tetrad_to_coordinate(Econ, Kcon_tetrad)
    for mu in 1:NDIM
        X[mu] = Xcam[mu]
    end
    return X, Kcon
end

function get_connection_analytic(X::MVec4)
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
    lconn = zeros(Float64, NDIM, NDIM, NDIM)

    dKcon = zeros(Float64, NDIM)
    Xh = zeros(Float64, NDIM)
    Kconh = zeros(Float64, NDIM)

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
    conn::Array{Float64,3} = zeros(Float64, NDIM, NDIM, NDIM)
    tmp = zeros(Float64, NDIM, NDIM, NDIM)
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
    dlx1::Float64 = eps / (abs(Kcon[2]) + SMALL*SMALL)
    dlx2::Float64 = eps * min(X[3], 1. - X[3]) / (abs(Kcon[3]) + SMALL*SMALL)
    dlx3::Float64 = eps / (abs(Kcon[4]) + SMALL*SMALL)

    idlx1::Float64 = 1.0 / (abs(dlx1) + SMALL*SMALL)
    idlx2::Float64 = 1.0 / (abs(dlx2) + SMALL*SMALL)
    idlx3::Float64 = 1.0 / (abs(dlx3) + SMALL*SMALL)

    dl::Float64 = 1.0 / (idlx1 + idlx2 + idlx3)
    return dl
end
function stop_backward_integration(X::MVec4, Kcon::MVec4)
    if (((X[2] > log(1.1 * Rstop)) && (Kcon[2] < 0.0)) || (X[2] < log(1.05 * Rh)))
        return 1
    end

    return 0
end

function trace_geodesic(
    Xi::MVec4, 
    Kconi::MVec4,   
    traj::Vector{OfTraj},      
    eps::Float64,              
    step_max::Int,                       
) :: Int
    
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
    while (stop_backward_integration(X, Kcon) == 0) && (nstep < step_max)
        dl = stepsize(X, Kcon, eps)

        traj[nstep].dl = dl * L_unit * HPL / (ME * CL^2)


        push_photon!(X, Kcon, -dl, Xhalf, Kconhalf)
  
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



function get_pixel(i::Int, j::Int, Xcam::MVec4, params::Params, 
                   fovx::Float64, fovy::Float64, freq::Float64)

    X = zeros(Float64, NDIM)
    Kcon = zeros(Float64, NDIM)

    X, Kcon = init_XK(i, j, Xcam, params, fovx, fovy)
    
    for mu in 1:NDIM
        Kcon[mu] *= freq
    end

    traj = Vector{OfTraj}()
    sizehint!(traj, params.maxnstep)  

    nstep = trace_geodesic(X, Kcon, traj, params.eps, params.maxnstep)
    resize!(traj, length(traj)) 

    if nstep >= params.maxnstep - 1
        @error "Max number of steps exceeded at pixel ($i, $j)"
    end

    return traj, nstep, X, Kcon
end

function theta_func(th::Float64)
    return th * π
end
function dtheta_func(th::Float64)
    return π
end

function root_find(th::Float64)
    x2a::Float64 = 0.0
    x2b::Float64 = 0.0
    X2a::Float64 = 0.0
    X2b::Float64 = 0.0


    if(th < π / 2.)
        X2a = 0.0 - SMALL
        X2b = 0.5 + SMALL
    else
        X2a = 0.5 - SMALL
        X2b = 1.0 + SMALL
    end

    tha = theta_func(x2a)
    thb = theta_func(x2b)

    for i in 1:10
        X2c = 0.5 * (X2a + X2b)
        thc = theta_func(X2c)

        if (thc - th) * (thb - th) < 0.0
            X2a = X2c
        else
            X2b = X2c
        end
    end

    tha = theta_func(X2a)
    for i in 1:2
        dthdX2 = dtheta_func(X2a)
        X2a -= (tha - th) / dthdX2
        tha = theta_func(X2a)
    end

    return X2a
end





function camera_position(cam_dist::Float64, cam_theta_angle::Float64, cam_phi_angle::Float64)
    
    X = zeros(Float64, NDIM)
    X[1] = 0.0
    X[2] = log(cam_dist)
    X[3] = root_find(cam_theta_angle / 180. * π)
    X[4] = cam_phi_angle/180 * π

    return X
end


function main()
    println("MBH = $MBH, L_unit = $L_unit")
    nx, ny = 128, 128 
    freq = 230e9 * HPL/(ME * CL * CL) 
    cam_dist, cam_theta_angle, cam_phi_angle = 240.0, 90., 0.

    Xcam = camera_position(cam_dist, cam_theta_angle, cam_phi_angle)
    p = Params(false, 0.0, 0.0, nx, ny, 0.0, 0.03, 5000, 1.69e+07 * PC)
    DX = 40.0
    DY = 40.0
    fovx = DX/cam_dist
    fovy = DY/cam_dist
    println("Camera position: Xcam = [$(Xcam[1]), $(Xcam[2]), $(Xcam[3]), $(Xcam[4])]")
    println("fovx = $fovx, fovy = $fovy")
    println("DX = $DX, DY = $DY")
    trajectories = Array{Vector{NTuple{2, Float64}}, 2}(undef, nx, ny)  

    for i in 0:(nx - 1)
        println("Processing pixel row ($i)")
        for j in 0:(ny - 1)
            traj, nstep, _, _ = get_pixel(i, j, Xcam, p, fovx, fovy, freq)

            if !isempty(traj)
            sph_coords = [bl_coord(pt.X) for pt in traj if length(pt.X) >= 4]
                trajectories[i+1, j+1] = sph_coords
            else
                trajectories[i+1, j+1] = []
            end
        end
    end

    colors = distinguishable_colors(nx * ny)
    color_index = 1
    plt = plot(xlim=(-50, 50), ylim=(-20, 260), legend=false, xlabel="x", ylabel="y")
    
    for j in 1:ny
        println("Plotting row ($j)")
        for i in 1:nx
            sph_traj = trajectories[i, j]
            if !isempty(sph_traj)
                xs, ys = Float64[], Float64[], Float64[]
                for sph in sph_traj
                    r = sph[1]
                    th = sph[2]
                    x = r * cos(th)
                    y = r * sin(th)
                    push!(xs, x)
                    push!(ys, y)
                end

                label_str = "px($i,$j)"
                plot!(plt, xs, ys, lw=1.0, color=colors[color_index], label=label_str)
                color_index += 1
            end
        end
    end

    savefig(plt, "./imgs/geodesics/geodesics.png")
    println("Saved geodesic plot to ./imgs/geodesics/geodesics.png")
end

const NDIM = 4  
const a = 0.9375
const SMALL = 1e-40
const Rout = 40.0
const Rh = 1 + sqrt(1. - a * a);
const R0 = 0
const Rstop = 40.0
const HPL = 6.6260693e-27 
const ME = 9.1093826e-28 
const GNEWT = 6.6742e-8 
const CL = 2.99792458e10 
const MSUN = 1.989e33 
const MUAS_PER_RAD = 2.06265e11
const MBH = 4.5e6
const L_unit = GNEWT * MBH * MSUN / (CL * CL);
const PC = 3.085678e18 
main()
GC.gc()
