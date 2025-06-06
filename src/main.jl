using Printf
using Base.Threads
using LinearAlgebra
using StaticArrays
using Krang
#Muttable 4-dimensional vector allocated on the stack
const MVec4  = MVector{4,Float64}
#Immutable 4-dimensional vector array allocated on the stack
const Mat4  = SMatrix{4,4,Float64} 
#Mutable 4x4 matrix allocated on the stack
const MMat4 = MMatrix{4,4,Float64}
#Mutable 3-dimensional vector allocated on the stack
const Tensor3D = MArray{Tuple{4,4,4}, Float64, 3, 64}  # 4×4×4 mutable tensor


mutable struct Params
    xoff::Float64
    yoff::Float64
    nx::Int
    ny::Int
    rotcam::Float64
    eps::Float64
    maxnstep::Int
end

mutable struct OfTraj
    dl::Float64
    X::MVec4
    Kcon::MVec4
    Xhalf::MVec4
    Kconhalf::MVec4
end

include("camera.jl")
include("debug_functions.jl")
include("metrics.jl")
include("coords.jl")
include("tetrads.jl")
include("utils.jl")
include("radiation.jl")
include("analytic.jl")
include("geodesics.jl")
include("constants.jl")


function get_pixel(i::Int, j::Int, Xcam::MVec4, params::Params, fovx::Float64, fovy::Float64, freq::Float64)
    """
    Evolves the geodesic and integrate emissivity along the geodesic for each pixel.
    Parameters:
    @i: x-index of the pixel in the image plane.
    @j: y-index of the pixel in the image plane.
    @Xcam: Position vector of the camera in internal coordinates.
    @params: Parameters for the camera.
    @fovx: Field of view in the x-direction.
    @fovy: Field of view in the y-direction.
    @freq: Frequency of the radiation.
    """
    X = MVec4(undef)
    Kcon = MVec4(undef)

    X, Kcon = init_XK(i, j, Xcam, params, fovx, fovy)

    for mu in 1:NDIM
        Kcon[mu] *= freq
    end
    traj = Vector{OfTraj}()
    sizehint!(traj, params.maxnstep)  
    nstep = trace_geodesic(X, Kcon, traj, params.eps, params.maxnstep, i, j)
    resize!(traj, length(traj)) 

    # if(i == 0 && j == 0)
    #     for i in 1:nstep
    #         r, th = bl_coord(traj[i].X)
    #         @printf("Step %d: R = %.15e, Theta = %.15e\n", i, r, th)
    #         @printf("Kcon: [%.15e, %.15e, %.15e, %.15e]\n", traj[i].Kcon[1], traj[i].Kcon[2], traj[i].Kcon[3], traj[i].Kcon[4])
    #     end
    # end
    if nstep >= params.maxnstep - 1
        @error "Max number of steps exceeded at pixel ($i, $j)"
    end

    return traj, nstep
end


function calcKcon(sr::Float64, sθ::Float64, r::Float64, θ::Float64, ϕ::Float64, metric::Kerr, αmin::Float64, αmax::Float64, βmin::Float64, βmax::Float64, θo::Float64, res::Int, I, J)
    """
    Calculates the covariant 4-velocity of the photon in internal coordinates.
    
    Observations:
    - We follow the equation 9 of Gelles et al.,2021 https://arxiv.org/pdf/2105.09440
    - The function Krang.p_bl_d returns the wavevector in BL coordinates, so we transform for the natural coordinates.
    """
    Kcovbl = MVec4(undef)
    α = αmin + (αmax - αmin) * (I - 1) / (res - 1)
    β = βmin + (βmax - βmin) * (J - 1) / (res - 1)
    η = Krang.η(metric, α, β, θo)
    λ = Krang.λ(metric, α, θo)
    X = MVec4(-1, log(r), θ/π, ϕ)
    Kcovbl = Krang.p_bl_d(metric, r, θ, η, λ, true, true)
    KcovKS = bl_to_ks(X, MVec4(Kcovbl))
    KcovNC = vec_from_ks(X, KcovKS)
    gcov = gcov_func(X)
    gcon = gcon_func(gcov)
    KconNC = flip_index(KcovNC, gcon)

    return KconNC
end

function main()
    @time begin
        println()
        println("Model Parameters: A = $A, α = $α, height = $height, l0 = $l0, alpha = $α")
        println("MBH = $MBH, L_unit = $L_unit")
        println("Rstop = $Rstop, Rh = $Rh")
        nx, ny = 128,128
        freqcgs = 230e9
        freq = freqcgs * HPL/(ME * CL * CL) 
        cam_dist, cam_theta_angle, cam_phi_angle = 1000.0, 60., 0.
        Dsource = 7.778e3 * PC
        Xcam = camera_position(cam_dist, cam_theta_angle, cam_phi_angle)
        p = Params(0.0, 0.0, nx, ny, 0.0, 0.01, 50000)
        DX = 30.0
        DY = 30.0
        Image =  zeros(Float64, nx, ny)
        fovx = DX/cam_dist
        fovy = DY/cam_dist
        scale_factor = (DX * L_unit / nx) * (DY * L_unit / ny) / (Dsource * Dsource) / JY;

        @printf("DX = %.15e, L_unit = %.15e, Dsource = %.15e, nx = %d, ny = %d, DY = %.15e, JY = %.15e\n",
            DX, L_unit, Dsource, nx, ny, DY, JY);

        println("Camera position: Xcam = [$(Xcam[1]), $(Xcam[2]), $(Xcam[3]), $(Xcam[4])]")
        println("fovx = $fovx, fovy = $fovy")
        println("DX = $DX, DY = $DY")
        println("Running on ", Threads.nthreads(), " threads")

        if(USE_KRANG)
            metric = Krang.Kerr(0.99);
            θo = 60. * π / 180;
            αmin, αmax = -3.0, 3.0
            βmin, βmax = -3.0, 3.0
            res = 4
            camera = Krang.IntensityCamera(metric, θo, αmin, αmax, βmin, βmax, res);
            lines = Krang.generate_ray.(camera.screen.pixels, 1_000)


            for idx in eachindex(lines)
                println("Processing Pixel $idx")
                J = fld(idx - 1, res) + 1
                I = mod(idx - 1, res) + 1
                line = lines[idx]
                nstep = length(line)
                # Print the field names of the struct type of pt
                t = [pt.ts for pt in line]
                r = [pt.rs for pt in line]
                th = [pt.θs for pt in line]
                phi = [pt.ϕs for pt in line]
                X = [t, log.(r), th/π, phi]
                sr = [pt.νr for pt in line]
                sθ = [pt.νθ for pt in line]
                dl = sqrt.((diff(t).^2 + diff(r).^2 + diff(th).^2 + diff(phi).^2))
                Kcon = zeros(Float64, nstep, NDIM)
                for k in 1:nstep
                    Kcon[k, :] = calcKcon((-1.)^(sr[k]), (-1.)^(sθ[k]), r[k], th[k], phi[k], metric, αmin, αmax, βmin, βmax, θo, res, I, J)
                end
                traj = [OfTraj(dl[k], MVec4(X[1][k], X[2][k], X[3][k], X[4][k]),
                                    MVec4(Kcon[k,1], Kcon[k,2], Kcon[k,3], Kcon[k,4]),
                                    MVec4(undef), MVec4(undef)) for k in 1:(nstep-1)]
                Xi = MVec4(undef)
                Kconi = MVec4(undef)
                Xf = MVec4(undef)
                Kconf = MVec4(undef)

                for k in 1:NDIM
                    Xi[k] = traj[nstep-1].X[k]
                    Kconi[k] = traj[nstep-1].Kcon[k]
                end
                ji, ki = get_analytic_jk(Xi, Kconi, freqcgs)
                #println("Pixel ($i, $j) has total nstep = $nstep")
                Intensity= 0.0
                while(nstep >= 2)
                    for k in 1:NDIM
                        Xf[k] = traj[nstep - 1].X[k]
                        Kconf[k] = traj[nstep - 1].Kcon[k]
                    end


                    if !radiating_region(Xf)
                        nstep -= 1
                        continue
                    end

                    jf, kf = get_analytic_jk(Xf, Kconf, freqcgs)

                    Intensity = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)        
                    ji = jf
                    ki = kf
                    nstep -= 1
                end 
            end
        end
        return
        
        for i in 0:(nx - 1)
            println("Processing pixel row ($i)")
            Threads.@threads for j in 0:(ny - 1)
                traj, nstep, = get_pixel(i, j, Xcam, p, fovx, fovy, freq)

                # if(i == 2 && j == 1)
                #     println("Trajectory for pixel ($i, $j):")
                #     for step in 1:nstep
                #         r, th = bl_coord(traj[step].X)
                #         @printf("Step %d: r =%.15e\n", step, r)
                #         @printf("Kcon = [%.15e, %.15e, %.15e, %.15e]\n", 
                #             traj[step].Kcon[1], traj[step].Kcon[2], 
                #             traj[step].Kcon[3], traj[step].Kcon[4])
                #     end
                # end
                Xi = MVec4(undef)
                Kconi = MVec4(undef)
                Xf = MVec4(undef)
                Kconf = MVec4(undef)

                for k in 1:NDIM
                    Xi[k] = traj[nstep].X[k]
                    Kconi[k] = traj[nstep].Kcon[k]
                end
                ji, ki = get_analytic_jk(Xi, Kconi, freqcgs)
                #println("Pixel ($i, $j) has total nstep = $nstep")
                Intensity= 0.0
                while(nstep >= 2)
                    for k in 1:NDIM
                        Xf[k] = traj[nstep - 1].X[k]
                        Kconf[k] = traj[nstep - 1].Kcon[k]
                    end


                    if !radiating_region(Xf)
                        nstep -= 1
                        continue
                    end

                    jf, kf = get_analytic_jk(Xf, Kconf, freqcgs)

                    Intensity = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)    
                    # if(i == 0 && j == 0)
                    #     r, th = bl_coord(Xf)
                    #     @printf("At step %d \n", nstep)
                    #     @printf("Radius = %.15e, theta = %.15e\n", r, th)
                    #     @printf("X = [%.15e, %.15e, %.15e, %.15e]\n", 
                    #         Xf[1], Xf[2], Xf[3], Xf[4])
                    #     @printf("Kcon = [%.15e, %.15e, %.15e, %.15e]\n",
                    #         Kconf[1], Kconf[2], Kconf[3], Kconf[4])
                    #     @printf("jf = %.15e, kf = %.15e, Intensity = %.15e\n", jf, kf, Intensity)
                    #     @printf("ji = %.15e, ki = %.15e\n", ji, ki)
                    #     @printf("dl = %.15e\n\n", traj[nstep - 1].dl)
                    # end     
                    ji = jf
                    ki = kf
                    nstep -= 1
                end 
                Image[i + 1, j + 1] = Intensity
                #println("Final intensity at pixel ($i, $j): $Intensity \n")
                # error("")
            end
        end

        Ftot::Float64 = 0.0
        Iavg::Float64 = 0.0
        Imax::Float64 = 0.0
        imax::Int = 0
        jmax::Int = 0   
        println("Image processing complete. Calculating total flux and averages...")
        for i in 1:nx
            for j in 1:ny
                Ftot += Image[i, j] * freqcgs^3 * scale_factor
                Iavg += Image[i, j]
                if (Image[i,j] * freqcgs^3) > Imax
                    imax = i
                    jmax = j
                    Imax = Image[i, j] * freqcgs^3
                end
            end
        end
        Iavg *= freqcgs^3/ (nx * ny)
        @printf("Scale = %.15e\n", scale_factor)
        println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
        println("Using freqcgs = $freqcgs, Ftot = $Ftot")
        println("nuLnu = $(Ftot * Dsource * Dsource * JY * freqcgs * 4.0 * π)")

        open("./output/Image.bin", "w") do file
            write(file, Image)
        end
    end
end

# Miscellaneous constants
const a = 0.9
const Rh = 1 + sqrt(1. - a * a);
const Rout = 1000.0
const cstartx = [0.0, log(Rh), 0.0, 0.0]
const cstopx = [0.0, log(Rout), 1.0, 2.0 * π]
const R0 = 0
const Rstop = 10000.0
const MBH = 4.063e6 * MSUN
const L_unit = GNEWT * MBH / (CL * CL)
const USE_KRANG = true

main()
GC.gc()
