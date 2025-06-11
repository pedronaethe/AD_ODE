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
include("constants.jl")
include("parameters.jl")
include("camera.jl")
include("debug_functions.jl")
include("metrics.jl")
include("coords.jl")
include("tetrads.jl")
include("utils.jl")
include("radiation.jl")
include("./models/$(MODEL).jl")
include("geodesics.jl")


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

    if nstep >= params.maxnstep - 1
        @error "Max number of steps exceeded at pixel ($i, $j)"
    end

    return traj, nstep
end


function calcKcon(sr::Bool, sθ::Bool, r::Float64, θ::Float64, ϕ::Float64, metric::Kerr, η::Float64, λ::Float64, nstep::Int)
    """
    Calculates the covariant 4-velocity of the photon for each point in the geodesics in internal coordinates.
    
    Observations:
    - We follow the equation 9 of Gelles et al. 2021 https://arxiv.org/pdf/2105.09440
    - The function Krang.p_bl_d returns the wavevector in BL coordinates, so we transform for the native coordinates.
    """
    X = MVec4(0, log(r), θ/π, ϕ)
    Kcovbl = Krang.p_bl_d(metric, r, θ, η, λ, sr, sθ) 
    Kcovbl *= HPL * 230e9/(ME * CL * CL)

    #Transform Kcov to Kcon
    bl_gcov = gcov_bl(r, θ)
    bl_gcon = gcon_func(bl_gcov)
    KconBL = flip_index(MVec4(Kcovbl), bl_gcon)

    #Transform from BL to KS coordinates
    KconKS = bl_to_ks(X, KconBL)

    #Now convert to Native Coordinates
    KconNC = vec_from_ks(X, KconKS)
    return KconNC
end

function integrate_emission!(traj::Vector{OfTraj}, nstep::Int, Image::Matrix{Float64}, I::Int, J::Int)
    """
    Integrates the emission along the geodesic trajectory.
    
    Parameters:
    @traj: Vector of OfTraj objects containing the geodesic trajectory.
    @nstep: Number of steps in the trajectory.
    @Image: Matrix to store the integrated intensity.
    @I: x-index of the pixel in the image plane.
    @J: y-index of the pixel in the image plane.

    """
    #println("For Pixel ($I, $J), with nstep $nstep:")
    Xi = MVec4(undef)
    Kconi = MVec4(undef)
    Xf = MVec4(undef)
    Kconf = MVec4(undef)
    nstep = nstep - 1
    #println("Pixel ($I, $J) with nstep = $nstep")
     for k in 1:NDIM
            Xi[k] = traj[nstep-1].X[k]
            Kconi[k] = traj[nstep-1].Kcon[k]
    end
    ji, ki = get_analytic_jk(Xi, Kconi, freqcgs)
    Intensity = 0.0
    while(nstep >= 2)
        for k in 1:NDIM
            Xf[k] = traj[nstep - 1].X[k]
            Kconf[k] = traj[nstep - 1].Kcon[k]
        end

        if(MODEL == "thin_disk")
            if(thindisk_region(Xi, Xf))
                get_thindisk_intensity(Xi, Intensity)
            end
        end

        if !radiating_region(Xf)
            nstep -= 1
            continue
        end

        jf, kf = get_jk(Xf, Kconf, freqcgs)
        Intensity::Float64 = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)

        # if(I == 1 && J == 1 && nstep > 2300)
        #     println("At step $nstep:")
        #     println("r = $(exp(Xf[2])), th = $(Xf[3] * π)")
        #     println("Kcon = $(Kconf[1]), $(Kconf[2]), $(Kconf[3]), $(Kconf[4])")
        #     println("jf = $jf, kf = $kf,  Intensity = $Intensity")
        #     println("ji = $ji, ki = $ki")
        #     println("dl = $(traj[nstep - 1].dl)\n")
        # end

        if(Intensity > 1e-30)
            println("Intensity at pixel ($I, $J) after step $nstep: $Intensity")
            println("r = $(exp(Xf[2])), θ = $(Xf[3] * π), ϕ = $(Xf[4])")
            println("ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            println("dl = $(traj[nstep - 1].dl)")
            error()
        end
        if (isnan(Intensity) || isinf(Intensity))
            @error "NaN or Inf encountered in intensity calculation at pixel ($I, $J)"
            println("Intensity = $Intensity, ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            print_vector("Kconf =", Kconf)
            print_vector("Kconi =", Kconi)

            error("NaN or Inf encountered in intensity calculation")
        end
        ji = jf
        ki = kf
        nstep -= 1
    end
    Image[I, J] = Intensity
    #println("Final intensity at pixel ($I, $J): $Intensity \n")
end

function calculate_intensity_krang(scale_factor_ipole::Float64)
    Image =  zeros(Float64, nx, ny)
    metric = Krang.Kerr(a);
    θo = thcam * π / 180;
    dα = (αmax - αmin) / (nx)
    dβ = (βmax - βmin) / (ny)
    #scale_factor = dα * dβ * L_unit^2 / (Dsource * Dsource) / JY
    scale_factor = scale_factor_ipole
    res = nx
    println("Using krang for geodesics calculations, this may take a while...")
    camera = Krang.IntensityCamera(metric, θo, αmin, αmax, βmin, βmax, res);
    lines = Krang.generate_ray.(camera.screen.pixels, krang_points)


    for idx in eachindex(lines)
        Intensity= 0.0
        I = div(idx - 1, res) + 1
        J = mod(idx - 1, res) + 1
        if J == 1
            println("Processing row $I out of $(res)")
        end
        line = lines[idx]
        nstep = length(line)

        t = [pt.ts for pt in line]
        r = [pt.rs for pt in line]
        th = [pt.θs for pt in line]
        phi = [pt.ϕs for pt in line]
        
        #print where the line started, where it is ending and the minimum value of R
        # println("RayStart r = $(r[1]), Ray Finish r = $(r[end]), min r = $(minimum(r))")
        X = [t, log.(r),th/π, phi]
        # println("StartX = $(t[1]), log(r)=$(log(r[1])), θ=$(th[1]/π), ϕ=$(phi[1])")

        sr = [pt.νr for pt in line]
        sθ = [pt.νθ for pt in line]
        Kcon = zeros(Float64, nstep, NDIM)
        dl = zeros(Float64, nstep)

        α = αmin + (αmax - αmin) * (I - 1) / (res - 1)
        β = βmin + (βmax - βmin) * (J - 1) / (res - 1)
        η = Krang.η(metric, α, β, θo)
        λ = Krang.λ(metric, α, θo)

        for k in 1:nstep
            if(r[k] < Rh + 0.0001)
                continue
            end
            Kcon[k, :] = calcKcon(sr[k], sθ[k], r[k], th[k], phi[k], metric, η, λ, k)
        end
        # println("Start Kcon = $(Kcon[1, :])")
        dl = sqrt.((diff(X[1]).^2 + diff(X[2]).^2 + diff(X[3]).^2 + diff(X[4]).^2))

        traj = [OfTraj(dl[k], MVec4(X[1][k], X[2][k], X[3][k], X[4][k]),
                            MVec4(Kcon[k,1], Kcon[k,2], Kcon[k,3], Kcon[k,4]),
                            MVec4(undef), MVec4(undef)) for k in 1:(nstep-1)]
        integrate_emission!(traj, nstep, Image, I, J)
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
    println("Using freqcgs = $freqcgs, Ftot = $(Ftot)")
    println("nuLnu = $(Ftot * Dsource * Dsource * JY * freqcgs * 4.0 * π)")

    open("./output/Image.bin", "w") do file
        write(file, Image)
    end
end

function main()
    @time begin
        check_parameters()
        println("Model Parameters: A = $A, α = $α_analytic, height = $height, l0 = $l0")
        println("MBH = $MBH, L_unit = $L_unit")
        freq = freqcgs * HPL/(ME * CL * CL) 
        cam_dist, cam_theta_angle, cam_phi_angle = rcam, thcam, phicam
        Xcam = camera_position(cam_dist, cam_theta_angle, cam_phi_angle)
        p = Params(0.0, 0.0, nx, ny, 0.0, eps, maxnstep)
        Image =  zeros(Float64, nx, ny)
        
        fovx = DX/(cam_dist)
        fovy = DY/cam_dist
        scale_factor = (DX * L_unit / nx) * (DY * L_unit / ny) / (Dsource * Dsource) / JY;

        println("Dsource = $Dsource")
        println("Running on ", Threads.nthreads(), " threads")

        
        if(USE_KRANG)
            calculate_intensity_krang(scale_factor)
            return
        end
        
        for i in 0:(nx - 1)
            println("Processing row $i out of $(nx)")
            Threads.@threads for j in 0:(ny - 1)
                traj, nstep, = get_pixel(i, j, Xcam, p, fovx, fovy, freq)
                integrate_emission!(traj, nstep, Image, i + 1, j + 1)
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

main()
GC.gc()
