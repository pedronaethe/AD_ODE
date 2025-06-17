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



struct Pixel{T} <: Krang.AbstractPixel{T}
    metric::Kerr{T}
    "Bardeen coordiantes if ro is \`Inf`, otherwise latitude and longitude"
    screen_coordinate::NTuple{2,T}
    "Radial roots"
    roots::NTuple{4,Complex{T}}
    "Radial antiderivative"
    I0_o::T
    "Total possible Mino time"
    total_mino_time::T
    "Angular antiderivative"
    absGθo_Gθhat::NTuple{2,T}
    "Inclination"
    θo::T
    ro::T
    η::T
    λ::T
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

        # if(I == 15 && J == 15)
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


function calculate_intensity_krang(scale_factor_ipole::Float64, rad_fovx::Float64, rad_fovy::Float64)
    Image =  zeros(Float64, nx, ny)
    metric = Krang.Kerr(a);
    θo = thcam * π / 180;
    dα = (αmax - αmin) / (nx)
    dβ = (βmax - βmin) / (ny)
    scale_factor = dα * dβ * L_unit^2 / (Dsource * Dsource) / JY
    scale_factor = scale_factor_ipole
    res = nx
    println("Using krang for geodesics calculations, this may take a while...")
    println("rad_fovx = $rad_fovx, rad_fovy = $rad_fovy [rad]")
    # Xcam = camera_position(rcam, thcam, phicam)
    # lines = Matrix{Vector{Krang.Intersection}}(undef, res, res)
    # camera = Matrix{Krang.AbstractPixel{Float64}}(undef, res, res)
    # for i in 1:res
    #     for j in 1:res
    #         X,Kconp = init_XK(i,j, Xcam, params, fovx, fovy)
    #         Kconp_KS = vec_to_ks(X, Kconp)
    #         kconp_bl = ks_to_bl(X, Kconp_KS)
    #         #Kcovp_BL = flip_index(kconp_bl, gcov_bl(r,th))

    #         camera[i,j] = KrangGeoPinHole(rcam, θo, kconp_bl, metric)
    #         lines[i,j] = Krang.generate_ray(camera[i,j], krang_points)
    #     end
    # end
    camera = Krang.IntensityCamera(metric, θo, 1000.0,-rad_fovx, rad_fovx, -rad_fovy, rad_fovy, res);
    #camera = Krang.IntensityCamera(metric, θo, αmin, αmax, βmin, βmax, res);
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
        X = [t, log.(r),th/π, phi]
        # println("StartX = $(t[1]), log(r)=$(log(r[1])), θ=$(th[1]/π), ϕ=$(phi[1])")


        # Print r, th, phi for each step to a file named "pixelIJ_coordinates.txt"
        if(print_geodesics)
            println("Printing geodesics in txt file for pixel ($I, $J)\n")
            filename = "./output/pixel$(I)$(J)_coordinates.txt"
            if isfile(filename)
                rm(filename)
            end
            open(filename, "a") do fp
                for k in 1:nstep
                @printf(fp, "Step %d: r = %.15f, th = %.15f, phi = %.15e\n", k, r[k], th[k], phi[k])
                end
            end
        end


        #check if any r is inf or nan, in case it is, print the r array index where it happens
        if any(isnan.(r)) || any(isinf.(r))
            @error "NaN or Inf encountered in r at pixel ($I, $J)"
            for k in eachindex(r)
                if isnan(r[k]) || isinf(r[k])
                    @error "NaN or Inf encountered in r at index $k: r[$k] = $(r[k])"
                end
            end
            error("NaN or Inf encountered in r")
        end

        sr = [pt.νr for pt in line]      
        sθ = [pt.νθ for pt in line]
        Kcon = zeros(Float64, nstep, NDIM)
        dl = zeros(Float64, nstep)

        η = Krang.η(camera.screen.pixels[idx])
        λ = Krang.λ(camera.screen.pixels[idx])

        last_valid_index = 0
        for k in 1:nstep
            if(r[k] == 0)
                break
            end
            if (r[k] < (Rh + 0.5 ))
                r[k] = 0.0
                break
            end
            Kcon[k, :] = calcKcon(sr[k], sθ[k], r[k], th[k], phi[k], metric, η, λ, k)
            last_valid_index = k
        end


        # println("Pixel ($I, $J) with nstep = $nstep")
        # println("RayStart r = $(r[1]), Ray Finish r = $(r[last_valid_index]), min r = $(minimum(r[1:last_valid_index]))\n")
        #println("At cartesian coordinates: x = $(r[1] * sin(th[1]) * cos(phi[1])), y = $(r[1] * sin(th[1]) * sin(phi[1])), z = $(r[1] * cos(th[1]))")
        #println("Cartesian coordinates at end of ray: x = $(r[last_valid_index] * sin(th[last_valid_index]) * cos(phi[last_valid_index])), y = $(r[last_valid_index] * sin(th[last_valid_index]) * sin(phi[last_valid_index])), z = $(r[last_valid_index] * cos(th[last_valid_index]))")
        #dl = sqrt.((diff(X[1]).^2 + diff(X[2]).^2 + diff(X[3]).^2 + diff(X[4]).^2))

        # Compute dl up to the last valid index
        dl[1:last_valid_index-1] = sqrt.((diff(X[1][1:last_valid_index]).^2 .+ 
                          diff(X[2][1:last_valid_index]).^2 .+ 
                          diff(X[3][1:last_valid_index]).^2 .+ 
                          diff(X[4][1:last_valid_index]).^2))
        traj = [OfTraj(dl[k], MVec4(X[1][k], X[2][k], X[3][k], X[4][k]),
                            MVec4(Kcon[k,1], Kcon[k,2], Kcon[k,3], Kcon[k,4]),
                            MVec4(undef), MVec4(undef)) for k in 1:last_valid_index]
        integrate_emission!(traj, last_valid_index, Image, I, J)
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


    println("Saving image to output/Image.txt")
    if isfile("./output/Image.txt")
        rm("./output/Image.txt")
    end
    open("./output/Image.txt", "w") do file
        for i in 1:nx
            for j in 1:ny
                @printf(file, "%.15e ", Image[i, j]* freqcgs^3 * scale_factor)
            end
            println(file)
        end
    end
end

function IpoleGeoIntensityIntegration(Xcam, fovx, fovy, freq, eps_ipole, maxnstep, res, θo; integrate_emission_flag=true)
    
    Image = integrate_emission_flag ? zeros(Float64, res, res) : nothing
    trajs = !integrate_emission_flag ? Matrix{Vector{OfTraj}}(undef, res, res) : nothing
    freq_unitless = freq * HPL/(ME * CL * CL)  # Convert frequency to unitless
    for i in 0:(res - 1)
        println("Processing row $i out of $(res)")
        Threads.@threads for j in 0:(res - 1)
            traj, nstep = get_pixel(i, j, Xcam, eps_ipole, maxnstep, fovx, fovy, freq_unitless, res, θo)
            if print_geodesics
                println("Printing geodesics in txt file for pixel ($i, $j)\n")
                filename = "./output/pixel$(i)$(j)_coordinates_ipole.txt"
                if isfile(filename)
                    rm(filename)
                end
                open(filename, "a") do fp
                    for k in 1:nstep
                        @printf(fp, "Step %d: r = %.15f, th = %.15f, phi = %.15e\n", k, exp(traj[k].X[2]), traj[k].X[3] * π, traj[k].X[4])
                    end
                end
            end
            if integrate_emission_flag
                integrate_emission!(traj, nstep, Image, i + 1, j + 1)
            else
                trajs[i+1, j+1] = traj
            end
        end
    end
    return integrate_emission_flag ? Image : (trajs)
end


function main()
    @time begin
        check_parameters()
        println("Generating an image with size $(nx) x $(ny) pixels")
        println("Model Parameters: A = $A, α = $α_analytic, height = $height, l0 = $l0, a = $a")
        println("MBH = $MBH, L_unit = $L_unit")
        println("Dsource = $Dsource")

        Xcam = camera_position(rcam, thcam, phicam)
        Image =  zeros(Float64, nx, ny)

        fovx = DX/(rcam)
        fovy = DY/(rcam) 

        #rad_fovx = DX /Dsource * L_unit
        #rad_fovy = DY /Dsource * L_unit
        # rad_fovx = π/2.
        # rad_fovy = π/2.
        scale_factor = (DX * L_unit / nx) * (DY * L_unit / ny) / (Dsource * Dsource) / JY 
        println("Running on ", Threads.nthreads(), " threads")

        
        if(USE_KRANG)
            calculate_intensity_krang(scale_factor *2, fovx * 0.25, fovy * 0.25)
            #calculate_intensity_krang(scale_factor * 2, fovx * 0.5, fovy * 0.5)
        end

        Image = IpoleGeoIntensityIntegration(Xcam, fovx, fovy, freqcgs, eps_ipole, maxnstep, nx, 0; integrate_emission_flag=true )

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
        println("Saving image to output/Image_ipole.txt")
        if isfile("./output/Image_ipole.txt")
            rm("./output/Image_ipole.txt")
        end
        open("./output/Image_ipole.txt", "w") do file
            for j in 1:nx
                for i in 1:ny
                    @printf(file, "%.15e ", Image[i, j]* freqcgs^3 * scale_factor)
                end
                println(file)
            end
        end
    end
end

#main()
#GC.gc()
