
include("../metrics.jl")

export thindisk_region, radiating_region, get_thindisk_intensity, thindisk_vals!, krolikc, fbbpolemis!, bnu, interp_chandra!

f::Float64 = 1.8
ch_mu = Float64[]
ch_I = Float64[]
ch_delta = Float64[]

z1 = 1.0 + (1.0 - a^2)^(1.0 / 3.0) * ((1.0 + a)^(1.0 / 3.0) + (1.0 - a)^(1.0 / 3.0))
z2 = sqrt(3.0 * a^2 + z1^2)
r_isco = 3.0 + z2 - copysign(sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)), a)
z1 = nothing
z2 = nothing

T0 = (3.0 / (8.0 * π) * GNEWT * MBH * M_unit / L_unit^3 / SIG)^(1.0 / 4.0)
function thindisk_region(Xi::MVec4, Xf::MVec4)::Bool
    """
    Checks if the geodesic segment crosses the thin disk region.
    Parameters:
    @Xi: Starting position vector in internal coordinates.
    @Xf: Ending position vector in internal coordinates.
    """
    _, th_i = bl_coord(Xi)
    r_f, th_f = bl_coord(Xf)
    
    midplane::Bool = (sign(th_i - π/2) != sign(th_f - π/2))
    em_region::Bool = (r_f > r_isco && r_f < Rout)
    return midplane && em_region
end
function radiating_region(X::MVec4)
    return false
end

function get_thindisk_intensity(X::MVec4, I::Float64)
    r, _ = bl_coord(X)

    if(r > Rh)
        Temp::Float64 = 0.
        omega::Float64 = 0.
        thindisk_vals!(r, Temp, omega)

        Ucon::MVec4 = zero(MVec4)
        Ucov::MVec4 = zero(MVec4)
        Bcon::MVec4 = zero(MVec4)
        Bcov::MVec4 = zero(MVec4)
        get_model_fourvel!(X, Ucon, Ucov, Bcon, Bcov)

        mu::Float64 = abs(cos(get_bk_angle(X, Kcon, Ucov, Bcon, Bcov)))
        nu::Float64 = get_fluid_nu(Kcon, Ucov)
        if(nu <= 0. || any(isnan(nu)) || any(isinf(nu)))
            println("At X = $X\n Kcon = $Kcon")
            println("Ucov = $Ucov")
            println("Kcon $Kcon")
            error("Frequency must be positive, got nu = $nu")
        end

        fbbpolemis(nu, Temp, mu, I)
    else
        I = 0.0
    end
end

function thindisk_vals!(r::Float64, Temp::Float64, omega::Float64)
    """
    Computes the temperature and angular frequency for the thin disk model.
    Parameters:
    @r: Radial coordinate in Boyer-Lindquist coordinates.
    @T: Temperature to be computed.
    @omega: Angular frequency to be computed.
    """
    b::Float64 = 1. - 3. /r + 2. *a/r^(3/2)
    kc = krolikc(r, a)
    d::Float64 = r *r - 2. * r + a * a
    lc::Float64 = (r_isco * r_isco - 2. * a * sqrt(r_isco) + a * a) / (sqrt(r_isco) - 2. * sqrt(r_isco) + a)
    hc::Float64 = (2. * r - a * lc) / d
    ar::Float64 = (r * r + a * a)^2 - a * a * d * sin(π/2.)^2
    om::Float64 = 2. * a * r / ar

    #Start the disk at r_isco, the marginally stable orbit which N-K take as an inner boundary condition.
    #End it eventually.
    if(r > r_isco)
        omega = max(1. / (r^(3/2) + a), om)
    else
        omega = max((lc + a * hc) / (r * r + 2. * r * (1. + hc)), om)
    end

    if(r > r_isco && r < Rout)
        Temp = T0 * (kc / b / r^3)^(1. /4.)
    else
        Temp = T0 / 1.e5
    end
end

function krolikc(r::Float64, a::Float64)
    """
    Computes the Krolik & Hawley (2002) disk model for a given radius and spin.
    Parameters:
    @r: Radial coordinate in Boyer-Lindquist coordinates.
    @a: Spin parameter of the black hole.
    """
    y::Float64 = sqrt(r)
    yms::Float64 = sqrt(r_isco)
    y1::Float64 = 2. * cos(1. / 3. * (acos(a) - π))
    y2::Float64 = 2. * cos(1. / 3. * (acos(a) + π))
    y3::Float64 = -2. * cos(1. / 3. * acos(a))
    arg1::Float64 = 3. * a / (2. * y)
    arg2::Float64 = 3. * (y1 - a)^2 / (y * y1 * (y1 - y2) * (y1 - y3))
    arg3::Float64 = 3. * (y2 - a)^2 / (y * y2 * (y2 - y1) * (y2 - y3))
    arg4::Float64 = 3. * (y3 - a)^2 / (y * y3 * (y3 - y1) * (y3 - y2))
    return 1. - yms / y - arg1 * log(y / yms) - arg2 * log((y - y1) / (yms - y1)) -
           arg3 * log((y - y2) / (yms - y2)) - arg4 * log((y - y3) / (yms - y3))
end

function fbbpolemis!(nu::Float64, Temp::Float64, cosne::Float64, I::Float64)
    """
    Computes the emissivity for the thin disk model.
    Parameters:
    @nu: Frequency in internal coordinates.
    @Temp: Temperature in internal coordinates.
    @mu: Cosine of the angle between the photon direction and the magnetic field.
    @I: Intensity to be computed.
    """
    I = f^(-4.) * bnu(nu, Temp * f)
    if(nu <= 0. || any(isnan(nu)) || any(isinf(nu)))
        error("Frequency must be positive, got nu = $nu")
    end
    interpI::Float64 = undef
    interpDel::Float64 = undef
    interp_chandra(cosne, interpI, interpDel)

    I *= interpI

    return I
end

function bnu(nu::Float64, Temp::Float64)
    """
    Computes the Planck function for a given frequency and temperature.
    Parameters:
    @nu: Frequency.
    @TTemp: Temperature
    """
    return 2 * HPL * nu^3 / (CL^2) / (exp(HPL * nu / (KBOL * Temp)) - 1)
end



function interp_chandra!(mu::Float64, i::Float64, del::Float64)
    """
    Interpolates the Chandra emissivity and absorption coefficient.
    Parameters:
    @mu: Cosine of the angle between the photon direction and the magnetic field.
    @interpI: Interpolated emissivity to be computed.
    @interpDel: Interpolated absorption coefficient to be computed.
    """
    indx::Int = 0
    weight::Float64 = get_weight(ch_mu, mu, indx)
    i = (1. - weight) * ch_I[indx] + weight * ch_I[indx + 1]
    del = (1. - weight) * ch_delta[indx] + weight * ch_delta[indx + 1]
end


function get_weight(xx::Vector{Float64}, x::Float64, jlo::Ref{Int})
    """
    Computes the weight for interpolation based on the given x value and the xx array.

    Parameters:
    @xx: Vector of x values for interpolation.
    @x: The x value for which to compute the weight.
    @jlo: Reference to the index of the lower bound for interpolation.
    """
    while(xx[jlo[]] < x)
        jlo[] += 1
    end
    jlo[] -= 1
    return (x - xx[jlo[]]) / (xx[jlo[] + 1] - xx[jlo[]])
end

function load_chandra_tab24()
    """
    Loads the Chandra emissivity and absorption coefficient data from a file.
    The data is expected to be in the format:
    mu I delta
    where mu is the cosine of the angle, I is the emissivity, and delta is the absorption coefficient.
    """
    fname::String = "ch24_vals.txt"
    vals::IO = open(fname, "r")
    if vals === nothing
        @error "Error reading file $fname!"
        error("File not found: $fname")
    end
    npts::Int = 21  # Number of points in the table
    ch_mu::Vector{Float64} = zeros(npts)
    ch_I::Vector{Float64} = zeros(npts)
    ch_delta::Vector{Float64} = zeros(npts)
    for i in 1:npts
        ch_mu[i] = parse(Float64, readline(vals))
        ch_I[i] = parse(Float64, readline(vals))
        ch_delta[i] = parse(Float64, readline(vals))
        # println("TABLE: $(ch_mu[i]) $(ch_I[i]) $(ch_delta[i])")
    end
    close(vals)
    @info "Chandra table loaded successfully from $fname"
    return ch_mu, ch_I, ch_delta
end

function get_model_fourv!(X::MVec4, Kcon::MVec4, Ucon::MVec4, Ucov::MVec4, Bcon::MVec4, Bcov::MVec4)
    gcov::MMat4 = gcov_func(X)
    r, th = bl_coord(X)
    thindisk_vals!(r, T, omega)

    Ucon[1] = sqrt(-1. / (gcov[1, 1] + 2. * gcov[1, 4] * omega + gcov[4, 4] * omega * omega))
    Ucon[2] = 0.0
    Ucon[3] = 0.0
    Ucon[4] = omega * Ucon[1]

    Ucov = flip_index(Ucon, gcov)
    # B is handled in native coordinates here
    calc_polvec(X, Kcon, a, Bcon)

    # Flip B
    Bcov = flip_index(Bcon, gcov)
end

function calc_polvec(X::MVec4, Kcon::MVec4, a::Float64, fourf::MVec4)
    """
    Calculates the polarization vector in Kerr-Schild coordinates.
    Parameters:
    @X: Position vector in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @a: Spin parameter of the black hole.
    @fourf: Output vector for the polarization vector.
    """
    fourf_bl::MVec4 = zeros(MVec4)
    fourf_ks::MVec4 = zeros(MVec4)
    # Transform to KS and then to eKS
    bl_to_ks(X, fourf_bl, fourf_ks)
    vec_to_ks(X, fourf_ks, fourf)
    # Normalize the polarization vector
    gcov::MMat4 = gcov_func(X)
    fourf_cov::MVec4 = zeros(MVec4)
end



void calc_polvec(double X[NDIM], double Kcon[NDIM], double a, double fourf[NDIM])
{
  double fourf_bl[NDIM], fourf_ks[NDIM];
  fourf_bl[0] = 0;
  fourf_bl[1] = 0;
  fourf_bl[2] = 1;
  fourf_bl[3] = 0;

  // Then transform to KS and to eKS
  bl_to_ks(X, fourf_bl, fourf_ks);
  vec_to_ks(X, fourf_ks, fourf);

  // Now normalize
  double gcov[NDIM][NDIM], fourf_cov[NDIM];
  gcov_func(X, gcov);
  flip_index(fourf, gcov, fourf_cov);
  double normf = sqrt(fourf[0] * fourf_cov[0] + fourf[1] * fourf_cov[1] + fourf[2] * fourf_cov[2]
      + fourf[3] * fourf_cov[3]);

  MULOOP fourf[mu] /= normf;
}
