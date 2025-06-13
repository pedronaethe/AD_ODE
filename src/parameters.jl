# BH spin parameter
const a = 0.9
# Event horizon radius
const Rh = 1 + sqrt(1. - a * a);

# Used to calculate the maximum grid radius
const Rout = 1000.0

# The minimum grid native coordinates.
const cstartx = [0.0, log(Rh), 0.0, 0.0]

# The maximum grid native coordinates. Stops at the event horizon.
const cstopx = [0.0, log(Rout), 1.0, 2.0 * π]

# Legacy for GRMHD data parameters
const R0 = 0

# Mass of the black hole in solar masses
const MBH = 4.063e6
const L_unit = GNEWT * MBH * MSUN / CL^2  # Length unit in gravitational radius (Rg)

# Whether or not to use Krang for geodesic calculations
const USE_KRANG = true

# Number of pixels in the x and y directions of the image
const nx = ny = 40

# Frequency in Hz for the image
const freqcgs = 230e9

# Source distance in parsecs
const Dsource = 7.778e3 * PC

## Camera parameters:
# Camera position in graviational radius units (Rg). Currently only used in ipole integration
const rcam = 1.e3
# Camera polar angle in degrees
const thcam = 60.
# Camera azimuthal angle in degrees
const phicam = - 90.

const krang_points = 1_000
## Krang camera parameters
# α is the x coordinate in the image plane
const αmin, αmax = -5.7 , 5.7
# β is the y coordinate in the image plane
const βmin, βmax = -5.7 , 5.7

## Ipole integration parameters
# Image size in gravitational radius units (Rg) used to calculate the field of view (ipole integration only)
const DX = 30.0
const DY = 30.0

# Precision factor for the geodesic integration (only when using ipole integration). 
const eps = 0.01

# Maximum number of steps for the geodesic integration (only when using ipole integration).
const maxnstep = 50000

# Radius at which stop tracking the geodesics (Used only when doing ipole integration)
const Rstop = 10000.0

# Choosing MODEL
const MODEL = "analytic"  # Options: "analytic", "thin_disk"

#print geodesics to ./output/pixelij_coordinates.txt file
const print_geodesics = false

