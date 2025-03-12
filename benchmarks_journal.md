# Benchmarks

This file will provide a benchmark comparison for every optimization I try to implement. This way, I can easily see if it worked. 

Right now, to benchmark it, I'm using "BenchmarkTools" package such as @benchmark include("AD_ODE2.jl")

## First Commit
julia> @benchmark include("AD_ODE.jl") samples=10
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  1.469 s …    1.828 s  ┊ GC (min … max): 1.55% … 3.37%
 Time  (median):     1.638 s               ┊ GC (median):    3.89%
 Time  (mean ± σ):   1.643 s ± 188.583 ms  ┊ GC (mean ± σ):  3.40% ± 1.28%

  █  █                                              █      █
  █▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█ ▁
  1.47 s         Histogram: frequency by time         1.83 s <

 Memory estimate: 194.38 MiB, allocs estimate: 6427634.

## Optimization with Jacobian

I started using the Jacobian instead of manually calculating each derivative. I suspect the increase in time occurred because the Jacobian is calculating two derivatives that I don't use, as they are zero. This makes the process more expensive.

BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  1.905 s …   2.005 s  ┊ GC (min … max): 2.32% … 2.74%
 Time  (median):     1.920 s              ┊ GC (median):    2.86%
 Time  (mean ± σ):   1.944 s ± 54.105 ms  ┊ GC (mean ± σ):  2.69% ± 0.35%

  █       █                                               █
  █▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.91 s         Histogram: frequency by time        2.01 s <

 Memory estimate: 285.23 MiB, allocs estimate: 9309389.

 ## RK2 implementation

 For this, I'm using  the AD_ODE_RK2.jl file to solve the ODE system.

  Memory estimate: 303.25 MiB, allocs estimate: 8910994.

julia> @benchmark include("AD_ODE_RK2.jl")
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  2.284 s …    2.495 s  ┊ GC (min … max): 4.58% … 0.00%
 Time  (median):     2.301 s               ┊ GC (median):    4.55%
 Time  (mean ± σ):   2.360 s ± 117.031 ms  ┊ GC (mean ± σ):  3.22% ± 2.90%

  █   █                                                    █
  █▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  2.28 s         Histogram: frequency by time          2.5 s <

 Memory estimate: 303.26 MiB, allocs estimate: 8911007.