# Jipole
## Overview

Jipole is a Julia-based code under development that performs radiative transfer in curved spacetimes. It's intended to use automatic differentiation (autodiff) to compute derivatives of input parameters.

## Current State
The current version of Jipole focuses on implementing and validating test problems described in Section 3.2 of the [Gold et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6) by porting the C-based code [ipole](https://github.com/moscibrodzka/ipole) to Julia.

## Running the Project
To run the code, open a terminal and navigate to the root project directory. Use the following command to start Julia with project-specific dependencies and set the number of threads:

`JULIA_NUM_THREADS=xx julia --project="."`

Replace xx with the number of threads you want Julia to use.

Then, in the Julia REPL, run:
`include("./src/main.jl")`
This will execute the main driver script.