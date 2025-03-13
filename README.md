
# AutoDiff + ODEsolver project

## Overview
This project is currently underdevelopment and aims to couple Autodifferenciation with differential equations solvers.

## Files

### `./src/CG_analysis.jl`
This file contains the code for the conjugate gradient method to solve the rabbits and foxes model using automatic differentiation.

### `./src/MCMC_analysis.jl`
This file is responsible for the MCMC analysis of the rabbits and foxes model using a hand made metropolis hastings algorithm. 


### `./src/AD_ODE_RK2.jl`
This file solves the system of ODEs using the DifferentialEquations package.

### `./old/AD_ODE2.jl`
This file solves the system of ODEs using a explicit Euler method made by myself.


## Running the Project
To run any of the scripts, open a terminal and navigate to the project directory. Use the `julia --project="." ` command to load the packages in this repository, followed by the script name. For example:
```
include("./src/MCMC_analysis.jl")
```

This should run the solving/MCMC algorithms and generate the respective img files inside the folder already
