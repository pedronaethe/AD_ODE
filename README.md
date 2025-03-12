
# AutoDiff + ODEsolver project

## Overview
This project is currently underdevelopment and aims to couple Autodifferenciation with differential equations solvers.

## Files

### `MCMC_analysis.jl`
Description: This file is responsible for the MCMC analysis of the rabbits and foxes model using a hand made metropolis hastings algorithm. 


### `AD_ODE_RK2.jl`
Description: This file solves the system of ODEs using the DifferentialEquations package.

### `old/AD_ODE2.jl`
Description: This file solves the system of ODEs using a explicit Euler method made by myself.


## Running the Project
To run any of the scripts, open a terminal and navigate to the project directory. Use the `julia --project="." ` command to load the packages in this repository, followed by the script name. For example:
```
include("MCMC_analysis.jl")
```

This should run the solving/MCMC algorithms and generate the respective img files inside the folder already

## Dependencies
For now, this project has used some packages available in the base such as
