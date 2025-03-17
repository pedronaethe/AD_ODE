using DifferentialEquations
using ForwardDiff
using Plots
using StatsBase
using Distributions
using Random
using Dates

"""
This file is responsible for the MCMC analysis of the rabbits and foxes model using a hand made metropolis hastings algorithm.
"""

function func(du, u, p, t)
    """
    This function defines the system of ODEs for the rabbits and foxes model.
    
    Parameters:
    @du - derivative of the state variables
    @u - state variables [nrabbits, nfoxes]
    @p - parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @t - time
    """
    nrabbits, nfoxes = u[1], u[2]
    epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes = p
    
    du[1] = (epsilon_rabbits - gamma_rabbits * nfoxes) * nrabbits
    du[2] = -(epsilon_foxes - gamma_foxes * nrabbits) * nfoxes
end

function solve_system(p; u0=[1000.0, 20.0], tspan=(0.0, 100.0), solver=Midpoint(), saveat=1.0)
    """
    This function solves the system of ODEs with simpler state variables (just rabbits and foxes).
    
    Parameters:
    @p - parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @u0 - initial conditions [rabbits0, foxes0]
    @tspan - time span (start_time, end_time)
    @solver - ODE solver algorithm
    @saveat - time points to save at

    Variables:
    @prob - ODE problem
    @sol - solution of the ODE system
    """
    prob = ODEProblem(func, u0, tspan, p)
    sol = solve(prob, solver, saveat=saveat)
    return sol
end

function generate_synthetic_data(true_params; noise_level=0.05, seed=nothing)
    """
    Generate synthetic data for the model.

    Parameters:
    @true_params are the correct chosen parameters of the model.
    @noise_level is a constant to set the level of the Gaussian noise to the true solution of the ODE system.
    @seed is the random seed for reproducibility.
    
    Variables:
    @sol - solution of the ODE system
    @rabbits_obs - observed rabbits
    @foxes_obs - observed foxes
    """
    if seed === nothing
        seed = Int(Int(floor(datetime2unix(Dates.now()))))  # Use current time to generate a seed if none is provided
    end
    Random.seed!(seed)  # Set the seed
    println("Using seed: $seed")
    
    sol = solve_system(true_params, tspan=(0.0, 100.0), saveat=5.0)
    
    t = sol.t
    rabbits = sol[1, :]
    foxes = sol[2, :]
    
    #Generating synthetic observations with noise
    rabbits_obs = rabbits .* (1 .+ noise_level * randn(length(rabbits)))
    foxes_obs = foxes .* (1 .+ noise_level * randn(length(foxes)))
    
    rabbits_obs = max.(rabbits_obs, 1.0)
    foxes_obs = max.(foxes_obs, 1.0)
    
    return t, rabbits_obs, foxes_obs, rabbits, foxes
end

using Distributions



function log_likelihood(params, t, rabbits_obs, foxes_obs; sigma_rabbits=0.01, sigma_foxes=0.01)
    """
    Calculate the log-likelihood of the model given the parameters and observed data.

    Parameters:
    @params - model parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @t - time points
    @rabbits_obs - observed rabbits
    @foxes_obs - observed foxes
    @sigma_rabbits - standard deviation of the Gaussian loglikelihood for rabbits
    @sigma_foxes - standard deviation of the Gaussian loglikelihood for foxes

    Variables:
    @sol - solution of the ODE system
    @rabbits_pred - predicted rabbits
    @foxes_pred - predicted foxes
    @ll_rabbits - log-likelihood of rabbits
    @ll_foxes - log-likelihood of foxes

    Returns:
    @ll_rabbits + ll_foxes - sum of log-likelihoods

    Observations:
    logpdf(Normal(μ, σ), x) calculates the log-likelihood of x given a Normal distribution.

    logpdf(Normal, x) = - (x - μ)^2 / (2 * σ^2) - log(sqrt(2 * π) * σ^2)
    """
    lower_bounds = [0.0, 0.0, 0.0, 0.0]
    upper_bounds = [0.1, 0.001, 0.1, 0.001]
    
    if any(params .< lower_bounds) || any(params .> upper_bounds)
        println("Parameters out of bounds, returning -Inf", params)
        return -Inf
    end
    
    try
        sol = solve_system(params, tspan=(0.0, 100.0), saveat=t)
        
        if sol.retcode != :Success
            println("Integration failed due to $(sol.retcode), returning -Inf")
            return -Inf
        end
        
        rabbits_pred = sol[1, :]
        foxes_pred = sol[2, :]
        
        
        ll_rabbits = sum(logpdf.(Normal.(rabbits_pred, sigma_rabbits), rabbits_obs))
        ll_foxes = sum(logpdf.(Normal.(foxes_pred, sigma_foxes), foxes_obs))
        
        return ll_rabbits + ll_foxes
    catch
        println("Integration failure unknown, returning -Inf")
        return -Inf
    end
end

# Prior distribution
function log_prior(params)
    """
    Calculate the log-prior of the model parameters.

    Parameters:
    @params - model parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]

    Variables:
    @priors - list of prior distributions for each parameter in the same order as params
    @log_p - sum of log-priors

    Observations:
    logpdf(Uniform(a, b), x) calculates the log-likelihood of x given a Uniform distribution.

    logpdf(Uniform(a,b), x) = - log(b - a)
    """
    priors = [
        Uniform(0, 0.1),     
        Uniform(0, 0.001),   
        Uniform(0.0, 0.1),    
        Uniform(0.0, 0.001)    
    ]
    
    log_p = sum(logpdf.(priors, params))
    return log_p
end

function log_posterior(params, t, rabbits_obs, foxes_obs)
    """
    Calculate the log-posterior of the model given the parameters and observed data.

    Parameters:
    @params - model parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @t - time points
    @rabbits_obs - observed rabbits
    @foxes_obs - observed foxes

    Variables:
    @lp - check if the parameters are outside the prior support
    @ll - log-likelihood of the model
    @lp + ll - sum of log-prior and log-likelihood
    """
    lp = log_prior(params)
    
    if isinf(lp)
        return -Inf
    end
    
    ll = log_likelihood(params, t, rabbits_obs, foxes_obs)
    return lp + ll
end

function metropolis_hastings(log_post_func, initial_params, n_samples; proposal_sd)
    """
    Perform Metropolis-Hastings MCMC sampling.

    Parameters:
    @log_post_func - function to calculate the log-posterior
    @initial_params - initial parameter values
    @n_samples - number of MCMC samples
    @proposal_sd - standard deviation of the Gaussian proposal distribution

    Variables:
    @samples - matrix to store the MCMC samples
    @current_params - current parameter values
    @current_log_post - current log-posterior
    @proposal_params - proposed parameter values
    @proposal_log_post - proposed log-posterior
    @alpha - acceptance probability
    @n_accepted - counter for accepted samples

    Returns:
    @samples - MCMC samples
    @acceptance_rate - acceptance rate
    """
    
    samples = zeros(n_samples, length(initial_params))
    current_params = copy(initial_params)
    current_log_post = log_post_func(current_params)
    
    if isinf(current_log_post)
        error("Initial point has zero posterior probability")
    end
    
    samples[1, :] = current_params
    
    n_accepted = 0
    
    for i in 2:n_samples
        proposal_params = current_params + proposal_sd .* randn(size(current_params))

        proposal_log_post = log_post_func(proposal_params)
        
        alpha = min(1.0, exp(proposal_log_post - current_log_post))
        
        if rand() < alpha
            current_params = proposal_params
            current_log_post = proposal_log_post
            n_accepted += 1
        end
        
        samples[i, :] = current_params
        
        if i % 500 == 0
            acceptance_rate = n_accepted / (i - 1)
            println("Iteration $i, acceptance rate: $(round(acceptance_rate, digits=3))")
        end
    end
    
    acceptance_rate = n_accepted / (n_samples - 1)
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate
end

function plot_results(t, rabbits_obs, foxes_obs, true_rabbits, true_foxes, samples)
    """
    Plot the results of the MCMC analysis.

    Parameters:
    @t - time points
    @rabbits_obs - observed rabbits (True + noise)
    @foxes_obs - observed foxes (True + noise)
    @true_rabbits - true rabbits (True + noise)
    @true_foxes - true foxes (True + noise)
    @samples - MCMC samples

    Variables:
    @param_medians - median of the parameter samples
    @param_lower - lower bound of the 95% Confidence Interval (CI)
    @param_upper - upper bound of the 95% Confidence Interval (CI)
    @param_names - names of the parameters
    @p1 - plot of the rabbit population
    @p2 - plot of the fox population
    @p_combined - combined plot of the rabbit and fox populations
    @p_trace - plot of the MCMC trace
    @p_hist - plot of the parameter posterior distributions

    Observations:
    quantile(x, q) returns the q-th quantile of x (The value below which qth of the data lies).


    """

    param_medians = median(samples, dims=1)[:]
    param_lower = [quantile(samples[:, i], 0.025) for i in 1:size(samples, 2)]
    param_upper = [quantile(samples[:, i], 0.975) for i in 1:size(samples, 2)]
    
    println("Parameter estimates (median and 95% CI):")
    param_names = ["epsilon_rabbits", "gamma_rabbits", "epsilon_foxes", "gamma_foxes"]
    for i in 1:length(param_names)
        println("$(param_names[i]): $(round(param_medians[i], digits=6)) ($(round(param_lower[i], digits=6)) - $(round(param_upper[i], digits=6)))")
    end
    
    # Solve system with median parameters
    sol_median = solve_system(param_medians, tspan=(0.0, 100.0), saveat=t)
    rabbits_pred = sol_median[1, :]
    foxes_pred = sol_median[2, :]
    
    p1 = plot(t, rabbits_obs, seriestype=:scatter, color=:blue, label="Observed rabbits", markersize=3, size=(1200, 900))
    plot!(p1, t, true_rabbits, color=:blue, linestyle=:dash, linewidth=2, label="True rabbits")
    plot!(p1, t, rabbits_pred, color=:red, linewidth=2, label="Predicted rabbits")
    xlabel!(p1, "Time")
    ylabel!(p1, "Rabbit population")
    
    p2 = plot(t, foxes_obs, seriestype=:scatter, color=:green, label="Observed foxes", markersize=3, size=(1200, 900))
    plot!(p2, t, true_foxes, color=:green, linestyle=:dash, linewidth=2, label="True foxes")
    plot!(p2, t, foxes_pred, color=:red, linewidth=2, label="Predicted foxes")
    xlabel!(p2, "Time")
    ylabel!(p2, "Fox population")
    
    p_combined = plot(p1, p2, layout=(2, 1), size=(1200, 900))
    savefig(p_combined, "./imgs/MCMC/population_fit.png")
    
    p_trace = plot(samples, layout=(4, 1), labels=reshape(param_names, 1, 4), 
                 title=["Trace of epsilon_rabbits" "Trace of gamma_rabbits" "Trace of epsilon_foxes" "Trace of gamma_foxes"], size=(1200, 900))
    savefig(p_trace, "./imgs/MCMC/parameter_traces.png")
    
    p_hist = histogram(samples, layout=(2, 2), labels=reshape(param_names, 1, 4),
                    title=["Posterior of epsilon_rabbits" "Posterior of gamma_rabbits" "Posterior of epsilon_foxes" "Posterior of gamma_foxes"], size=(1200, 900))
    savefig(p_hist, "./imgs/MCMC/parameter_posteriors.png")
    
    return p_combined, p_trace, p_hist
end

# Main execution
function main()
    """
    Main function to run the MCMC analysis (I like to have a main function in order to wrap around things nicely in my head lol)
    This script it based off an old script I developed myself in Python a few years back. I have adapted it to Julia and my current problem.

    Variables:
    @true_params - true parameters of the model [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @t - time points
    @rabbits_obs - observed rabbits (True + noise)
    @foxes_obs - observed foxes (True + noise)
    @true_rabbits - true rabbits 
    @true_foxes - true foxes
    @initial_params - initial parameter values
    @log_post - log-posterior function
    @n_samples - number of MCMC samples
    @samples - MCMC samples
    @burnin - number of burn-in samples
    @samples_post_burnin - MCMC samples after burn-in
    @p_data - plot of the synthetic data

    Observations:
    The workflow of this function is as follows
        1. Generate synthetic data by solving the system with true parameters and adding Gaussian noise.
        2. Plot the synthetic data.
        3. Generate initial guess by adding noise to the true parameters
        4. Assign the log-posterior function to a variable and pass this to MCMC sampler so I can easily change the loglikelihood function if I want to.
        5. Run the Metropolis-Hastings MCMC algorithm.
        6. Discard the burn-in period.
        7. Plot the results.
    """


    true_params = [0.015, 0.0001, 0.03, 0.0001]
    
    println("Generating synthetic data with true parameters:")
    println("epsilon_rabbits: $(true_params[1])")
    println("gamma_rabbits: $(true_params[2])")
    println("epsilon_foxes: $(true_params[3])")
    println("gamma_foxes: $(true_params[4])")
    
    t, rabbits_obs, foxes_obs, true_rabbits, true_foxes = generate_synthetic_data(true_params, noise_level=0.1)
    
    # Plotting synthetic data so we can easily compare perturbed to unperturbed data
    p_data = plot(t, rabbits_obs, seriestype=:scatter, label="Observed rabbits", markersize=3)
    plot!(p_data, t, foxes_obs, seriestype=:scatter, label="Observed foxes", markersize=3)
    plot!(p_data, t, true_rabbits, label="True rabbits", linewidth=2)
    plot!(p_data, t, true_foxes, label="True foxes", linewidth=2)
    xlabel!(p_data, "Time")
    ylabel!(p_data, "Population")
    title!(p_data, "Synthetic Data")
    savefig(p_data, "./imgs/MCMC/synthetic_data.png")
    
    initial_params = true_params .* (1 .+ randn(length(true_params)))  # Initial guess (perturbed true parameters)
    initial_params = max.(initial_params, [0.001, 0.00001, 0.001, 0.00001]) # Ensure parameters are positive
    
    println("Starting MCMC with initial parameters:")
    println("epsilon_rabbits: $(initial_params[1])")
    println("gamma_rabbits: $(initial_params[2])")
    println("epsilon_foxes: $(initial_params[3])")
    println("gamma_foxes: $(initial_params[4])")

    log_post = params -> log_posterior(params, t, rabbits_obs, foxes_obs)
    
    n_samples = 200000
    println("Running Metropolis-Hastings MCMC with $n_samples samples...")
    samples, acceptance_rate = metropolis_hastings(log_post, initial_params, n_samples, proposal_sd= 0.001 * initial_params)

    println("The initial parameters were:")
    println("epsilon_rabbits: $(initial_params[1])")
    println("gamma_rabbits: $(initial_params[2])")
    println("epsilon_foxes: $(initial_params[3])")
    println("gamma_foxes: $(initial_params[4])")

    # Discard burn-in period (first 20% of samples)
    burnin = Int(round(n_samples * 0.2))
    samples_post_burnin = samples[(burnin+1):end, :]
    
    println("Plotting results...")
    plot_results(t, rabbits_obs, foxes_obs, true_rabbits, true_foxes, samples_post_burnin)
        
    return samples_post_burnin
end

samples = main()
println("Done!")
