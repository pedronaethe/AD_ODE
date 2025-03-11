using DifferentialEquations
using ForwardDiff
using Plots
using StatsBase
using Distributions
using Random
using MCMCChains
using Dates

function rabbits_ODE(nrabbits, nfoxes, epsilon, gamma)
    return (epsilon - gamma * nfoxes) * nrabbits
end

function foxes_ODE(nrabbits, nfoxes, epsilon, gamma)
    return -(epsilon-gamma * nrabbits) * nfoxes
end

function system(nrabbits, nfoxes, epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes)
    f1 = rabbits_ODE(nrabbits, nfoxes, epsilon_rabbits, gamma_rabbits)
    f2 = foxes_ODE(nrabbits, nfoxes, epsilon_foxes, gamma_foxes)
    return [f1, f2]
end

function func(du, u, p, t)
    # Extract states and parameters
    nrabbits, nfoxes = u[1], u[2]
    epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes = p
    
    # Rabbits
    du[1] = (epsilon_rabbits - gamma_rabbits * nfoxes) * nrabbits
    # Foxes
    du[2] = -(epsilon_foxes - gamma_foxes * nrabbits) * nfoxes
end

function solve_system(p; u0=[1000.0, 20.0], tspan=(0.0, 100.0), solver=Midpoint(), saveat=1.0)
    """
    This function solves the system of ODEs with simpler state variables (just rabbits and foxes).
    
    Parameters:
    p - parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    u0 - initial conditions [rabbits0, foxes0]
    tspan - time span (start_time, end_time)
    solver - ODE solver algorithm
    saveat - time points to save at
    """
    prob = ODEProblem(func, u0, tspan, p)
    sol = solve(prob, solver, saveat=saveat)
    return sol
end

# Generate synthetic data
function generate_synthetic_data(true_params; noise_level=0.05, seed=498)
    """
    Generate synthetic data for the model.
    @true_params are the true parameters of the model.
    @noise_level is the standard deviation of the Gaussian noise.
    @seed is the random seed for reproducibility.
    
    Returns:
    @t - time points
    @rabbits_obs - observed rabbits (True values with noise)
    @foxes_obs - observed foxes (True values with noise)
    @rabbits - true rabbits
    @foxes - true foxes
    """
    println("Using seed: $seed")
    Random.seed!(seed)
    
    sol = solve_system(true_params, tspan=(0.0, 100.0), saveat=5.0)
    
    t = sol.t
    rabbits = sol[1, :]
    foxes = sol[2, :]
    
    rabbits_obs = rabbits .* (1 .+ noise_level * randn(length(rabbits)))
    foxes_obs = foxes .* (1 .+ noise_level * randn(length(foxes)))
    
    rabbits_obs = max.(rabbits_obs, 1.0)
    foxes_obs = max.(foxes_obs, 1.0)
    
    return t, rabbits_obs, foxes_obs, rabbits, foxes
end

using Distributions

# function log_likelihood(params, t, rabbits_obs, foxes_obs; sigma_rabbits=100.0, sigma_foxes=5.0, rho=0.5)
#     lower_bounds = [0.0, 0.0, 0.0, 0.0]
#     upper_bounds = [0.1, 0.001, 0.1, 0.001]
    
#     if any(params .< lower_bounds) || any(params .> upper_bounds)
#         println("Parameters out of bounds, returning -Inf", params)
#         return -Inf
#     end
    
#     try
#         sol = solve_system(params, tspan=(0.0, 100.0), saveat=t)
        
#         if sol.retcode != :Success
#             println("Integration failed, returning -Inf")
#             return -Inf
#         end
        
#         rabbits_pred = sol[1, :]
#         foxes_pred = sol[2, :]
        
#         Σ = [sigma_rabbits^2  rho * sigma_rabbits * sigma_foxes;
#              rho * sigma_rabbits * sigma_foxes  sigma_foxes^2] # Covariance matrix
        
#         ll = 0.0
#         for i in eachindex(t)
#             mu = [rabbits_pred[i], foxes_pred[i]]  # Mean vector
#             obs = [rabbits_obs[i], foxes_obs[i]]  # Observations at time i
#             ll += logpdf(MvNormal(mu, Σ), obs)  # Joint log-likelihood
#         end
        
#         return ll
#     catch
#         println("Integration failure unknown, returning -Inf")
#         return -Inf
#     end
# end

function log_likelihood(params, t, rabbits_obs, foxes_obs; sigma_rabbits=2.0, sigma_foxes=2.0)
    lower_bounds = [0.0, 0.0, 0.0, 0.0]
    upper_bounds = [0.1, 0.001, 0.1, 0.001]
    
    if any(params .< lower_bounds) || any(params .> upper_bounds)
        println("Parameters out of bounds, returning -Inf", params)
        return -Inf
    end
    
    try
        sol = solve_system(params, tspan=(0.0, 100.0), saveat=t)
        
        if sol.retcode != :Success
            println("Integration failed, returning -Inf")
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
    # Prior distributions for each parameter
    priors = [
        Uniform(0, 0.1),     # epsilon_rabbits
        Uniform(0, 0.001),   # gamma_rabbits
        Uniform(0.0, 0.1),     # epsilon_foxes
        Uniform(0.0, 0.001)    # gamma_foxes
    ]
    
    # Calculate log prior
    log_p = sum(logpdf.(priors, params))
    return log_p
end

# Log posterior (proportional to)
function log_posterior(params, t, rabbits_obs, foxes_obs)
    lp = log_prior(params)
    
    # If the parameters are outside the prior support, return -Inf
    if isinf(lp)
        return -Inf
    end
    
    # Otherwise, calculate the log-likelihood and add it to the log-prior
    ll = log_likelihood(params, t, rabbits_obs, foxes_obs)
    return lp + ll
end

# Metropolis-Hastings MCMC sampler
function metropolis_hastings(log_post_func, initial_params, n_samples; proposal_sd)
    # Initialize
    samples = zeros(n_samples, length(initial_params))
    current_params = copy(initial_params)
    current_log_post = log_post_func(current_params)
    
    # Check if initial point is valid
    if isinf(current_log_post)
        error("Initial point has zero posterior probability")
    end
    
    # Set first sample to initial parameters
    samples[1, :] = current_params
    
    # Acceptance counter
    n_accepted = 0
    
    # Main MCMC loop
    for i in 2:n_samples
        # Propose new parameters
        proposal_params = current_params + proposal_sd .* randn(size(current_params))

        
        # Calculate log posterior for proposal
        proposal_log_post = log_post_func(proposal_params)
        
        # Calculate acceptance probability
        alpha = min(1.0, exp(proposal_log_post - current_log_post))
        
        # Accept or reject
        if rand() < alpha
            current_params = proposal_params
            current_log_post = proposal_log_post
            n_accepted += 1
        end
        
        # Store current parameters
        samples[i, :] = current_params
        
        # Print progress
        if i % 500 == 0
            acceptance_rate = n_accepted / (i - 1)
            println("Iteration $i, acceptance rate: $(round(acceptance_rate, digits=3))")
        end
    end
    
    # Calculate final acceptance rate
    acceptance_rate = n_accepted / (n_samples - 1)
    println("Final acceptance rate: $(round(acceptance_rate, digits=3))")
    
    return samples, acceptance_rate
end
function plot_results(t, rabbits_obs, foxes_obs, true_rabbits, true_foxes, samples)
    # Get median and 95% credible intervals for parameters
    param_medians = median(samples, dims=1)[:]
    param_lower = [quantile(samples[:, i], 0.025) for i in 1:size(samples, 2)]
    param_upper = [quantile(samples[:, i], 0.975) for i in 1:size(samples, 2)]
    
    println("Parameter estimates (median and 95% CI):")
    param_names = ["epsilon_rabbits", "gamma_rabbits", "epsilon_foxes", "gamma_foxes"]
    for i in 1:length(param_names)
        println("$(param_names[i]): $(round(param_medians[i], digits=6)) ($(round(param_lower[i], digits=6)) - $(round(param_upper[i], digits=6)))")
    end
    
    # Predict with median parameters
    sol_median = solve_system(param_medians, tspan=(0.0, 100.0), saveat=t)
    rabbits_pred = sol_median[1, :]
    foxes_pred = sol_median[2, :]
    
    # Plot data and predictions with larger size
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
    
    # Plot MCMC trace with larger size
    p_trace = plot(samples, layout=(4, 1), labels=reshape(param_names, 1, 4), 
                 title=["Trace of epsilon_rabbits" "Trace of gamma_rabbits" "Trace of epsilon_foxes" "Trace of gamma_foxes"], size=(1200, 900))
    savefig(p_trace, "./imgs/MCMC/parameter_traces.png")
    
    # Plot parameter posterior distributions with larger size
    p_hist = histogram(samples, layout=(2, 2), labels=reshape(param_names, 1, 4),
                    title=["Posterior of epsilon_rabbits" "Posterior of gamma_rabbits" "Posterior of epsilon_foxes" "Posterior of gamma_foxes"], size=(1200, 900))
    savefig(p_hist, "./imgs/MCMC/parameter_posteriors.png")
    
    return p_combined, p_trace, p_hist
end

# Main execution
function main()
    # Define true parameters
    true_params = [0.015, 0.0001, 0.03, 0.0001]  # [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    
    println("Generating synthetic data with true parameters:")
    println("epsilon_rabbits: $(true_params[1])")
    println("gamma_rabbits: $(true_params[2])")
    println("epsilon_foxes: $(true_params[3])")
    println("gamma_foxes: $(true_params[4])")
    
    # Generate synthetic data
    #This function solves the system given the true_params. It adds noise to the solution to create synthetic observations.
    #It returns the time points, observed rabbits and foxes, and true rabbits and foxes.
    t, rabbits_obs, foxes_obs, true_rabbits, true_foxes = generate_synthetic_data(true_params, noise_level=0.1)
    
    # Plot synthetic data
    p_data = plot(t, rabbits_obs, seriestype=:scatter, label="Observed rabbits", markersize=3)
    plot!(p_data, t, foxes_obs, seriestype=:scatter, label="Observed foxes", markersize=3)
    plot!(p_data, t, true_rabbits, label="True rabbits", linewidth=2)
    plot!(p_data, t, true_foxes, label="True foxes", linewidth=2)
    xlabel!(p_data, "Time")
    ylabel!(p_data, "Population")
    title!(p_data, "Synthetic Data")
    savefig(p_data, "./imgs/MCMC/synthetic_data.png")
    
    # Initial guess (perturbed true parameters)
    initial_params = true_params .* (1 .+ randn(length(true_params)))
    # Ensure non-negative values
    initial_params = max.(initial_params, [0.001, 0.00001, 0.001, 0.00001])

    println("Starting MCMC with initial parameters:")
    println("epsilon_rabbits: $(initial_params[1])")
    println("gamma_rabbits: $(initial_params[2])")
    println("epsilon_foxes: $(initial_params[3])")
    println("gamma_foxes: $(initial_params[4])")
    
    # Create log_posterior function with fixed data
    log_post = params -> log_posterior(params, t, rabbits_obs, foxes_obs)
    
    # Run MCMC
    n_samples = 200000
    println("Running Metropolis-Hastings MCMC with $n_samples samples...")
    samples, acceptance_rate = metropolis_hastings(log_post, initial_params, n_samples, proposal_sd= 0.002 * initial_params)
    println("epsilon_rabbits: $(initial_params[1])")
    println("gamma_rabbits: $(initial_params[2])")
    println("epsilon_foxes: $(initial_params[3])")
    println("gamma_foxes: $(initial_params[4])")
    # Discard burn-in period (first 20% of samples)
    burnin = Int(round(n_samples * 0.2))
    samples_post_burnin = samples[(burnin+1):end, :]
    
    # Plot results
    println("Plotting results...")
    plot_results(t, rabbits_obs, foxes_obs, true_rabbits, true_foxes, samples_post_burnin)
        
    return samples_post_burnin
end

# Run the analysis
samples = main()
println("Done! Check the output figures for results.")
