using DifferentialEquations
using ForwardDiff
using Plots
using StatsBase
using Distributions
using Random
using LinearAlgebra
using Printf
using LineSearches



"""
This file is responsible for the CG analysis of the rabbits and foxes model using a ForwardDiff package to calculate the gradient of the cost function.
"""
# Physical constants in cgs units
const H_PLANCK = 6.62607015e-27
const C_LIGHT  = 2.99792458e10
const K_BOLTZ  = 1.380649e-16

function radtransODE(I_nu, nu, Teff)
    """
    Ode responsible to calculate the intensity of the radiation as it travels through the medium.
    Parameters:
    @Teff: effective temperature of the black body.
    @nu: frequency of the radiation.
    @I_nu: intensity of the radiation.

    Observations:
    dI_nu/ds = - I_nu + B_nu
    """
    return - I_nu + planckFunction(nu, Teff)
end

function planckFunction(nu, Teff)
    """
    Absorption coefficient calculation for the radiation for the optically thick case.
    Parameters:
    @nu: frequency of the radiation.
    @Teff: effective temperature of the black body.
    
    Observations:
    The absorption coefficient is calculated using the Planck function and the frequency of the radiation.
    """
    B_nu = (2 * h * nu^3) / (c^2 * (exp(h * nu / (k * Teff)) - 1))

    return B_nu
end


function problemFunc(u,du,p)
    """
    Function to calculate the derivatives of the intensity of the radiation through autodiff.
    Parameters:
    @nu: frequency of the radiation.
    @Teff: effective temperature of the black body.
    @I_nu: intensity of the radiation.
    @du: derivative of the state variables.
    @u: state variables.
    @p: parameters of the model [nu, Teff].

    Observations:
    The indexes are organized as following
    @du = [
        1 - dI_nu/ds,       2 - d(dI_nu/dnu)/ds,
        3 - d(dI_nu/dTeff)/ds   
    ]

    @u = [
        1 - I_nu,           2 - dI_nu/dnu,
        3 - dI_nu/dTeff
    ]

    @jacobian_matrix format is:
    |df1/dI_nu df1/dnu df1/dTeff|
    (1 x 3)
    """
    # Calculate the absorption coefficient
    jacobian_matrix = ForwardDiff.jacobian(x -> radtransODE(x[1], x[2], x[3]), [u[1], p[1], p[2]])
    # Calculate the derivatives using autodiff
    du[1] = radtransODE(u[1], p[1], p[2])
    du[2] = jacobian_matrix[1, 1] * u[2] + jacobian_matrix[1, 2]
    if(jacobian_matrix[1,1] == 0)
        println("jacobian_matrix[1,1] == 0")
    end
    du[3] = jacobian_matrix[1, 1] * u[3] + jacobian_matrix[1, 3]
    # Return the derivatives
    return dI_nu, dTeff, dnu
end

function solveODE(p)
    """
    Function to solve the ODE
    Parameters:
    @p: parameters of the model [nu, Teff].

    Variables:
    @nu: frequency of the radiation.
    @Teff: effective temperature of the black body.
    @u0: initial conditions.
    @path_span: path of the radiation.
    @prob: ODEProblem.
    @sol: solution of the ODE system.

    Observations:
    @u0 = [
        1 - I_nu,           2 - dI_nu/dnu,
        3 - dI_nu/dTeff
    ]

    """
    # Initial conditions
    u0 = [100, 0.0, 0.0]
    path_span = (0.0, 100.0)
    
    # Define the ODE problem
    prob = ODEProblem(problemFunc, u0, path_span, p)
    
    # Solve the ODE problem
    sol = solve(prob, Rodas5P(), saveat=5.0)
    
    return sol
end


function generate_synthetic_data(true_params; noise_level=0.05, seed=498)
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

    println("Using seed: $seed")
    Random.seed!(seed)
    sol = solve_system(true_params)
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




function plot_results(t, rabbits_obs, foxes_obs, true_rabbits, true_foxes, pred_params, history, cost_history)
    """
    Plot the results of the CG analysis.

    Parameters:
    @t - time points
    @rabbits_obs - observed rabbits (True + noise)
    @foxes_obs - observed foxes (True + noise)
    @true_rabbits - true rabbits (True + noise)
    @true_foxes - true foxes (True + noise)
    @pred_params - estimated parameters from CG
    @history - history of the parameters during the CG analysis
    @cost_history - history of the cost function during the CG analysis

    Variables:
    @sol_pred - solution of the ODE system with the estimated parameters
    @rabbits_pred - predicted rabbits
    @foxes_pred - predicted foxes
    @p1 - plot of the rabbits
    @p2 - plot of the foxes
    @p_combined - combined plot of the rabbits and foxes
    @p_cost_history - plot of the cost function history
    @param_names - names of the parameters
    @p_history - plot of the parameter history

    """
    
    # Solve system with median parameters
    #println("Predicted parameters: $(pred_params)")
    sol_pred = solve_system(pred_params)
    rabbits_pred = sol_pred[1, :]
    foxes_pred = sol_pred[2, :]
    
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
    savefig(p_combined, "./imgs/CG/population_fit.png")

    p_cost_history = plot(cost_history, xlabel="Iterations", ylabel="Quadratic error", title="Predicted vs Data Error History", size=(1200, 900))
    savefig(p_cost_history, "./imgs/CG/cost_history.png")

    param_names = ["epsilon_rabbits", "gamma_rabbits", "epsilon_foxes", "gamma_foxes"]
    p_history = plot(size=(1200, 900), layout=(4, 1))
    for i in 1:4
        plot!(p_history[i], 1:length(history), getindex.(history, i), xlabel="Iterations", ylabel=param_names[i], label=param_names[i])
    end
    savefig(p_history, "./imgs/CG/parameter_history.png")
end

# Cost function - Mean squared error between model and data
function cost_function(params, rabbits_obs, foxes_obs, t)
    """
    Cost function to minimize. The closer to zero, better the predicted parameters.
    
    Parameters:
    @params - parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @rabbits_obs - observed rabbits
    @foxes_obs - observed foxes
    @t - time points

    Variables:
    @sol - solution of the ODE system
    @nrabbits - predicted rabbits
    @nfoxes - predicted foxes
    @error - mean squared error

    Observations:
    The cost function is the mean squared error between the observed and predicted populations of rabbits and foxes.
    """
    sol = solve_system(params)
    nrabbits = sol[1, :]
    nfoxes = sol[2, :]
    
    if sol.retcode != :Success
        return Inf
    end
    
    error = sum((nrabbits .- rabbits_obs).^2) / length(rabbits_obs) + sum((nfoxes .- foxes_obs).^2) / length(foxes_obs)
    
    return error
end


function numerical_gradient(params, rabbits_obs, foxes_obs)
    """
    Calculate the numerical gradient of the cost function.
    
    Parameters:
    @fcost - cost function
    @params - parameters

    Observations:
    Since the gradient will point to the direction of greatest increase, we will set this to a negative value in order to point to the direction of error decrease.
    This function uses AD to calculate the gradient of the cost function.
    """
    sol = solve_system(params)
    
    rabbits = sol[1, :]
    foxes = sol[2, :]
    drabbits_de1 = sol[3, :]
    dfoxes_de1 = sol[4, :]
    drabbits_dg1 = sol[5, :]
    dfoxes_dg1 = sol[6, :]
    drabbits_de2 = sol[7, :]
    dfoxes_de2 = sol[8, :]
    drabbits_dg2 = sol[9, :]
    dfoxes_dg2 = sol[10, :]

    dcost_drabbits = 2 * (rabbits .- rabbits_obs)/length(rabbits_obs)
    dcost_dfoxes = 2* (foxes .- foxes_obs)/length(foxes_obs)

    grad = zeros(Float64, 4)
    grad[1] = sum(dcost_drabbits .* drabbits_de1) + sum(dcost_dfoxes .* dfoxes_de1)
    #println("grad1:sum 1: $(sum(dcost_drabbits .* drabbits_de1)), sum 2: $(sum(dcost_dfoxes .* dfoxes_de1)), grad[1]: $(grad[1])")
    grad[2] = sum(dcost_drabbits .* drabbits_dg1) + sum(dcost_dfoxes .* dfoxes_dg1)
    #println("grad2:sum 1: $(sum(dcost_drabbits .* drabbits_dg1)), sum 2: $(sum(dcost_dfoxes .* dfoxes_dg1)), grad[2]: $(grad[2])")
    grad[3] = sum(dcost_drabbits .* drabbits_de2) + sum(dcost_dfoxes .* dfoxes_de2)
    #println("grad3:sum 1: $(sum(dcost_drabbits .* drabbits_de2)), sum 2: $(sum(dcost_dfoxes .* dfoxes_de2)), grad[3]: $(grad[3])")
    grad[4] = sum(dcost_drabbits .* drabbits_dg2) + sum(dcost_dfoxes .* dfoxes_dg2)
    #println("grad4:sum 1: $(sum(dcost_drabbits .* drabbits_dg2)), sum 2: $(sum(dcost_dfoxes .* dfoxes_dg2)), grad[4]: $(grad[4])")

   
    return grad
end

function line_search(fcost, current_params, direction, grad, initial_step=1., α=1e-4, β=0.7, max_iter=500)
    """
    Backtracking line search with Armijo condition to find optimal step size. 
    Parameters:
    @fcost - cost function
    @current_params - current parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @direction - direction of the gradient
    @initial_step - initial step size
    @α - parameter for Armijo condition
    @β - parameter for step size reduction
    @max_iter - maximum number of iterations

    Variables:
    @step - step size
    @fcost_x - cost function value at current parameters
    @min_params - minimum values that parameters can take
    @max_params - maximum values that parameters can take
    @fcost_new - cost function value at new parameters
    @new_params - current_parameters + step * direction

    Observations:
    Main idea taken from: https://en.wikipedia.org/wiki/Backtracking_line_search

    This function guesses a high initial step value and keeps reducing it (by multiplying by β ∈ [0,1[) until the Armijo condition is satisfied.

    This was one of the "easiest" lines search algorithms to implement and understand. It is based on the Armijo condition. I've read that maybe the optimization package
    in Julia has a better line search algorithm, but I wanted to implement this one myself to understand it better.

    The Armijo condition compares the cost function at the new parameters with the cost function at the old parameters plus a small step in the direction of the gradient.
    I think this avoids the algorithm from taking too large steps and overshooting the minimum.

    This function sometimes, depending on the step it takes, may get to a point where the parameters are out of bounds. In this case, the solver was having trouble converging due to the stiffness of the problem.
    """
    step = initial_step
    fcost_x = fcost(current_params)
    min_params = [1e-6,1e-6,1e-6,1e-6]
    max_params = [0.1,0.1,0.1,0.1]
    fcost_new = 0.0
    for _ in 1:max_iter
        new_params = current_params + step * direction
        
        if (new_params > max_params || new_params < min_params)
            step *= β
            continue
        end
        fcost_new = fcost(new_params)
        
        # Armijo condition: f(x + step*d) <= f(x) + α * step * dot(grad, direction)
        if fcost_new <= fcost_x + α * step * dot(grad, direction)
            break
        else
            step *= β
        end
    end
    new_params = current_params + step * direction
    for i in 1:length(new_params)
        if new_params[i] > max_params[i]
            new_params[i] = max_params[i]
        elseif new_params[i] < min_params[i]
            new_params[i] = min_params[i]
        end
    end


    
    return step, new_params, fcost_new 
end


function line_search_pkg(fcost, current_params, direction, rabbits_obs, foxes_obs, initial_step=1.)
    """
    Line search using the LineSearches package to find the optimal step size.

    Parameters:
    @fcost - cost function
    @current_params - current parameters at each iteration of CG algorithm [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @direction - direction of the - gradient
    @initial_step - initial step size

    Variables:
    @min_params - minimum values that parameters can take
    @max_params - maximum values that parameters can take
    @ϕ - objective function along the line
    @dϕ - derivative of the objective function along the line
    @dϕ0 - directional derivative at the initial point
    @α0 - initial step size
    @ϕ0 - objective function value at the initial point
    @grad0 - gradient at the initial point
    @step_size - optimal step size
    @new_params - new parameters after updating (current_parameters + step_size * direction)

    Observations:
    This function uses the LineSearches package to find the optimal step size. This package has several line search algorithms implemented: Static, BackTracking, HagerZhang, MoreThuente, StrongWolfe.
    I've chosen the BackTracking algorithm because it is the one that I've implemented in the line_search function.
    
    I based myself on the simple one-dimensional algorithm in the https://github.com/JuliaNLSolvers/LineSearches.jl page.

    """
    min_params = [1e-6,1e-6,1e-6,1e-6]
    max_params = [0.1,0.1,0.1,0.1]

    ϕ(t) = fcost(current_params + t * direction)
    function dϕ(t)
        grad = numerical_gradient(current_params + t * direction, rabbits_obs, foxes_obs)
        return dot(grad, direction)
    end
    
    function ϕdϕ(t)
        f_val = ϕ(t)
        df_val = dϕ(t)
        return f_val, df_val
    end
    
    α0 = initial_step
    ϕ0 = fcost(current_params)
    grad0 = numerical_gradient(current_params, rabbits_obs, foxes_obs)
    dϕ0 = dot(grad0, direction)
    
    step_size, f_val = (BackTracking())(ϕ, dϕ, ϕdϕ, α0, ϕ0, dϕ0)
    new_params = current_params + step_size * direction

    #In case the step size is too large, we reduce it by half until the parameters are within the bounds
    while(any(new_params .> max_params) || any(new_params .< min_params))
        step_size *= 0.5
        new_params = current_params + step_size * direction
    end

    return step_size, new_params, f_val

end


function conjugate_gradient(cost_func, initial_params, rabbits_obs, foxes_obs, max_iter=200, tol=1e-6)
    """
    Conjugate gradient optimization algorithm.

    Parameters:
    @cost_func - cost function
    @initial_params - initial parameters [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @max_iter - maximum number of iterations
    @tol - tolerance for convergence

    Variables:
    @old_params - parameter in the previous iteration
    @grad - gradient of the cost function
    @direction - direction of the gradient
    @history - history of the parameters
    @cost_history - history of the cost function
    @min_params - minimum values that parameters can take
    @max_params - maximum values that parameters can take
    @step_size - step size as calculated by line search function
    @new_params - new parameters after updating (current_parameters + step_size * direction). Sometimes these parameters are not accepted
    @grad_new - gradient of the cost function at the new parameters
    @denom - dot product of the gradient at the old parameters
    @beta - Fletcher-Reeves formula for updating the direction
    @iter - iteration number so far

    Observations:
    Main idea here is to keep updating the parameters in the -gradient direction of the cost function (direction of decrease) until convergence.

    The Fletcher-Reeves formula is used to update the direction of the gradient. This formula is used to avoid the zig-zagging of the gradient descent algorithm. 
    It is used in non-linear conjugate gradient methods. More information can be found here: https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
    This way, the new direction is a combination of the gradient at the new parameters and the gradient at the old parameters.
    """

    old_params = copy(initial_params)
    grad = numerical_gradient(old_params, rabbits_obs, foxes_obs)
    direction = -grad
    
    history = [old_params]
    cost_history = [cost_func(old_params)]
    for iter in 1:max_iter
        if(true)
            #use line search package function
            step_size, new_params, cost_new= line_search(cost_func, old_params, direction, grad)
        else
            #use line search handmaid function 
            step_size, new_params, cost_new= line_search_pkg(cost_func, old_params, direction, rabbits_obs, foxes_obs)
        end
        grad_new = numerical_gradient(new_params, rabbits_obs, foxes_obs)
        
        denom = dot(grad, grad)
        # Fletcher-Reeves formula
        if denom > 1e-10
            beta = dot(grad_new, grad_new) / denom
        else
            beta = 0
        end
                    
        direction = -grad_new + beta * direction
            
        old_params = new_params
        grad = grad_new
            
        push!(history, old_params)
        push!(cost_history, cost_new)

        
        if(cost_func(old_params) < 1)
            println("Converged in $iter iterations")
            break
        end

        if (norm(grad) < tol)
            println("Converged in $iter iterations")
            break
        end
        
        if iter % 1 == 0
            println("Iteration $iter: Cost = $(cost_func(old_params)), Gradient norm = $(norm(grad))", " step size: $step_size")
            println("Parameters: $old_params")
            println("gradient: $grad")
        end
    end
    
    return old_params, history, cost_history
end

function gradient_descent(cost_func, initial_params, rabbits_obs, foxes_obs, max_iter=200, tol=1e-6)
    gamma = 1.e-18
    old_params = initial_params
    grad = numerical_gradient(old_params, rabbits_obs, foxes_obs)
    for iter in 1:max_iter
        new_params = old_params - gamma * grad
        grad_new = numerical_gradient(new_params, rabbits_obs, foxes_obs)

        if (norm(grad_new) < tol)
            println("Converged in $iter iterations")
            break
        end
        if (cost_func(new_params) < 1)
            println("Converged in $iter iterations")
            break
        end

        if(cost_func(new_params) > cost_func(old_params))
            println("Cost function increased")
        end

        if iter % 1 == 0
            println("Iteration $iter: Cost = $(cost_func(old_params)), Gradient norm = $(norm(grad))", " step size: $gamma")
            println("Parameters: $old_params")
            println("gradient: $grad \n")
        end
        #calculate gamma according to barzlai-borwein methods
        gamma = norm((new_params .- old_params) .* (grad_new .- grad))/ (norm(grad_new .- grad)^2)
        # Update parameters
        old_params = new_params
        grad = grad_new

    end
end

# Main execution
function main()
    """
    Main function to run the CG analysis

    Variables:
    @true_params - true parameters of the model [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes]
    @t - time points
    @rabbits_obs - observed rabbits (True + noise)
    @foxes_obs - observed foxes (True + noise)
    @true_rabbits - true rabbits 
    @true_foxes - true foxes
    @initial_params - initial parameter values
    @cost_func - cost function that calculate the error between the observed and predicted populations
    @n_samples - number of CG iterations
    @estimated_params - estimated parameters from CG
    @history - history of the parameters during the CG analysis
    @cost_history - history of the cost function during the CG analysis
    @param_names - names of the parameters
    @mape - mean absolute percentage error


    Observations:
    The workflow of this function is as follows
        1. Generate synthetic data by solving the system with true parameters and adding Gaussian noise.
        2. Plot the synthetic data.
        3. Generate initial guess by adding noise to the true parameters
        4. Assign the cost_function function to a variable and pass this to CG sampler so I can easily change the cost function if I want to.
        5. Run the CG algorithm.
        6. Plot the results.
    """


    true_params = [0.015, 0.0001, 0.03, 0.0001]
    noise_level = 0.0
    
    println("Generating synthetic data with true parameters:")
    println("epsilon_rabbits: $(true_params[1])")
    println("gamma_rabbits: $(true_params[2])")
    println("epsilon_foxes: $(true_params[3])")
    println("gamma_foxes: $(true_params[4])")
    
    t, rabbits_obs, foxes_obs, true_rabbits, true_foxes = generate_synthetic_data(true_params, noise_level=noise_level)
    println("Done!")
    # Plotting synthetic data so we can easily compare perturbed to unperturbed data
    p_data = plot(t, rabbits_obs, seriestype=:scatter, label="Observed rabbits", markersize=3)
    plot!(p_data, t, foxes_obs, seriestype=:scatter, label="Observed foxes", markersize=3)
    plot!(p_data, t, true_rabbits, label="True rabbits", linewidth=2)
    plot!(p_data, t, true_foxes, label="True foxes", linewidth=2)
    xlabel!(p_data, "Time")
    ylabel!(p_data, "Population")
    title!(p_data, "Synthetic Data")
    savefig(p_data, "./imgs/CG/synthetic_data.png")
    
    initial_params = true_params .* (1 .+ randn(length(true_params)))  # Initial guess (perturbed true parameters)
    initial_params = max.(initial_params, [0.001, 0.00001, 0.001, 0.00001]) # Ensure parameters are positive
    println("Starting CGs with initial parameters:")
    println("epsilon_rabbits: $(initial_params[1])")
    println("gamma_rabbits: $(initial_params[2])")
    println("epsilon_foxes: $(initial_params[3])")
    println("gamma_foxes: $(initial_params[4])")
    #rabbits_obs = true_rabbits
    #foxes_obs = true_foxes
    
    cost_func = params -> cost_function(params, rabbits_obs, foxes_obs, t)
    n_samples = 200000
    println("Running Conjugate Gradient approach with $n_samples samples...")
    estimated_params, history, cost_history  = conjugate_gradient(cost_func, initial_params, rabbits_obs, foxes_obs, n_samples)

    #if you want to gradient descent, uncomment the two lines below and comment the two lines above
    #println("Running Gradient descent approach with $n_samples samples...")
    #gradient_descent(cost_func, initial_params, rabbits_obs, foxes_obs, n_samples)
    println("Done!")
    # Print results
    param_names = ["epsilon_rabbits", "gamma_rabbits", "epsilon_foxes", "gamma_foxes"]

    println("\nParameter Estimation Results:")
    println("Parameter   | True Value | Initial Guess | Estimated Value | Error (%)")
    println("------------------------------------------------------------------------")
    for i in 1:length(true_params)
        error_percent = abs(estimated_params[i] - true_params[i]) / true_params[i] * 100
        println(
            @sprintf("%-12s | %10.6f | %13.6f | %15.6f | %8.2f%%", 
                     param_names[i], true_params[i], initial_params[i], estimated_params[i], error_percent)
        )
    end
    
    # Calculate mean absolute percentage error
    mape = mean(abs.(estimated_params .- true_params) ./ true_params) * 100
    println("\nMean Absolute Percentage Error: $(round(mape, digits=2))%")
    println("Noise level used: $(noise_level * 100)%")
    println("Number of iterations: $(length(cost_history))")
    println("Final cost: $(cost_history[end])")
    
    println("Plotting results...")
    plot_results(t, rabbits_obs, foxes_obs, true_rabbits, true_foxes, estimated_params, history, cost_history)
end
        

samples = main()
println("Done!")
