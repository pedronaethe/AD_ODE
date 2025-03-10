using Plots
using ForwardDiff



function rabbits_ODE(nrabbits, nfoxes, epsilon, gamma)
    return (epsilon - gamma * nfoxes) * nrabbits
end

function foxes_ODE(nrabbits, nfoxes, epsilon, gamma)
    return -(epsilon-gamma * nrabbits) * nfoxes
end


# Explicit Euler method
function euler_method(t0, tf, h, nrabbits0, nfoxes0)

    """This function solves the system of ODEs using the explicit Euler method.
        The derivatives are organized as the following matrix:
        
        |dy1/de1 dy1/dgamma1 dy1/de2 dy1/dgamma2|
        |dy2/de1 dy2/dgamma1 dy2/de2 dy2/dgamma2|
        2 x 4

        For the dy_dy0 array
        |dy1/dy10 dy1/dy20|
        |dy2/dy10 dy2/dy20|

    """
    t::Float64 = t0
    rabbits::Float64 = nrabbits0      
    foxes::Float64 = nfoxes0     
    drabbits_deps::Float64 = 0.0
    dfoxes_deps::Float64 = 0.0

    epsilon_rabbits::Float64 = 0.015
    gamma_rabbits::Float64 = 0.0001
    epsilon_foxes::Float64 = 0.03
    gamma_foxes::Float64 = 0.0001


    t_values = Float64[]  
    rabbits_arr = Float64[]  
    foxes_arr = Float64[]  
    dy_dp = zeros(Float64, 2, 4, Int((tf - t0) / h) + 1)
    dy_dy0 = zeros(Float64, 2, 2, Int((tf - t0) / h) + 1)
    dy_dy0[1,1,1] = 1.0
    dy_dy0[2,2,1] = 1.0

    push!(rabbits_arr, rabbits)  
    push!(foxes_arr, foxes)  
    push!(t_values, t)
    step = 2
    # Integrate until t reaches tf
    while t < tf
        # rabbits derivative with respect to epsilon_rabbits
        dy_dp[1,1,step] = dy_dp[1,1,step - 1] +  h * (ForwardDiff.derivative(diff_val -> rabbits_ODE(diff_val, foxes, epsilon_rabbits, gamma_rabbits), rabbits) *  dy_dp[1,1,step-1]
        +ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, foxes, diff_val, gamma_rabbits), epsilon_rabbits) 
        + ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, diff_val, epsilon_rabbits, gamma_rabbits), foxes) * dy_dp[2,1,step-1])

        # # rabbits derivative with respect to gamma_rabbits
        dy_dp[1,2,step] = dy_dp[1,2,step - 1] + h * (ForwardDiff.derivative(diff_val -> rabbits_ODE(diff_val, foxes, epsilon_rabbits, gamma_rabbits), rabbits) *  dy_dp[1,2,step-1]
        + ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, foxes, epsilon_rabbits, diff_val), gamma_rabbits)
        + ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, diff_val, epsilon_rabbits, gamma_rabbits), foxes) *  dy_dp[2,2,step-1])
        
        # # rabbits derivative with respect to epsilon_foxes
        dy_dp[1,3,step] = dy_dp[1,3,step - 1] + h * (ForwardDiff.derivative(diff_val -> rabbits_ODE(diff_val, foxes, epsilon_rabbits, gamma_rabbits), rabbits) *  dy_dp[1,3,step-1] + 
        ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, diff_val, epsilon_rabbits, gamma_rabbits), foxes) * dy_dp[2,3,step-1])

        # # rabbits derivative with respect to gamma_foxes
        dy_dp[1,4,step] = dy_dp[1,4,step - 1] + h * (ForwardDiff.derivative(diff_val -> rabbits_ODE(diff_val, foxes, epsilon_rabbits, gamma_rabbits), rabbits) *  dy_dp[1,4,step-1] + 
        ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, diff_val, epsilon_rabbits, gamma_rabbits), foxes) * dy_dp[2,4,step-1])

        # foxes derivative with respect to epsilon_rabbits
        dy_dp[2,1,step] = dy_dp[2,1,step - 1] + h * (ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, diff_val, epsilon_foxes, gamma_foxes), foxes) *  dy_dp[2,1,step-1] + 
        ForwardDiff.derivative(diff_val -> foxes_ODE(diff_val, foxes, epsilon_foxes, gamma_foxes), rabbits) *  dy_dp[1,1,step-1])

        # foxes derivative with respect to gamma_rabbits
        dy_dp[2,2,step] = dy_dp[2,2,step - 1] + h * (ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, diff_val, epsilon_foxes, gamma_foxes), foxes) *  dy_dp[2,2,step-1] +
        ForwardDiff.derivative(diff_val -> foxes_ODE(diff_val, foxes, epsilon_foxes, gamma_rabbits), rabbits) * dy_dp[1,2,step-1])
                

        # # foxes derivative with respect to epsilon_foxes
        dy_dp[2,3,step] = dy_dp[2,3,step - 1] + h * (ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, diff_val, epsilon_foxes, gamma_foxes), foxes) *  dy_dp[2,3,step-1] + 
        ForwardDiff.derivative(diff_val -> foxes_ODE(diff_val, foxes, epsilon_foxes, gamma_foxes), rabbits) * dy_dp[1,3,step-1] +
        ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, foxes, diff_val, gamma_foxes), epsilon_foxes))

        # # foxes derivative with respect to gamma_foxes
        dy_dp[2,4,step] = dy_dp[2,4,step - 1] + h * (ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, diff_val, epsilon_foxes, gamma_foxes), foxes) *  dy_dp[2,4,step-1] +
        ForwardDiff.derivative(diff_val -> foxes_ODE(diff_val, foxes, epsilon_foxes, gamma_foxes), rabbits) * dy_dp[1,4,step-1] +
        ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, foxes, epsilon_foxes, diff_val), gamma_foxes))

        dy_dy0[1,1,step] = dy_dy0[1,1,step - 1] + h * (ForwardDiff.derivative(diff_val -> rabbits_ODE(diff_val, foxes, epsilon_rabbits, gamma_rabbits), rabbits) * dy_dy0[1,1,step-1] +
        ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, diff_val, epsilon_rabbits, gamma_rabbits), foxes) * dy_dy0[2,1,step-1])

        dy_dy0[2,1,step] = dy_dy0[2,1,step - 1] + h * (ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, diff_val, epsilon_foxes, gamma_foxes), foxes) * dy_dy0[2,1,step-1] +
        ForwardDiff.derivative(diff_val -> foxes_ODE(diff_val, foxes, epsilon_foxes, gamma_foxes), rabbits) * dy_dy0[1,1,step-1])

        dy_dy0[1,2,step] = dy_dy0[1,2,step - 1] + h * (ForwardDiff.derivative(diff_val -> rabbits_ODE(diff_val, foxes, epsilon_rabbits, gamma_rabbits), rabbits) * dy_dy0[1,2,step-1] +
        ForwardDiff.derivative(diff_val -> rabbits_ODE(rabbits, diff_val, epsilon_rabbits, gamma_rabbits), foxes) * dy_dy0[2,2,step-1])

        dy_dy0[2,2,step] = dy_dy0[2,2,step - 1] + h * (ForwardDiff.derivative(diff_val -> foxes_ODE(rabbits, diff_val, epsilon_foxes, gamma_foxes), foxes) * dy_dy0[2,2,step-1] +
        ForwardDiff.derivative(diff_val -> foxes_ODE(diff_val, foxes, epsilon_foxes, gamma_foxes), rabbits) * dy_dy0[1,2,step-1])
        
        
        rabbits += h * rabbits_ODE(rabbits, foxes, epsilon_rabbits, gamma_rabbits)
        foxes += h * foxes_ODE(rabbits, foxes, epsilon_foxes, gamma_foxes)

        t += h
        step += 1
       # println("t: ", t, " rabbits: ", rabbits, " foxes: ", foxes)
        push!(rabbits_arr, rabbits)
        push!(t_values, t)
        push!(foxes_arr, foxes)
    end

    return t_values, rabbits_arr, foxes_arr, dy_dp, dy_dy0  # Return the time and solution values
end



function main()

    nrabbits0::Float64 = 1000.0
    nfoxes0::Float64 = 20.0
    t0::Float64 = 0.0   # Initial time
    tf::Float64 = 1000.0   # Final time
    step::Float64 = 0.1    # Time step size

    dy_dp = zeros(Float64, 2, 4, Int((tf - t0) / step) + 1)
    dy_dy0 = zeros(Float64, 2, 2, Int((tf - t0) / step) + 1)

    # Call the Euler method
    t_values, rabbits_arr, foxes_arr, dy_dp, dy_dy0 = euler_method(t0, tf, step, nrabbits0, nfoxes0)
    
    # Plot the results
    p = plot(t_values, rabbits_arr, label="rabbits Population(t)", xlabel="Time (t)", ylabel="Population", title="Explicit Euler Method: Y vs t", color = "blue")
    plot!(p, t_values, foxes_arr, label="Foxes Population(t)", xlabel="Time (t)", ylabel="Population", color = "lime")
    savefig(p, "./imgs/population.png")  # Save the plot as a PNG file
    p = plot(t_values, dy_dp[1,1,:], label="drabbits/de1 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim=(-1.5e6, 1.5e6), color = "red")
    plot!(p, t_values, dy_dp[2,1,:], label="dFoxes/de1 (t)", xlabel="Time (t)", ylabel="Derivative", color = "light blue")
    savefig(p, "./imgs/derivative_e1.png")  # Save the plot as a PNG file
    p = plot(t_values, dy_dp[1,2,:], label="drabbits/dgamma1 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim =(-4e7, 3e7), color = "pink")
    plot!(p, t_values, dy_dp[2,2,:], label="dFoxes/dgamma1 (t)", xlabel="Time (t)", ylabel="Derivative", color = "yellow")
    savefig(p, "./imgs/derivative_gamma1.png")  # Save the plot as a PNG file

    p = plot(t_values, dy_dp[1,3,:], label="drabbits/de2 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-1.5e6, 1e6), color = "red")
    plot!(p, t_values, dy_dp[2,3,:], label="dFoxes/de2 (t)", xlabel="Time (t)", ylabel="Derivative", color = "light blue")
    savefig(p, "./imgs/derivative_e2.png")  # Save the plot as a PNG file

    p = plot(t_values, dy_dp[1,4,:], label="drabbits/dgamma2 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-2e8, 2e8), color = "pink")
    plot!(p, t_values, dy_dp[2,4,:], label="dFoxes/dgamma2 (t)", xlabel="Time (t)", ylabel="Derivative", color = "yellow")
    savefig(p, "./imgs/derivative_gamma2.png")  # Save the plot as a PNG file

    p = plot(t_values, dy_dy0[1,1,:], label="drabbits/drabbits0 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-20, 20), color = "red")
    plot!(p, t_values, dy_dy0[2,1,:], label="dFoxes/drabbits0 (t)", xlabel="Time (t)", ylabel="Derivative", color = "light blue")
    savefig(p, "./imgs/derivative_rabbits0.png")  # Save the plot as a PNG file

    p = plot(t_values, dy_dy0[1,2,:], label="drabbits/dfoxes0 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-200, 200), color = "pink")
    plot!(p, t_values, dy_dy0[2,2,:], label="dFoxes/dfoxes0 (t)", xlabel="Time (t)", ylabel="Derivative", color = "yellow")
    savefig(p, "./imgs/derivative_foxes0.png")  # Save the plot as a PNG file

end

main()
