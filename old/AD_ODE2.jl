using Plots
using ForwardDiff

"""
This is an old file that solves the system of ODEs using a explicit Euler method made by myself.
The system of ODEs is the Lotka-Volterra model with two parameters for each species.
"""

function rabbits_ODE(nrabbits, nfoxes, epsilon, gamma)
    """
    ODE responsible to calculate the rabbits population over time.

    Parameters:
    @nrabbits: number of rabbits.
    @nfoxes: number of foxes.
    @epsilon: parameter that controls the growth of the rabbits population.
    @gamma: parameter that controls the decrease of the rabbits population.
    """
    return (epsilon - gamma * nfoxes) * nrabbits
end

function foxes_ODE(nrabbits, nfoxes, epsilon, gamma)
    """
    ODE responsible to calculate the foxes population over time.

    Parameters:
    @nrabbits: number of rabbits.
    @nfoxes: number of foxes.
    @epsilon: parameter that controls the growth of the foxes population.
    @gamma: parameter that controls the decrease of the foxes population.
    """

    return -(epsilon-gamma * nrabbits) * nfoxes
end

function system(nrabbits, nfoxes, epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes)
    """
    This function packs the system of ODEs so I can pass this as a sole function to the ForwardDiff.jacobian.
    
    Parameters:
    @nrabbits: number of rabbits.
    @nfoxes: number of foxes.
    @epsilon_rabbits: parameter that controls the growth of the rabbits population.
    @gamma_rabbits: parameter that controls the decrease of the rabbits population.
    @epsilon_foxes: parameter that controls the growth of the foxes population.
    @gamma_foxes: parameter that controls the decrease of the foxes population.
    """

    f1 = rabbits_ODE(nrabbits, nfoxes, epsilon_rabbits, gamma_rabbits)
    f2 = foxes_ODE(nrabbits, nfoxes, epsilon_foxes, gamma_foxes)
    return [f1, f2]
end


function euler_method(t0, tf, h, nrabbits0, nfoxes0)

    """
    This function solves the system of ODEs using the explicit Euler method.

    Parameters:
    @t0: initial time.
    @tf: final time.
    @h: step size.
    @nrabbits0: Initial number of rabbits
    @nfoxes0: Initial number of foxes

    Variables:
    @t: time
    @rabbits: number of rabbits
    @foxes: number of foxes
    @epsilon_rabbits: parameter that controls the growth of the rabbits population.
    @gamma_rabbits: parameter that controls the decrease of the rabbits population.
    @epsilon_foxes: parameter that controls the growth of the foxes population.
    @gamma_foxes: parameter that controls the decrease of the foxes population.
    @t_values: Array to store @t.
    @rabbits_arr: Array to store @rabbits.
    @foxes_arr: Array to store @foxes
    @dy_dp: Array to store the derivatives of rabbits and foxes in relation to parameters over time
    @dy_dy0: Array to store the derivatives of rabbits and foxes in relation to their respective initial values over time


    Observations:
    The derivatives are organized as the following matrix:
        
        |dy1/de1 dy1/dgamma1 dy1/de2 dy1/dgamma2|
        |dy2/de1 dy2/dgamma1 dy2/de2 dy2/dgamma2|
        2 x 4

        For the dy_dy0 array:
        |dy1/dy10 dy1/dy20|
        |dy2/dy10 dy2/dy20|

        Jacobian format is:
        |df1/dy1 df1/dy2 df1/de1 df1/dgamma1 df1/de2 df1/dgamma2|
        |df2/dy1 df2/dy2 df2/de1 df2/dgamma1 df2/de2 df2/dgamma2|
        2 x 6
    The evolution of the equations and parameters are given in the paper https://arxiv.org/pdf/1802.02247, equations (1), (2) and (3).

    """
    t::Float64 = t0
    rabbits::Float64 = nrabbits0      
    foxes::Float64 = nfoxes0     

    epsilon_rabbits::Float64 = 0.015
    gamma_rabbits::Float64 = 0.0001
    epsilon_foxes::Float64 = 0.03
    gamma_foxes::Float64 = 0.0001


    t_values = zeros(Float64, Int((tf - t0) / h) + 1)
    rabbits_arr = zeros(Float64, Int((tf - t0) / h) + 1)
    foxes_arr = zeros(Float64, Int((tf - t0) / h) + 1)  
    dy_dp = zeros(Float64, 2, 4, Int((tf - t0) / h) + 1)
    dy_dy0 = zeros(Float64, 2, 2, Int((tf - t0) / h) + 1)

    dy_dy0[1,1,1] = 1.0
    dy_dy0[2,2,1] = 1.0
    t_values[1] = t0
    rabbits_arr[1] = nrabbits0
    foxes_arr[1] = nfoxes0
    jacobian_matrix = zeros(Float64, 2, 6)
    step = 2

    while t < tf
        jacobian_matrix = ForwardDiff.jacobian(x -> system(x[1], x[2], x[3], x[4], x[5], x[6]), [rabbits, foxes, epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes])
        
        # rabbits and foxes derivative with respect to epsilon_rabbits
        dy_dp[1,1,step] = dy_dp[1,1,step - 1] +  h * (jacobian_matrix[1,1] *  dy_dp[1,1,step-1] + jacobian_matrix[1,2] * dy_dp[2,1,step-1] + jacobian_matrix[1,3])
        dy_dp[2,1,step] = dy_dp[2,1,step - 1] + h * (jacobian_matrix[2,2] *  dy_dp[2,1,step-1] + jacobian_matrix[2,1] *  dy_dp[1,1,step-1])
  

        # rabbits and foxes derivative with respect to gamma_rabbits
        dy_dp[1,2,step] = dy_dp[1,2,step - 1] + h * (jacobian_matrix[1,1] *  dy_dp[1,2,step-1] + dy_dp[2,2,step-1]  * jacobian_matrix[1,2] + jacobian_matrix[1,4])
        dy_dp[2,2,step] = dy_dp[2,2,step - 1] + h * (jacobian_matrix[2,2] *  dy_dp[2,2,step-1] + dy_dp[1,2,step-1] * jacobian_matrix[2,1])
  
        # rabbits and foxes derivative with respect to epsilon_foxes
        dy_dp[1,3,step] = dy_dp[1,3,step - 1] + h * (jacobian_matrix[1,1] *  dy_dp[1,3,step-1] + dy_dp[2,3,step-1]  * jacobian_matrix[1,2])
        dy_dp[2,3,step] = dy_dp[2,3,step - 1] + h * (jacobian_matrix[2,2] *  dy_dp[2,3,step-1] + dy_dp[1,3,step-1] * jacobian_matrix[2,1] + jacobian_matrix[2,5])

        # rabbits and foxes derivative with respect to gamma_foxes
        dy_dp[1,4,step] = dy_dp[1,4,step - 1] + h * (jacobian_matrix[1,1] *  dy_dp[1,4,step-1] + dy_dp[2,4,step-1]  * jacobian_matrix[1,2])
        dy_dp[2,4,step] = dy_dp[2,4,step - 1] + h * (jacobian_matrix[2,2] *  dy_dp[2,4,step-1] + dy_dp[1,4,step-1] * jacobian_matrix[2,1] + jacobian_matrix[2,6])

        # rabbits and foxes derivative with respect to nrabbits0
        dy_dy0[1,1,step] = dy_dy0[1,1,step - 1] + h * (jacobian_matrix[1,1] *  dy_dy0[1,1,step-1] + jacobian_matrix[1,2] * dy_dy0[2,1,step-1])
        dy_dy0[2,1,step] = dy_dy0[2,1,step - 1] + h * (jacobian_matrix[2,2] *  dy_dy0[2,1,step-1] + jacobian_matrix[2,1] * dy_dy0[1,1,step-1])       

        # rabbits and foxes derivative with respect to nfoxes0
        dy_dy0[1,2,step] = dy_dy0[1,2,step - 1] + h * (jacobian_matrix[1,1] *  dy_dy0[1,2,step-1] + jacobian_matrix[1,2] * dy_dy0[2,2,step-1])
        dy_dy0[2,2,step] = dy_dy0[2,2,step - 1] + h * (jacobian_matrix[2,2] *  dy_dy0[2,2,step-1] + jacobian_matrix[2,1] * dy_dy0[1,2,step-1])
        
        # rabbits and foxes over time
        rabbits += h * rabbits_ODE(rabbits, foxes, epsilon_rabbits, gamma_rabbits)
        foxes += h * foxes_ODE(rabbits, foxes, epsilon_foxes, gamma_foxes)

        t += h
        step += 1
        if(t < tf)
            t_values[step] = t
            rabbits_arr[step] = rabbits
            foxes_arr[step] = foxes
        end
    end

    return t_values, rabbits_arr, foxes_arr, dy_dp, dy_dy0 
end



function main()
    """
    Main function to wrap around what is happening in the code.

    Variables:
    @nrabbits0: Initial number of rabbits
    @nfoxes0: Initial number of foxes
    @t0: initial time.
    @tf: final time.
    @step: step size.
    @dy_dp: Array to store the derivatives of rabbits and foxes in relation to parameters over time
    @dy_dy0: Array to store the derivatives of rabbits and foxes in relation to their respective initial values over time
    @t_values: Array to store @t.
    @rabbits_arr: Array to store @rabbits.
    @foxes_arr: Array to store @foxes
    """


    nrabbits0::Float64 = 1000.0
    nfoxes0::Float64 = 20.0
    t0::Float64 = 0.0
    tf::Float64 = 1000.0 
    step::Float64 = 0.1   

    dy_dp = zeros(Float64, 2, 4, Int((tf - t0) / step) + 1)
    dy_dy0 = zeros(Float64, 2, 2, Int((tf - t0) / step) + 1)
    t_values = zeros(Float64, Int((tf - t0) / step) + 1)
    rabbits_arr = zeros(Float64, Int((tf - t0) / step) + 1)
    foxes_arr = zeros(Float64, Int((tf - t0) / step) + 1)

    t_values, rabbits_arr, foxes_arr, dy_dp, dy_dy0 = euler_method(t0, tf, step, nrabbits0, nfoxes0)
    
    # These plots should be wrapped around in a function, but right now I'm lazy
    p = plot(t_values, rabbits_arr, label="rabbits Population(t)", xlabel="Time (t)", ylabel="Population", title="Explicit Euler Method: Y vs t", color = "blue")
    plot!(p, t_values, foxes_arr, label="Foxes Population(t)", xlabel="Time (t)", ylabel="Population", color = "lime")
    savefig(p, "./imgs/population.png")
    p = plot(t_values, dy_dp[1,1,:], label="drabbits/de1 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim=(-1.5e6, 1.5e6), color = "red")
    plot!(p, t_values, dy_dp[2,1,:], label="dFoxes/de1 (t)", xlabel="Time (t)", ylabel="Derivative", color = "light blue")
    savefig(p, "./imgs/derivative_e1.png") 
    p = plot(t_values, dy_dp[1,2,:], label="drabbits/dgamma1 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim =(-4e7, 3e7), color = "pink")
    plot!(p, t_values, dy_dp[2,2,:], label="dFoxes/dgamma1 (t)", xlabel="Time (t)", ylabel="Derivative", color = "yellow")
    savefig(p, "./imgs/derivative_gamma1.png")

    p = plot(t_values, dy_dp[1,3,:], label="drabbits/de2 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-1.5e6, 1e6), color = "red")
    plot!(p, t_values, dy_dp[2,3,:], label="dFoxes/de2 (t)", xlabel="Time (t)", ylabel="Derivative", color = "light blue")
    savefig(p, "./imgs/derivative_e2.png") 

    p = plot(t_values, dy_dp[1,4,:], label="drabbits/dgamma2 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-2e8, 2e8), color = "pink")
    plot!(p, t_values, dy_dp[2,4,:], label="dFoxes/dgamma2 (t)", xlabel="Time (t)", ylabel="Derivative", color = "yellow")
    savefig(p, "./imgs/derivative_gamma2.png")

    p = plot(t_values, dy_dy0[1,1,:], label="drabbits/drabbits0 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-20, 20), color = "red")
    plot!(p, t_values, dy_dy0[2,1,:], label="dFoxes/drabbits0 (t)", xlabel="Time (t)", ylabel="Derivative", color = "light blue")
    savefig(p, "./imgs/derivative_rabbits0.png")

    p = plot(t_values, dy_dy0[1,2,:], label="drabbits/dfoxes0 (t)", xlabel="Time (t)", ylabel="Derivative", title="Explicit Euler Method: dY vs t", ylim = (-200, 200), color = "pink")
    plot!(p, t_values, dy_dy0[2,2,:], label="dFoxes/dfoxes0 (t)", xlabel="Time (t)", ylabel="Derivative", color = "yellow")
    savefig(p, "./imgs/derivative_foxes0.png") 
    println("Done!")

end

#Calling the main function
main()
