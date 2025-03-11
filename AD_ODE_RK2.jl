
#Based off: https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#:~:text=Native%20Julia%20solvers%20compatibility%20with,analysis%20page%20for%20more%20details.
using DifferentialEquations
using ForwardDiff
using Plots

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


function func(du, u, p, t)~

    """
    This function declares the system of ODEs that are going to be solved by solve.
    @du is the derivative of the state variables.
    @u is the state variables.
    @p is the parameters.
    @t is the time.

    Indexes:
    @du and @u = [
        1  - drabbits/dt,       2  - dfoxes/dt,
        3  - drabbits/de1,      4  - dfoxes/de1,
        5  - drabbits/dg1,      6  - dfoxes/dg1,
        7  - drabbits/de2,      8  - dfoxes/de2,
        9  - drabbits/dg2,      10 - dfoxes/dg2,
        11 - drabbits/dnrabbits0, 12 - dfoxes/dnrabbits0,
        13 - drabbits/dnfoxes0, 14 - dfoxes/dnfoxes0
    ]
    @p = [epsilon_rabbits, gamma_rabbits, epsilon_foxes, gamma_foxes, nrabbits0, nfoxes0]
    """
    #drabbits/erab and dfoxes/erab
    jacobian_matrix = ForwardDiff.jacobian(x -> system(x[1], x[2], x[3], x[4], x[5], x[6]), [u[1], u[2], p[1], p[2], p[3], p[4]])
    
    # rabbits and foxes derivative with respect to epsilon_rabbits
    du[3] = jacobian_matrix[1,1] *  u[3] + jacobian_matrix[1,2] * u[4] + jacobian_matrix[1,3]
    du[4] = (jacobian_matrix[2,2] *  u[4] + jacobian_matrix[2,1] *  u[3])

    # rabbits and foxes derivative with respect to gamma_rabbits
    du[5] = jacobian_matrix[1,1] *  u[5] + jacobian_matrix[1,2] * u[6] + jacobian_matrix[1,4]
    du[6] = (jacobian_matrix[2,2] *  u[6] + jacobian_matrix[2,1] *  u[5])

    # rabbits and foxes derivative with respect to epsilon_foxes
    du[7] = jacobian_matrix[1,1] *  u[7] + jacobian_matrix[1,2] * u[8]
    du[8] = (jacobian_matrix[2,2] *  u[8] + jacobian_matrix[2,1] *  u[7] + jacobian_matrix[2,5])

    # rabbits and foxes derivative with respect to gamma_foxes
    du[9] = jacobian_matrix[1,1] *  u[9] + jacobian_matrix[1,2] * u[10]
    du[10] = (jacobian_matrix[2,2] *  u[10] + jacobian_matrix[2,1] *  u[9] + jacobian_matrix[2,6])

    # rabbits and foxes derivative with respect to nrabbits0      
    du[11] = jacobian_matrix[1,1] *  u[11] + jacobian_matrix[1,2] * u[12]
    du[12] = (jacobian_matrix[2,2] *  u[12] + jacobian_matrix[2,1] *  u[11])

    # rabbits and foxes derivative with respect to nfoxes0
    du[13] = jacobian_matrix[1,1] *  u[13] + jacobian_matrix[1,2] * u[14]
    du[14] = (jacobian_matrix[2,2] *  u[14] + jacobian_matrix[2,1] *  u[13])
    
    #rabbits
    du[1] = (p[1] - p[2] *u[2]) * u[1]
    #foxes
    du[2] = -(p[3] - p[4] * u[1]) * u[2]
end

function solve_system(p)
    """
    This function solves the system of ODEs.
    @u0 is the initial conditions.
    @tspan is the time span.
    @p is the parameters.
    
    Indexes:
    @u0 = [
        1  - rabbits,             2  - foxes,
        3  - drabbits/de1,        4  - dfoxes/de1,
        5  - drabbits/dg1,        6  - dfoxes/dg1,
        7  - drabbits/de2,        8  - dfoxes/de2,
        9  - drabbits/dg2,        10 - dfoxes/dg2,
        11 - drabbits/dnrabbits0, 12 - dfoxes/dnrabbits0,
        13 - drabbits/dnfoxes0,   14 - dfoxes/dnfoxes0
    ]
    """
    u0 = [1000.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0] 
    tspan = (0.0, 1000.0)
    prob = ODEProblem(func, u0, tspan, p)
    sol = solve(prob, Midpoint(), saveat=0.1)
    return sol
end

function plot_and_save(sol)
    """
    This function plots the results of the system of ODEs.
    @sol is the solution of the system of ODEs.

    Indexes:
    @sol = [
        1  - time,                2  - rabbits,
        3  - foxes,               4  - drabbits/de1,
        5  - dfoxes/de1,          6  - drabbits/dg1,
        7  - dfoxes/dg1,          8  - drabbits/de2,
        9  - dfoxes/de2,          10 - drabbits/dg2,
        11 - dfoxes/dg2,          12 - drabbits/dnrabbits0,
        13 - dfoxes/dnrabbits0,   14 - drabbits/dnfoxes0,
        15 - dfoxes/dnfoxes0
    ]
    """
    t = sol.t
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
    drabbits_dnrabbits0 = sol[11, :]
    dfoxes_dnrabbits0 = sol[12, :]
    drabbits_dnfoxes0 = sol[13, :]
    dfoxes_dnfoxes0 = sol[14, :]
    
    plot(t, rabbits, xlabel="Time", ylabel="Rabbits", title="Population over time", lw=2, label = "Rabbits")    
    plot!(t, foxes, label ="Foxes", lw=2)
    savefig("./imgs/population.png")
    
    plot(t, drabbits_de1, xlabel="Time", ylabel="dY", title="dY/de1", lw=2, label = "dRabbits/de1")    
    plot!(t, dfoxes_de1, label = "dFoxes/de1", lw=2)
    savefig("./imgs/derivative_e1.png")

    plot(t, drabbits_dg1, xlabel="Time", ylabel="dY", title="dY/dg1", lw=2, label = "dRabbits/dg1")
    plot!(t, dfoxes_dg1, label = "dFoxes/dg1", lw=2)
    savefig("./imgs/derivative_g1.png")

    plot(t, drabbits_de2, xlabel="Time", ylabel="dY", title="dY/de2", lw=2, label = "dRabbits/de2")
    plot!(t, dfoxes_de2, label = "dFoxes/de2", lw=2)
    savefig("./imgs/derivative_e2.png")

    plot(t, drabbits_dg2, xlabel="Time", ylabel="dY", title="dY/dg2", lw=2, label = "dRabbits/dg2")
    plot!(t, dfoxes_dg2, label = "dFoxes/dg2", lw=2)
    savefig("./imgs/derivative_g2.png")

    plot(t, drabbits_dnrabbits0, xlabel="Time", ylabel="dY", title="dY/dnrabbits0", lw=2, label = "dRabbits/dnrabbits0")
    plot!(t, dfoxes_dnrabbits0, label = "dFoxes/dnrabbits0", lw=2)
    savefig("./imgs/derivative_rab0.png")

    plot(t, drabbits_dnfoxes0, xlabel="Time", ylabel="dY", title="dY/dnfoxes0", lw=2, label = "dRabbits/dnfoxes0")
    plot!(t, dfoxes_dnfoxes0, label = "dFoxes/dnfoxes0", lw=2)
    savefig("./imgs/derivative_fox0.png")

    return
end

p = [0.015, 0.0001, 0.03, 0.0001]
sol = solve_system(p)
plot_and_save(sol)



