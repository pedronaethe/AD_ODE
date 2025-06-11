using Printf
export print_vector, print_matrix, check_parameters



function print_vector(name::String, vec::MVec4)
    """
    Returns a string representation of a vector with the given name.

    Parameters:
    @name: The name of the vector to be printed.
    @vec: The vector to be printed.
    """
    println("Vector: $name")
    for i in eachindex(vec)
        print("$(vec[i]) ")
    end
    println()
end
function print_matrix(name::String, mat::MMat4)
    """
    Returns a string representation of a matrix with the given name.

    Parameters:
    @name: The name of the matrix to be printed.
    @mat: The matrix to be printed.
    """
    println("Matrix: $name")
    for i in axes(mat, 1)
        for j in axes(mat, 2)
            @printf("%.15e ", mat[i, j])
        end
        println()
    end
end

function check_parameters()
    """
    Checks the consistency of the parameters used in the simulation.
    """

    if(MODEL == "thin_disk")
        error("The thin disk model is not implemented yet!")
    end

    if(nx != ny && Krang == true)
        error("Currently Nx must be equal to Ny when using Krang!")
    end
end

