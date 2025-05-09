using PauliPropagation
using LinearAlgebra
using Format
using CSV
using DataFrames	

function interpretPauli(p::String)
    I = [1 0; 0 1]
    X = [0 1; 1 0]
    Y = [0 -1im; 1im 0]
    Z = [1 0; 0 -1]

    if p[1] == 'I'
        mat = I
    elseif p[1] == 'X'
        mat = X
    elseif p[1] == 'Y'
        mat = Y
    elseif p[1] == 'Z'
        mat = Z
    end

    for i in 2:length(p)
        if p[i] == 'I'
            mat = kron(mat, I)
        elseif p[i] == 'X'
            mat = kron(mat, X)
        elseif p[i] == 'Y'
            mat = kron(mat, Y)
        elseif p[i] == 'Z'
            mat = kron(mat, Z)
        end
    end
    return mat
end

function imaginaryTimeEvolution(h::Matrix, beta::Float64)
    operator = exp(-beta*h)
    return operator/tr(exp(-beta*h))
end

function expectationValue(rho::Matrix, pauli::String)
    mat = interpretPauli(pauli)
    return tr(rho*mat)
end

function XMatrix(n::Int, i::Int)
    Xmat = [0 1; 1 0]
    Imat = [1 0; 0 1]

    operator = (i == 1) ? Xmat : Imat

    for j in 2:n
        if j == i
            operator = kron(operator, Xmat)
        else
            operator = kron(operator, Imat)
        end
    end
    
    return operator
end

function YMatrix(n::Int, i::Int)
    Ymat = [0 -im; im 0]
    Imat = [1 0; 0 1]

    operator = (i == 1) ? Ymat : Imat

    for j in 2:n
        if j == i
            operator = kron(operator, Ymat)
        else
            operator = kron(operator, Imat)
        end
    end
    
    return operator
end

function ZMatrix(n::Int, i::Int)
    Zmat = [1 0; 0 -1]
    Imat = [1 0; 0 1]

    operator = (i == 1) ? Zmat : Imat

    for j in 2:n
        if j == i
            operator = kron(operator, Zmat)
        else
            operator = kron(operator, Imat)
        end
    end
    
    return operator
end


function data_generation_heisenberg()
    
    model = "Heisenberg"

    # system setup
    nq = 127 #127
    nl = 10 
    tstep = 0.01
    time = range(tstep, tstep*nl, nl)
    # obs = PauliString(nq, [:Z,:Z], [21,20]) as a string
    obs = "IIIIIIIIII 
           ZZIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIIIIII
           IIIIIII"
    
    println("obs: ", obs)

    # system parameters
    H = zeros(2^nq, 2^nq)
    for i in 1:nq-1
        H -= XMatrix(nq, i)*XMatrix(nq, i+1)
        H -= YMatrix(nq, i)*YMatrix(nq, i+1)
        H -= ZMatrix(nq, i)*ZMatrix(nq, i+1)
    end

    exactExpectations = []
    for t in time
        exactState = imaginaryTimeEvolution(H, t)

        exactExpectation = expectationValue(exactState, obs)
        push!(exactExpectations, exactExpectation)
    end
    println("exactExpectations: ", exactExpectations)
    fn = format("data/{}_nq={:d}_nl={:d}_theta={:.3e}.dat", model, nq, nl, tstep)

    if isfile(fn)
        println("File already exists: ", fn)

    else
        mkpath(dirname(fn))
        CSV.write(fn, DataFrame(time=time, exact=exactExpectations))
        println("Data saved to ", fn)
    end

end

data_generation_heisenberg()