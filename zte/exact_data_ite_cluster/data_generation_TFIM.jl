using PauliPropagation
using Plots
using Logging
using Statistics
using LinearAlgebra
using GLM
using StatsModels
using LsqFit
using Distributions
using Format
using DataFrames
using CSV
using LaTeXStrings

function jordan_wigner_tfim(n_sites::Int, Jx::Float64=1.0, Jy::Float64=1.0, h::Float64=1.0)
    """
    Construct the Bogoliubov-de Gennes matrix for the Ising model:

    Args:
        n_sites (Int): Number of sites
        J (Float64): Coupling strength for σ^x σ^x interaction
        h (Float64): Transverse field strength

    Returns:
        Matrix{Float64}: 2n_sites × 2n_sites BdG matrix
    """
    J = Jx+Jy
    k = (Jx-Jy)/J
    A = zeros(Float64, n_sites, n_sites)
    B = zeros(Float64, n_sites, n_sites)

    for i in 1:(n_sites-1)
        A[i, i] = h
        A[i+1, i+1] = h
    end

    for i in 1:(n_sites-1)
        A[i, i+1] = -J/2
        A[i+1, i] = -J/2
    end

    for i in 1:(n_sites-1)
        B[i, i+1] = -J*k/2
        B[i+1, i] = J*k/2
    end

    M = [A  B;
        -B -A]

    return M
end

function zz_correlation(M::Matrix{Float64}, beta::Float64, i::Int, j::Int)
    N = size(M, 1) ÷ 2

    eigvals, eigvecs = eigen(Hermitian(M))

    # Extract u and v components
    Uk = eigvecs[1:N, :]
    Vk = eigvecs[N+1:end, :]

    # Thermal occupations
    f = 1.0 ./ (exp.(2 * beta .* eigvals) .+ 1.0)
    f_diag = Diagonal(f)

    G = Uk * f_diag * Uk'
    F = Uk * f_diag * Vk'

    zz = 4 * (G[i,i] - 0.5) * (G[j,j] - 0.5) - 4 * G[i,j] * G[j,i] + 4 * abs2(F[i,j])

    return zz
end

function data_generation_TFIM()
    model = "TFIM"

    #system setup
    #theta = 0.01im # same as tstep
    #θ = theta
    nq = 30

    # system parameters
    obs_i = 21
    obs_j = 20
    #observable = PauliString(nq, [:Z,:Z], [obs_i,obs_j])
    nl =  30

    tstep = 0.01
    time = range(tstep, tstep*nl, nl)

    # exact expectation values calculations
    M = jordan_wigner_tfim(nq, 1.0, 0.0, 1.0) # J=1.0, h=1.0
    zz = zeros(length(time))
    for i in 1:length(time)
        zz[i] = zz_correlation(M, time[i], obs_i, obs_j)
    end

    fn = format("data/{}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}.dat", model, nq, nl, tstep, obs_i, obs_j)

    if isfile(fn)
        println("File already exists: ", fn)

    else
        mkpath(dirname(fn))
        CSV.write(fn, DataFrame(time=time, zz=zz))
        println("Data saved to ", fn)
    end
end

data_generation_TFIM()