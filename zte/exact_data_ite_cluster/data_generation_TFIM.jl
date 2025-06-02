include("functions_zte.jl")

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
    println("generating data for nq = ", nq, " nl = ", nl, " tstep = ", tstep)
    # exact expectation values calculations
    M = jordan_wigner_tfim(nq, 3.0, 0.0, 1.0) # J=1.0, h=1.0
    zz = zeros(length(time))
    for i in 1:length(time)
        zz[i] = zz_correlation(M, time[i], obs_i, obs_j)
    end

    fn = format("data/exact_J=3.0_{}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}.dat", model, nq, nl, tstep, obs_i, obs_j)

    if isfile(fn)
        println("File already exists: ", fn)

    else
        mkpath(dirname(fn))
        CSV.write(fn, DataFrame(time=time, zz=zz))
        println("Data saved to ", fn)
    end
end

data_generation_TFIM()