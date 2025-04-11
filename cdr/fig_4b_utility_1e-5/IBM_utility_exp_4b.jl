include("cpa.jl")

function IBM_utility_exp_4b()

    global_logger(UnbufferedLogger(stdout,MainInfo))
    IBM_angles = [0.3, 1.0, 0.7, 0.0, 0.2, 0.8, 0.5, 0.1, 0.4, 1.5707, 0.6]
    IBM_unmitigated_vals =  [ 0.4188991191900761,
    0.004107759335343423,
    0.11944580478416555,
    0.49038646460776864,
    0.4552471452020139,
    0.055064655494323766,
    0.3061535376123831,
    0.4889782663914668,
    0.3622122171682965,
   -0.001980699802309258,
    0.20175539633925924]

    nq = 127 # number of qubits
    nl = 20 #20 # number of layers/steps
    T = nl/20  # time resolution
    topology = ibmeagletopology

    observable = PauliSum(nq)
    add!(observable, :Z, 62) # different to Manuel

    h_values = IBM_angles*nl/(2*T)
    #h_values = h_values[1:2]
    println("h_values = ", h_values)
    noise_kind = "gate_kickedising"

    min_abs_coeff = 1e-5; # training set (exact and noisy)
    min_abs_coeff_noisy = min_abs_coeff;
    sigma_star = pi/20; # for our CPA methods

    collect_corr_energy = []
    for (i, h) in enumerate(h_values)
        trotter = trotter_kickedising_setup(nq, nl, T, h;topology = topology)
        global_logger(UnbufferedLogger(stdout, MainInfo))
        training_set = training_set_generation_loose_perturbation(trotter; sample_function = "CPA", num_samples = 10)

        exact_expval_target, noisy_expval_target, corr_energy, rel_error_before, rel_error_after = full_run(
            trotter, sigma_star, noise_kind;
            min_abs_coeff = min_abs_coeff,
            min_abs_coeff_noisy = min_abs_coeff_noisy,
            training_set = training_set,
            observable = observable,
            record = true,
            cdr_method = "end",
            use_target = false,
            real_qc_noisy_data = IBM_unmitigated_vals[i]
        )
        append!(collect_corr_energy, corr_energy)
    end
end

IBM_utility_exp_4b()
