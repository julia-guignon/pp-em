include("../src/cpa.jl")

function IBM_utility_exp_4b_all()
    global_logger(UnbufferedLogger(stdout, SubInfo))
    
    nq = 127
    nl = 20 
    T = nl/20
    topology = ibmeagletopology
    IBM_angles = [0.3,1.0,0.7,0.0,0.2,0.8,0.5,0.1,0.4,1.5707,0.6]
    h_values = IBM_angles .* nl/(2*T)
    
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
    
    noise_kind="gate_kickedising"; angle_definition=π/8
    min_abs_coeff, min_abs_coeff_noisy, max_weight = 1e-5,1e-5,20
    depol_strength,dephase_strength = 0.01,0.01
    depol_strength_double,dephase_strength_double = 0.0033,0.0033
    noise_levels=[1.0,1.5,2.0,2.5]; lambda=0.0; use_target=false; cdr_method="end"
    
    observable = PauliSum(nq); add!(observable,:Z,62)
    collect_exact = Float64[]; collect_noisy = Float64[]
    collect_zne = Float64[]; collect_cdr = Float64[]; collect_vncd = Float64[]
    
    for (i,h) in enumerate(h_values)
    trotter = trotter_kickedising_setup(nq, nl, T, h; topology=topology)
    training_set = training_circuit_generation_loose_perturbation(trotter; sample_function="small", num_samples=10)
    
    exact, noisy,
    zne_corr, cdr_corr, vn_corr,
    _, _, _,
    _, _, _ = full_run_all_methods(
    trotter, angle_definition, noise_kind;
    min_abs_coeff=min_abs_coeff, max_weight=max_weight,
    min_abs_coeff_noisy=min_abs_coeff_noisy,
    training_set=training_set, observable=observable,
    num_samples=10,
    depol_strength=depol_strength, dephase_strength=dephase_strength,
    depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double,
    noise_levels=noise_levels, lambda=lambda,
    use_target=use_target,
    real_qc_noisy_data=IBM_unmitigated_vals[i]
    )
    
    push!(collect_exact, exact)
    push!(collect_noisy, noisy)
    push!(collect_zne, zne_corr)
    push!(collect_cdr, cdr_corr)
    push!(collect_vncd, vn_corr)
    end
    
     # log file for this utility run, stamped with current datetime
     run_ts = Dates.format(Dates.now(), "YYYYmmdd_HHMMSS")
     logfname = "tfim_utility_nq=$(nq)_angle_def=$(round(theta_star;digits = 3))_$(run_ts).log"
     
    # write summary table to log
    open(logfname, "a") do io
        # header
        println(io, "idx,Exact_targets,Noisy_targets,ZNE_outputs,CDR_outputs,vnCDR_outputs")
        # rows
        for i in 1:length(collect_exact)
            println(io, join((
                i,
                collect_exact[i],
                collect_noisy[i],
                collect_zne[i],
                collect_cdr[i],
                collect_vncd[i]
            ), ","))
        end
    end

    end
    
    IBM_utility_exp_4b_all()