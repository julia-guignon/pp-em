include("../src/cpa_v2.0.jl")

function IBM_utility_exp_4b_all()
    global_logger(UnbufferedLogger(stdout, SubInfo))
    
    nq = 127# 127
    nl = 20# 0
    dt = 0.05 
    T = nl*dt
    # h = 1.0
    J = π/(dt*4.0)
    topology = ibmeagletopology
    # topology = bricklayertopology(nq)

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
    
    noise_kind="both"; angle_definition=π/20
    min_abs_coeff, min_abs_coeff_noisy, max_weight = 1e-5,1e-5,8
    depol_strength = 0.02
    dephase_strength = 0.02
    noise_levels=[1.0,1.3,1.5,1.8,2.0,2.2,2.5]; lambda=0.0; use_target=false
    
    observable = PauliSum(nq); add!(observable,:Z,62)
    # observable = PauliSum(nq); add!(observable,:Z,1)
    collect_exact = Float64[]; collect_noisy = Float64[]
    collect_zne = Float64[]; collect_cdr = Float64[]; collect_vncd = Float64[]
    collect_vncd_lin = Float64[]; collect_zne_lin = Float64[]

    layer = kickedisingcircuit(nq, 1; topology=topology)
    for (i,h) in enumerate(h_values)
        training_set = training_circuit_generation_loose_perturbation(layer, J, h, dt, angle_definition; sample_function="small", num_samples=10)

        exact, noisy,
        zne_corr,zne_corr_lin, cdr_corr, vn_corr, vn_corr_lin,
        _, _, _,_,_,
        _, _, _,_,_ = full_run_all_methods(
            nq, nl, topology, layer, J, h, dt, angle_definition, noise_kind;
            min_abs_coeff=min_abs_coeff, max_weight=max_weight,
            min_abs_coeff_noisy=min_abs_coeff_noisy,
            training_set=training_set, observable=observable,
            num_samples=10,
            depol_strength=depol_strength,
            dephase_strength=dephase_strength,
            noise_levels=noise_levels, lambda=lambda,
            use_target=use_target,
            real_qc_noisy_data=IBM_unmitigated_vals[i], record_fit_data = true, fit_type="exponential", fit_intercept = false
        )
        
        push!(collect_exact, exact)
        push!(collect_noisy, noisy)
        push!(collect_zne, zne_corr)
        push!(collect_zne_lin, zne_corr_lin)
        push!(collect_cdr, cdr_corr)
        push!(collect_vncd, vn_corr)
        push!(collect_vncd_lin, vn_corr_lin)
    end
    
     # log file for this utility run, stamped with current datetime
     run_ts = Dates.format(Dates.now(), "YYYYmmdd_HHMMSS")
     logfname = "tfim_utility_nq=$(nq)_angle_def=$(round(angle_definition;digits = 3))_$(run_ts)_cpa_v2.0.log"
     
    # write summary table to log
    open(logfname, "a") do io
        # header
        println(io, "idx,h_value,Exact_targets,Noisy_targets,ZNE_outputs,ZNE_outputs_lin,CDR_outputs,vnCDR_outputs, vnCDR_outputs_lin")
        # rows
        for i in 1:length(collect_exact)
            println(io, join((
                i,
                h_values[i],
                collect_exact[i],
                collect_noisy[i],
                collect_zne[i],
                collect_zne_lin[i],
                collect_cdr[i],
                collect_vncd[i],
                collect_vncd_lin[i]
            ), ","))
        end
    end

    end
    
    IBM_utility_exp_4b_all()
    println("done")