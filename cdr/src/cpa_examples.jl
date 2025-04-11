include("cpa.jl")

function full_run_test() ###
    global_logger(UnbufferedLogger(stdout, SubInfo))

    nq = 20 #20
    steps = 8 #8
    sigma_star = pi/20
    T = 0.8 #factor 10 with steps
    J = 2.0 #J > 0 in ferromagnetic phase, J < 0 in antiferromagnetic phase
    h = 1.0 #abs(h) < abs(J) in ordered phase

    trotter = trotter_setup(nq, steps, T, J, h);
    noise_kind = "gate" #takes the default realistic values
    min_abs_coeff = 1e-8;
    min_abs_coeff_target = 1e-12;
    min_abs_coeff_noisy = min_abs_coeff;
    training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "small", num_samples=10);
    full_run(trotter, sigma_star, noise_kind; min_abs_coeff = min_abs_coeff, min_abs_coeff_noisy = min_abs_coeff_noisy, observable = obs_magnetization(trotter))
    return 0
end

full_run_test()

### Figure 4 ###
function fig_4ab()
    depol_strength_double = 0.0033
    dephase_strength_double = 0.0033
    depol_strength = 0.015 #0.00035
    dephase_strength = 0.015 #0.00035
    #noise_kind = "realistic"
    noise_kind = "gate"


    global_logger(UnbufferedLogger(stdout,MainInfo))
    nq = 9
    steps = 5
    sigma_star = pi/20
    T = 0.5 
    sigma_h = 0.01:0.135:1.51 #0.135
    sigma_J = 0.01:0.135:1.51 

    min_abs_coeff = 1e-10; 
    min_abs_coeff_target =0.0; 
    min_abs_coeff_noisy = min_abs_coeff; #also taken as truncation for noisy target circuit
    num_samples = 10 


    ### Fig 4a ###

    MSE_loose_CPA_exp_list = []
    MSE_loose_CPA_ind_list = []
    MSE_loose_small_exp_list = []
    MSE_loose_small_ind_list = []
    MSE_noise_exp_list = []
    MSE_noise_ind_list = []
    MSE_strict_CPA_exp_list = []
    MSE_strict_CPA_ind_list = []
    MSE_strict_small_exp_list = []
    MSE_strict_small_ind_list = []
    MSE_brut_exp_list = []
    MSE_brut_ind_list = []

    for sig_h in sigma_h 
        h = sig_h*steps/(2*T)
        for sig_J in sigma_J
            J = -sig_J*steps/(2*T)
            trotter = trotter_setup(nq, steps, T, J, h);
            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            exact_expval_target, noisy_expval_target, corr_energy, rel_error_before, rel_error_after = full_run(trotter, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, observable = obs_magnetization(trotter), training_set = training_set, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double,min_abs_coeff_target=min_abs_coeff_target);
            
            MSE_ind = (exact_expval_target - corr_energy)^2
            push!(MSE_loose_CPA_ind_list, MSE_ind)

            MSE_noise = (exact_expval_target - noisy_expval_target)^2
            push!(MSE_noise_ind_list, MSE_noise)

            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_loose_small_ind_list, MSE_ind)

            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_strict_CPA_ind_list, MSE_ind)
            
            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)         
            push!(MSE_strict_small_ind_list, MSE_ind)

            training_set = training_set_generation_brut(trotter, sigma_star; num_samples = num_samples, non_replaced_gates=30);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_brut_ind_list, MSE_ind)

        end

        push!(MSE_loose_CPA_exp_list, MSE_loose_CPA_ind_list)
        push!(MSE_loose_small_exp_list, MSE_loose_small_ind_list)
        push!(MSE_strict_CPA_exp_list, MSE_strict_CPA_ind_list)
        push!(MSE_strict_small_exp_list, MSE_strict_small_ind_list)
        push!(MSE_brut_exp_list, MSE_brut_ind_list)
        push!(MSE_noise_exp_list, MSE_noise_ind_list)

        MSE_loose_CPA_ind_list = []
        MSE_loose_small_ind_list = []
        MSE_strict_CPA_ind_list = []
        MSE_strict_small_ind_list = []
        MSE_brut_ind_list = []
        MSE_noise_ind_list = []
    end
    MSE_loose_CPA_fig_c = []
    MSE_loose_small_fig_c = []
    MSE_strict_CPA_fig_c = []
    MSE_strict_small_fig_c = []
    MSE_brut_fig_c = []
    MSE_noise_fig_c = []

    for i in MSE_loose_CPA_exp_list
        MSE = mean(i)
        push!(MSE_loose_CPA_fig_c, MSE)
    end
    for i in MSE_loose_small_exp_list
        MSE = mean(i)
        push!(MSE_loose_small_fig_c, MSE)
    end
    for i in MSE_strict_CPA_exp_list
        MSE = mean(i)
        push!(MSE_strict_CPA_fig_c, MSE)
    end
    for i in MSE_strict_small_exp_list
        MSE = mean(i)
        push!(MSE_strict_small_fig_c, MSE)
    end
    for i in MSE_noise_exp_list
        MSE = mean(i)
        push!(MSE_noise_fig_c, MSE)
    end
    for i in MSE_brut_exp_list
        MSE = mean(i)
        push!(MSE_brut_fig_c, MSE)
    end

    ### save data ###
    df = DataFrame( sigma_h = sigma_h,MSE_loose_small = MSE_loose_small_fig_c, MSE_loose_CPA = MSE_loose_CPA_fig_c,  MSE_strict_small = MSE_strict_small_fig_c, MSE_strict_CPA = MSE_strict_CPA_fig_c, MSE_brut = MSE_brut_fig_c, MSE_noise = MSE_noise_fig_c)
    fn = format("pp-em/cdr/data/Fig_4a_noise_type={:s}_T={:.2f}_angdef={:.2f}_steps={:n}_nqubits={:n}_nsamples={:n}_minabs={:.2e}_minabsnoisy={:.2e}_depol={:.2e}_dephase={:.2e}.csv",  
    noise_kind,T, sigma_star, steps, nq, num_samples, min_abs_coeff, min_abs_coeff_noisy,depol_strength, dephase_strength)
    CSV.write(fn, df)
    plot_MSE_csv_data(fn,"sigma_h")

    ### Fig 4b ###

    MSE_loose_CPA_exp_list = []
    MSE_loose_CPA_ind_list = []
    MSE_loose_small_exp_list = []
    MSE_loose_small_ind_list = []
    MSE_noise_exp_list = []
    MSE_noise_ind_list = []
    MSE_strict_CPA_exp_list = []
    MSE_strict_CPA_ind_list = []
    MSE_strict_small_exp_list = []
    MSE_strict_small_ind_list = []
    MSE_brut_exp_list = []
    MSE_brut_ind_list = []

    for sig_J in sigma_J 
        J = -sig_J*steps/(2*T)
        for sig_h in sigma_h
            h = sig_h*steps/(2*T)
            
            trotter = trotter_setup(nq, steps, T, J, h);
            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            exact_expval_target, noisy_expval_target, corr_energy, rel_error_before, rel_error_after = full_run(trotter, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, min_abs_coeff_target=min_abs_coeff_target, observable = obs_magnetization(trotter), training_set = training_set, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double);
            MSE_ind = (exact_expval_target - corr_energy)^2
            push!(MSE_loose_CPA_ind_list, MSE_ind)

            MSE_noise = (exact_expval_target - noisy_expval_target)^2
            push!(MSE_noise_ind_list, MSE_noise)

            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_loose_small_ind_list, MSE_ind)

            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_strict_CPA_ind_list, MSE_ind)
            
            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)         
            push!(MSE_strict_small_ind_list, MSE_ind)

            training_set = training_set_generation_brut(trotter, sigma_star; num_samples = num_samples, non_replaced_gates=30);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_brut_ind_list, MSE_ind)
        end

        push!(MSE_loose_CPA_exp_list, MSE_loose_CPA_ind_list)
        push!(MSE_loose_small_exp_list, MSE_loose_small_ind_list)
        push!(MSE_strict_CPA_exp_list, MSE_strict_CPA_ind_list)
        push!(MSE_strict_small_exp_list, MSE_strict_small_ind_list)
        push!(MSE_brut_exp_list, MSE_brut_ind_list)
        push!(MSE_noise_exp_list, MSE_noise_ind_list)

        MSE_loose_CPA_ind_list = []
        MSE_loose_small_ind_list = []
        MSE_strict_CPA_ind_list = []
        MSE_strict_small_ind_list = []
        MSE_brut_ind_list = []
        MSE_noise_ind_list = []
    end
    MSE_loose_CPA_fig_d = []
    MSE_loose_small_fig_d = []
    MSE_strict_CPA_fig_d = []
    MSE_strict_small_fig_d = []
    MSE_brut_fig_d = []
    MSE_noise_fig_d = []

    for i in MSE_loose_CPA_exp_list
        MSE = mean(i)
        push!(MSE_loose_CPA_fig_d, MSE)
    end
    for i in MSE_loose_small_exp_list
        MSE = mean(i)
        push!(MSE_loose_small_fig_d, MSE)
    end
    for i in MSE_strict_CPA_exp_list
        MSE = mean(i)
        push!(MSE_strict_CPA_fig_d, MSE)
    end
    for i in MSE_strict_small_exp_list
        MSE = mean(i)
        push!(MSE_strict_small_fig_d, MSE)
    end
    for i in MSE_noise_exp_list
        MSE = mean(i)
        push!(MSE_noise_fig_d, MSE)
    end
    for i in MSE_brut_exp_list
        MSE = mean(i)
        push!(MSE_brut_fig_d, MSE)
    end

    df = DataFrame(sigma_J = sigma_J,MSE_loose_small = MSE_loose_small_fig_d, MSE_loose_CPA = MSE_loose_CPA_fig_d,  MSE_strict_small = MSE_strict_small_fig_d, MSE_strict_CPA = MSE_strict_CPA_fig_d, MSE_brut = MSE_brut_fig_d, MSE_noise = MSE_noise_fig_d)
    fn = format("pp-em/cdr/data/Fig_4b_noise_type={:s}_T={:.2f}_angdef={:.2f}_steps={:n}_nqubits={:n}_nsamples={:n}_minabs={:.2e}_minabsnoisy={:.2e}_depol={:.2e}_dephase={:.2e}.csv",  
    noise_kind, T, sigma_star, steps, nq, num_samples, min_abs_coeff, min_abs_coeff_noisy,depol_strength, dephase_strength)
    CSV.write(fn, df)
    plot_MSE_csv_data(fn,"sigma_J")
    return 0
end

#fig_4ab()


function fig_4cd()
    # depol_strength_double = 0.0033
    # dephase_strength_double = 0.0033
    # depol_strength = 0.00035
    # dephase_strength = 0.00035
    # noise_kind = "realistic"
    noise_kind = "gate"
    depol_strength_double = 0.0033
    dephase_strength_double = 0.0033
    depol_strength = 0.01 #0.00035
    dephase_strength = 0.01 #0.00035

    global_logger(UnbufferedLogger(stdout,MainInfo))
    nq = 20
    steps = 8
    sigma_star = pi/20
    T = 0.8 #0.8
    sigma_h = 0.01:0.135:1.51 #0.135
    sigma_J = 0.01:0.135:1.51 

    min_abs_coeff = 1e-8; #1e-8
    min_abs_coeff_target =1e-12; #1e-12
    min_abs_coeff_noisy = min_abs_coeff; #also taken as truncation for noisy target circuit
    num_samples = 10


    ### Fig 4c ###

    MSE_loose_CPA_exp_list = []
    MSE_loose_CPA_ind_list = []
    MSE_loose_small_exp_list = []
    MSE_loose_small_ind_list = []
    MSE_noise_exp_list = []
    MSE_noise_ind_list = []
    MSE_strict_CPA_exp_list = []
    MSE_strict_CPA_ind_list = []
    MSE_strict_small_exp_list = []
    MSE_strict_small_ind_list = []
    MSE_brut_exp_list = []
    MSE_brut_ind_list = []

    for sig_h in sigma_h 
        h = sig_h*steps/(2*T)
        for sig_J in sigma_J
            J = -sig_J*steps/(2*T)
            trotter = trotter_setup(nq, steps, T, J, h);
            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            exact_expval_target, noisy_expval_target, corr_energy, rel_error_before, rel_error_after = full_run(trotter, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, observable = obs_magnetization(trotter), training_set = training_set, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double,min_abs_coeff_target=min_abs_coeff_target);
            
            MSE_ind = (exact_expval_target - corr_energy)^2
            push!(MSE_loose_CPA_ind_list, MSE_ind)

            MSE_noise = (exact_expval_target - noisy_expval_target)^2
            push!(MSE_noise_ind_list, MSE_noise)

            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_loose_small_ind_list, MSE_ind)

            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_strict_CPA_ind_list, MSE_ind)
            
            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)         
            push!(MSE_strict_small_ind_list, MSE_ind)

            training_set = training_set_generation_brut(trotter, sigma_star; num_samples = num_samples, non_replaced_gates=30);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_brut_ind_list, MSE_ind)

        end

        push!(MSE_loose_CPA_exp_list, MSE_loose_CPA_ind_list)
        push!(MSE_loose_small_exp_list, MSE_loose_small_ind_list)
        push!(MSE_strict_CPA_exp_list, MSE_strict_CPA_ind_list)
        push!(MSE_strict_small_exp_list, MSE_strict_small_ind_list)
        push!(MSE_brut_exp_list, MSE_brut_ind_list)
        push!(MSE_noise_exp_list, MSE_noise_ind_list)

        MSE_loose_CPA_ind_list = []
        MSE_loose_small_ind_list = []
        MSE_strict_CPA_ind_list = []
        MSE_strict_small_ind_list = []
        MSE_brut_ind_list = []
        MSE_noise_ind_list = []
    end
    MSE_loose_CPA_fig_c = []
    MSE_loose_small_fig_c = []
    MSE_strict_CPA_fig_c = []
    MSE_strict_small_fig_c = []
    MSE_brut_fig_c = []
    MSE_noise_fig_c = []

    for i in MSE_loose_CPA_exp_list
        MSE = mean(i)
        push!(MSE_loose_CPA_fig_c, MSE)
    end
    for i in MSE_loose_small_exp_list
        MSE = mean(i)
        push!(MSE_loose_small_fig_c, MSE)
    end
    for i in MSE_strict_CPA_exp_list
        MSE = mean(i)
        push!(MSE_strict_CPA_fig_c, MSE)
    end
    for i in MSE_strict_small_exp_list
        MSE = mean(i)
        push!(MSE_strict_small_fig_c, MSE)
    end
    for i in MSE_noise_exp_list
        MSE = mean(i)
        push!(MSE_noise_fig_c, MSE)
    end
    for i in MSE_brut_exp_list
        MSE = mean(i)
        push!(MSE_brut_fig_c, MSE)
    end

    ### save data ###
    df = DataFrame( sigma_h = sigma_h,MSE_loose_small = MSE_loose_small_fig_c, MSE_loose_CPA = MSE_loose_CPA_fig_c,  MSE_strict_small = MSE_strict_small_fig_c, MSE_strict_CPA = MSE_strict_CPA_fig_c, MSE_brut = MSE_brut_fig_c, MSE_noise = MSE_noise_fig_c)
    fn = format("pp-em/cdr/data/Fig_4c_noise_type={:s}_T={:.2f}_angdef={:.2f}_steps={:n}_nqubits={:n}_nsamples={:n}_minabs={:.2e}_minabsnoisy={:.2e}_depol={:.2e}_dephase={:.2e}.csv",  
    noise_kind,T, sigma_star, steps, nq, num_samples, min_abs_coeff, min_abs_coeff_noisy,depol_strength, dephase_strength)
    CSV.write(fn, df)
    plot_MSE_csv_data(fn,"sigma_h")
    ### Fig 4d ###

    MSE_loose_CPA_exp_list = []
    MSE_loose_CPA_ind_list = []
    MSE_loose_small_exp_list = []
    MSE_loose_small_ind_list = []
    MSE_noise_exp_list = []
    MSE_noise_ind_list = []
    MSE_strict_CPA_exp_list = []
    MSE_strict_CPA_ind_list = []
    MSE_strict_small_exp_list = []
    MSE_strict_small_ind_list = []
    MSE_brut_exp_list = []
    MSE_brut_ind_list = []

    for sig_J in sigma_J 
        J = -sig_J*steps/(2*T)
        for sig_h in sigma_h
            h = sig_h*steps/(2*T)
            
            trotter = trotter_setup(nq, steps, T, J, h);
            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            exact_expval_target, noisy_expval_target, corr_energy, rel_error_before, rel_error_after = full_run(trotter, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, min_abs_coeff_target=min_abs_coeff_target, observable = obs_magnetization(trotter), training_set = training_set, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double);
            MSE_ind = (exact_expval_target - corr_energy)^2
            push!(MSE_loose_CPA_ind_list, MSE_ind)

            MSE_noise = (exact_expval_target - noisy_expval_target)^2
            push!(MSE_noise_ind_list, MSE_noise)

            training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind; min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy, num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_loose_small_ind_list, MSE_ind)

            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "CPA", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_strict_CPA_ind_list, MSE_ind)
            
            training_set = training_set_generation_strict_perturbation(trotter, sigma_star; sample_function = "small", num_samples = num_samples);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)         
            push!(MSE_strict_small_ind_list, MSE_ind)

            training_set = training_set_generation_brut(trotter, sigma_star; num_samples = num_samples, non_replaced_gates=30);
            MSE_ind = run_method(trotter, training_set, sigma_star, noise_kind, min_abs_coeff=min_abs_coeff, min_abs_coeff_noisy=min_abs_coeff_noisy; num_samples=num_samples, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
            push!(MSE_brut_ind_list, MSE_ind)
        end

        push!(MSE_loose_CPA_exp_list, MSE_loose_CPA_ind_list)
        push!(MSE_loose_small_exp_list, MSE_loose_small_ind_list)
        push!(MSE_strict_CPA_exp_list, MSE_strict_CPA_ind_list)
        push!(MSE_strict_small_exp_list, MSE_strict_small_ind_list)
        push!(MSE_brut_exp_list, MSE_brut_ind_list)
        push!(MSE_noise_exp_list, MSE_noise_ind_list)

        MSE_loose_CPA_ind_list = []
        MSE_loose_small_ind_list = []
        MSE_strict_CPA_ind_list = []
        MSE_strict_small_ind_list = []
        MSE_brut_ind_list = []
        MSE_noise_ind_list = []
    end
    MSE_loose_CPA_fig_d = []
    MSE_loose_small_fig_d = []
    MSE_strict_CPA_fig_d = []
    MSE_strict_small_fig_d = []
    MSE_brut_fig_d = []
    MSE_noise_fig_d = []

    for i in MSE_loose_CPA_exp_list
        MSE = mean(i)
        push!(MSE_loose_CPA_fig_d, MSE)
    end
    for i in MSE_loose_small_exp_list
        MSE = mean(i)
        push!(MSE_loose_small_fig_d, MSE)
    end
    for i in MSE_strict_CPA_exp_list
        MSE = mean(i)
        push!(MSE_strict_CPA_fig_d, MSE)
    end
    for i in MSE_strict_small_exp_list
        MSE = mean(i)
        push!(MSE_strict_small_fig_d, MSE)
    end
    for i in MSE_noise_exp_list
        MSE = mean(i)
        push!(MSE_noise_fig_d, MSE)
    end
    for i in MSE_brut_exp_list
        MSE = mean(i)
        push!(MSE_brut_fig_d, MSE)
    end

    df = DataFrame(sigma_J = sigma_J,MSE_loose_small = MSE_loose_small_fig_d, MSE_loose_CPA = MSE_loose_CPA_fig_d,  MSE_strict_small = MSE_strict_small_fig_d, MSE_strict_CPA = MSE_strict_CPA_fig_d, MSE_brut = MSE_brut_fig_d, MSE_noise = MSE_noise_fig_d)
    fn = format("pp-em/cdr/data/Fig_4d_noise_type={:s}_T={:.2f}_angdef={:.2f}_steps={:n}_nqubits={:n}_nsamples={:n}_minabs={:.2e}_minabsnoisy={:.2e}_depol={:.2e}_dephase={:.2e}.csv",  
    noise_kind, T, sigma_star, steps, nq, num_samples, min_abs_coeff, min_abs_coeff_noisy,depol_strength, dephase_strength)
    CSV.write(fn, df)
    plot_MSE_csv_data(fn,"sigma_J")
    return 0
end

#fig_4cd()
