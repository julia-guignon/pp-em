using PauliPropagation
using Random
using Optim
using Plots
using ReverseDiff: GradientTape, gradient!, compile, gradient
using LinearAlgebra
using StatsBase 
using GLM
using DataFrames
using CSV
using Format
using Logging
using Distributions
using LaTeXStrings
using MLJLinearModels
using MLJBase
using Dates 

################# Logging Setup ####################
struct UnbufferedLogger <: Logging.AbstractLogger
    stream::IO
    level::Logging.LogLevel
end

const MainInfo = Base.CoreLogging.LogLevel(200)
const SubInfo = Base.CoreLogging.LogLevel(100)

const LOG_LEVEL_NAMES = Dict(
    Logging.Debug => "Debug",
    Logging.Info => "Info",
    Logging.Warn => "Warn",
    Logging.Error => "Error",
    SubInfo => "SubInfo",
    MainInfo => "MainInfo"
)

Logging.min_enabled_level(logger::UnbufferedLogger) = logger.level
Logging.shouldlog(logger::UnbufferedLogger, level, _module, group, id) = level ≥ logger.level
Logging.catch_exceptions(::UnbufferedLogger) = true

function Logging.handle_message(logger::UnbufferedLogger, level, message, _module, group, id, file, line; kwargs...)
    level_name = get(LOG_LEVEL_NAMES, level, "$level")  # Default for custom levels
    print(logger.stream, "[$level_name] ", message, "\n")  # Include log level
    flush(logger.stream)  # Ensure immediate output
end

#################### Structure and Setup ####################
struct trotter_ansatz_tfim
    target_circuit::Vector{Gate}
    target_circuit_layer::Vector{Gate}
    topology::Vector{Tuple{Int64, Int64}}
    nqubits::Integer
    steps::Integer #layers
    time::Float64
    J::Float64
    h::Float64
    sigma_J::Float64
    sigma_h::Float64
    sigma_J_indices::Vector{Int64}
    sigma_h_indices::Vector{Int64}
    sigma_J_indices_layer::Vector{Int64}
    sigma_h_indices_layer::Vector{Int64}
end

#to generate the structure
function kickedisingcircuit(nq, nl; topology=nothing)
    
    if isnothing(topology)
        topology = bricklayertopology(nq)
    end
    
    xlayer(circuit) = append!(circuit, (PauliRotation([:X], [qind]) for qind in 1:nq))
    zzlayer(circuit) = append!(circuit, (PauliRotation([:Z, :Z], pair, -π/2) for pair in topology))
    #Manuel does not have -pi/2 here!
    circuit = Gate[]

    for _ in 1:nl
        zzlayer(circuit)
        xlayer(circuit)
    end

    return circuit
end

function trotter_setup(nqubits::Integer, steps::Integer, time::Float64, J::Float64, h::Float64;topology = nothing)
    if isnothing(topology)
        topology = bricklayertopology(nqubits)
    end
    target_circuit = tfitrottercircuit(nqubits,steps,topology=topology) #starts with RZZ layer
    target_circuit_layer = tfitrottercircuit(nqubits,1,topology=topology) #starts with RZZ layer
    sigma_J = -2*time*J/steps
    sigma_h = 2*time*h/steps 

    sigma_J_indices = getparameterindices(target_circuit, PauliRotation, [:Z,:Z]) 
    sigma_h_indices = getparameterindices(target_circuit, PauliRotation, [:X])
    
    sigma_J_indices_layer = getparameterindices(target_circuit_layer, PauliRotation, [:Z,:Z])
    sigma_h_indices_layer = getparameterindices(target_circuit_layer, PauliRotation, [:X])
    
    return trotter_ansatz_tfim(target_circuit,target_circuit_layer, topology, nqubits, steps, time, J, h,sigma_J, sigma_h,sigma_J_indices, sigma_h_indices, sigma_J_indices_layer, sigma_h_indices_layer)
end

#setup for fixed \theta_J angle (less params)
function trotter_kickedising_setup(nqubits::Integer, steps::Integer, time::Float64, h::Float64;topology = nothing)
    if isnothing(topology)
        topology = bricklayertopology(nqubits)
    end
    target_circuit = kickedisingcircuit(nqubits,steps,topology=topology) #starts with RZZ layer
    target_circuit_layer = kickedisingcircuit(nqubits,1,topology=topology) #starts with RZZ layer
    sigma_J = -pi/2 
    J = -sigma_J*steps/(2*time)
    sigma_h = 2*time*h/steps 

    sigma_J_indices = getparameterindices(target_circuit, PauliRotation, [:Z,:Z]) 
    sigma_h_indices = getparameterindices(target_circuit, PauliRotation, [:X])
    
    sigma_J_indices_layer = getparameterindices(target_circuit_layer, PauliRotation, [:Z,:Z])
    sigma_h_indices_layer = getparameterindices(target_circuit_layer, PauliRotation, [:X])
    
    return trotter_ansatz_tfim(target_circuit,target_circuit_layer, topology, nqubits, steps, time, J, h,sigma_J, sigma_h,sigma_J_indices, sigma_h_indices, sigma_J_indices_layer, sigma_h_indices_layer)
end


function constrain_params(ansatz; layer=false)
    """
    Set all RX gates and all RZZ gates to have the same parameter value respectively.
    """
    if layer
        nparams = countparameters(ansatz.target_circuit_layer)
        thetas = zeros(nparams)
        thetas[ansatz.sigma_h_indices_layer] .= ansatz.sigma_h
        thetas[ansatz.sigma_J_indices_layer] .= ansatz.sigma_J
    else
        nparams = countparameters(ansatz.target_circuit)
        thetas = zeros(nparams)
        thetas[ansatz.sigma_h_indices] .= ansatz.sigma_h
        thetas[ansatz.sigma_J_indices] .= ansatz.sigma_J
    end
    
    return thetas
end

function obs_interaction(ansatz)
    interaction = PauliSum(ansatz.nqubits)
    
    for i in 1:length(ansatz.topology)
        q1 = ansatz.topology[i][1]
        q2 = ansatz.topology[i][2]
        add!(interaction, [:Z, :Z], [q1, q2])
    end
    return interaction/length(ansatz.topology)
end

function obs_magnetization(ansatz)
    """
    Returns the normalised magnetization.
    """
    magnetization = PauliSum(ansatz.nqubits)
    for i in 1:ansatz.nqubits
        add!(magnetization,:Z,i)
    end
    magnetization = magnetization/ansatz.nqubits
    return magnetization
end

function training_set_generation_brut(ansatz::trotter_ansatz_tfim, angle_definition::Float64=pi/2; num_samples::Int = 10, non_replaced_gates::Int = 30)
    """
    Generates a training set of thetas for the ansatz. The training set is generated by selecting a number of cliffords and non-cliffords
    and setting the corresponding thetas to multiples of angle_definition. 
    """
    nparams = countparameters(ansatz.target_circuit)
    replaced_gates = nparams - non_replaced_gates
    ratio = length(ansatz.sigma_J_indices)/(length(ansatz.sigma_h_indices)+length(ansatz.sigma_J_indices))
    num_h = Int(round((1-ratio)*replaced_gates))
    num_J = Int(round(ratio*replaced_gates))
    training_thetas_list = Vector{Vector{Float64}}()
    thetas = constrain_params(ansatz)
    k_h =round(ansatz.sigma_h/(angle_definition))
    k_J =round(ansatz.sigma_J/(angle_definition))
    
    
    for _ in 1:num_samples
        training_thetas = deepcopy(thetas)
        shuffled_sigma_h_indices =  Random.shuffle!(ansatz.sigma_h_indices)
        shuffled_sigma_J_indices = Random.shuffle!(ansatz.sigma_J_indices)
        selected_indices_h = shuffled_sigma_h_indices[1:num_h]
        selected_indices_J = shuffled_sigma_J_indices[1:num_J]

        for i in selected_indices_h
            training_thetas[i] = k_h*angle_definition
        end
        for i in selected_indices_J
            training_thetas[i] = k_J*angle_definition
        end
        push!(training_thetas_list, training_thetas)
    end
    return training_thetas_list
end

function training_set_generation_strict_perturbation(ansatz::trotter_ansatz_tfim,sigma_star::Float64 = pi/20; sample_function = nothing, num_samples::Int = 10)
    """
    Generates a training set according to the CPA approach. We do not use data augmentation here and stick to standard CPA.
    Their bound holds only if we replace all gates (we can't keep original gates).
    """
    function sample_theta_CPA(sigma_star)
        # sig_h ∈ [0, sigma_star] ∪ [π/2 - sigma_star, π/2]
        sig_h = rand(Bool) ? rand(Uniform(0.0, sigma_star)) : rand(Uniform(π/2 - sigma_star, π/2))
    
        # sig_J ∈ [−sigma_star, 0] ∪ [−π/2, −π/2 + sigma_star]
        sig_J = rand(Bool) ? rand(Uniform(-sigma_star, 0.0)) : rand(Uniform(-π/2, -π/2 + sigma_star))
    
        return sig_h, sig_J
    end

    function sample_theta_small(sigma_star)
        # sig_h ∈ [0, sigma_star] ∪ [π/2 - sigma_star, π/2]
        sig_h =  rand(Uniform(0.0, sigma_star)) 
    
        # sig_J ∈ [−sigma_star, 0] ∪ [−π/2, −π/2 + sigma_star]
        sig_J = rand(Uniform(-sigma_star, 0.0))
    
        return sig_h, sig_J
    end
    
    if sample_function == nothing
        sample_function = sample_theta_CPA
    elseif sample_function == "CPA"
        sample_function = sample_theta_CPA
    elseif sample_function == "small"
        sample_function = sample_theta_small
    end


    training_thetas_list = Vector{Vector{Float64}}()
    thetas = constrain_params(ansatz)
    training_thetas = deepcopy(thetas)
    
    for _ in 1:num_samples
        sig_h_perturbed, sig_J_perturbed = sample_function(sigma_star)
        training_thetas[ansatz.sigma_h_indices] .= sig_h_perturbed
        training_thetas[ansatz.sigma_J_indices] .= sig_J_perturbed
        push!(training_thetas_list, copy(training_thetas))
    
    end

    return training_thetas_list
end

function training_set_generation_loose_perturbation(ansatz::trotter_ansatz_tfim,sigma_star::Float64=pi/20; sample_function = nothing, num_samples::Int = 10)
    """
    Generates a training set according to the CPA approach. We do not use data augmentation here and stick to standard CPA.
    Their bound holds only if we replace all gates (we can't keep original gates).
    """

    if !(0.0 <= ansatz.sigma_h <= sigma_star) && !(pi/2 - sigma_star <= ansatz.sigma_h <= pi/2)
        change_sigma_h = true
    else
        change_sigma_h = false
    end
    if !(-sigma_star <= ansatz.sigma_J <= 0.0) && !(-pi/2 <= ansatz.sigma_J <= -pi/2 + sigma_star)
        change_sigma_J = true
    else
        change_sigma_J = false
    end
    
    function sample_theta_CPA(sigma_star)
        # sig_h ∈ [0, sigma_star] ∪ [π/2 - sigma_star, π/2]
        sig_h = rand(Bool) ? rand(Uniform(0.0, sigma_star)) : rand(Uniform(π/2 - sigma_star, π/2))
    
        # sig_J ∈ [−sigma_star, 0] ∪ [−π/2, −π/2 + sigma_star]
        sig_J = rand(Bool) ? rand(Uniform(-sigma_star, 0.0)) : rand(Uniform(-π/2, -π/2 + sigma_star))
    
        return sig_h, sig_J
    end

    function sample_theta_small(sigma_star)
        # sig_h ∈ [0, sigma_star]
        sig_h =  rand(Uniform(0.0, sigma_star)) 
    
        # sig_J ∈ [−sigma_star, 0] ∪ [−π/2, −π/2 + sigma_star]
        sig_J = rand(Uniform(-sigma_star, 0.0))
    
        return sig_h, sig_J
    end
    
    if sample_function == nothing
        sample_function = sample_theta_small
    elseif sample_function == "CPA"
        sample_function = sample_theta_CPA
    elseif sample_function == "small"
        sample_function = sample_theta_small
    end


    training_thetas_list = Vector{Vector{Float64}}()
    thetas = constrain_params(ansatz)
    training_thetas = deepcopy(thetas)
    
    for _ in 1:num_samples
        if change_sigma_h
            sig_h_perturbed, _ = sample_function(sigma_star)
        else 
            sig_h_perturbed = ansatz.sigma_h
        end
        if change_sigma_J
            _, sig_J_perturbed = sample_function(sigma_star)
        else 
            sig_J_perturbed = ansatz.sigma_J
        end
        training_thetas[ansatz.sigma_h_indices] .= sig_h_perturbed
        training_thetas[ansatz.sigma_J_indices] .= sig_J_perturbed
        push!(training_thetas_list, copy(training_thetas))
    
    end

    return training_thetas_list
end

function trotter_time_evolution(ansatz; observable = nothing, special_thetas=nothing, noise_kind="noiseless", record=false, min_abs_coeff=0.0, max_weight = Inf, noise_level = 1,  depol_strength=0.01, dephase_strength=0.01,depol_strength_double=0.0033, dephase_strength_double=0.0033) 
    """
    Function that computes the time evolution of the ansatz using the first order Trotter approximation exact time evolution operator.
    The function returns the overlap of the final state with the |0> state.
    """
    if observable==nothing
        obs = obs_interaction(ansatz)
    else
        obs = deepcopy(observable)
    end

    if special_thetas==nothing
        thetas = constrain_params(ansatz)
    else
        thetas = special_thetas
    end

    if noise_kind=="naive"
        if record
            error("Naive noise model doesn't support recording :(")
        else
            circuit = final_noise_layer_circuit(ansatz; depol_strength = noise_level*depol_strength,dephase_strength = noise_level*dephase_strength)
        end

    elseif noise_kind=="gate_ising"
        circuit = gate_noise_circuit(ansatz; depol_strength = noise_level*depol_strength, dephase_strength = noise_level*dephase_strength, layer=record)

    elseif noise_kind=="noiseless"
        if record
            circuit = ansatz.target_circuit_layer
        else
            circuit = ansatz.target_circuit
        end
        
    elseif noise_kind=="realistic_ising"
        circuit = realistic_gate_noise_circuit(ansatz; depol_strength_single = noise_level*depol_strength, dephase_strength_single = noise_level*dephase_strength, depol_strength_double = noise_level*depol_strength_double, dephase_strength_double = noise_level*dephase_strength_double, layer = record)

    elseif noise_kind=="gate_kickedising"
        circuit = kicked_gate_noise_circuit(ansatz; depol_strength = noise_level*depol_strength, dephase_strength = noise_level*dephase_strength, layer = record)

    elseif noise_kind=="realistic_kickedising"
        circuit = realistic_kicked_gate_noise_circuit(ansatz; depol_strength_single = noise_level*depol_strength, dephase_strength_single = noise_level*dephase_strength, depol_strength_double = noise_level*depol_strength_double, dephase_strength_double = noise_level*dephase_strength_double, layer = record)

    else
        error("Noise kind $(noise_kind) unknown.")
    end

    if record
        nparams = countparameters(ansatz.target_circuit)
        expvals_trotter = Float64[]   
        push!(expvals_trotter, overlapwithzero(obs))
        for i in 1:ansatz.steps
            psum = propagate!(circuit, obs, thetas[Int(nparams/ansatz.steps*(i-1)+1):Int(nparams/ansatz.steps*i)];min_abs_coeff=min_abs_coeff, max_weight = max_weight)
            push!(expvals_trotter, overlapwithzero(psum))
        end
        return expvals_trotter  
    else 
        psum = propagate!(circuit, obs,  thetas; min_abs_coeff=min_abs_coeff, max_weight = max_weight)
        return overlapwithzero(psum)
    end
end

######### ZNE isolated implementation ##########
function zne_time_evolution(ansatz::trotter_ansatz_tfim;observable = nothing, noise_kind="noiseless", min_abs_coeff=0.0, max_weight = Inf, noise_levels = [1,1.5,2.0], depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033, record = false)

    if record
        noisy_expvals = Array{Float64,2}(undef, length(noise_levels), ansatz.steps+1)
    else
        noisy_expvals = Array{Float64,1}(undef, length(noise_levels))
    end

    for (idx,i) in enumerate(noise_levels)
        noisy_expval_target = trotter_time_evolution(ansatz; observable = observable, record=record,noise_kind=noise_kind, noise_level = i,min_abs_coeff=min_abs_coeff,max_weight = max_weight, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
        #println("Noisy expval with noise level $(i): ", noisy_expval_target)
        if record
            for j in 1:length(noisy_expval_target)
                noisy_expvals[idx,:] .= noisy_expval_target
            end
        else
            for j in 1:length(noisy_expval_target)
            noisy_expvals[idx] = noisy_expval_target[end]
            end
        end
    end
    return noisy_expvals
end

# 1st method of ZNE
function zne(noisy_exp::Vector{Float64}; noise_levels = [1,1.5,2.0], fit_type = "linear", exact_target_exp_value::Union{Nothing, Float64}=nothing, use_target::Bool=true)

    training_data = DataFrame(x=noise_levels, y= noisy_exp)
    if fit_type == "linear"
        ols = lm(@formula(y ~ x), training_data)
        cdr_em(x) = coef(ols)[1] + coef(ols)[2] * x
        corrected = cdr_em(0.0)
    end

    # ToDo: add polynomial fit (Richardson extrapolation) for comparison
    
    if use_target && exact_target_exp_value !== nothing
        rel_error_after = abs(exact_target_exp_value - corrected) / abs(exact_target_exp_value)
        rel_error_before = abs(exact_target_exp_value - noisy_exp[1]) / abs(exact_target_exp_value) #fixed to the first noise_level to be one
        return corrected, rel_error_after, rel_error_before
    else
        return corrected

    end
end

# 2nd method of ZNE
function zne(noisy_exp::Matrix{Float64}; noise_levels = [1,1.5,2.0], fit_type = "linear", exact_target_exp_value::Union{Nothing, Vector{Float64}}=nothing, use_target::Bool=true)
    nsteps = size(noisy_exp,2) #this is one more than nsteps
    corrected = Vector{Float64}(undef, nsteps) # undef allocates memory
    rel_errors_after = Vector{Float64}()
    rel_errors_before = Vector{Float64}()
    for i in 2:nsteps
        result = zne(noisy_exp[:,i]; noise_levels = noise_levels, 
        fit_type = fit_type,
         exact_target_exp_value = use_target ? (exact_target_exp_value === nothing ? nothing : exact_target_exp_value[i]) : nothing,
         use_target = use_target)
         if use_target && exact_target_exp_value !== nothing
            corrected[i], err_after, err_before = result
            push!(rel_errors_after, err_after)
            push!(rel_errors_before, err_before)
         else
            corrected[i] = result
            #corrected[i] = max(corrected[i], 1e-16)
        end

    end

        return use_target ? (corrected, rel_errors_after, rel_errors_before) : corrected

end


function training_trotter_time_evolution(ansatz::trotter_ansatz_tfim, training_thetas::Vector{Vector{Float64}};observable = nothing, noise_kind="noiseless", min_abs_coeff=0.0, max_weight = Inf, noise_level = 1, depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033, record = false)
    """
    Function that computes the time evolution of the ansatz using the first order Trotter approximation exact time evolution operator.
    The function returns the overlap of the final state with the |0> state.
    """
    if record 
        exact_expvals = Vector{Vector{Float64}}()
    else
        exact_expvals = Vector{Float64}()
    end
    for thetas in training_thetas
        push!(exact_expvals, trotter_time_evolution(ansatz; observable = observable, record=record, special_thetas=thetas, noise_kind=noise_kind, min_abs_coeff=min_abs_coeff,max_weight = max_weight, noise_level = noise_level, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double))
    end
    return exact_expvals
end


###### vnCDR (ZNE and CDR combined) ##########

function vnCDR_training_trotter_time_evolution(ansatz::trotter_ansatz_tfim, training_thetas::Vector{Vector{Float64}}; observable=nothing, noise_kind="noiseless", min_abs_coeff=0.0, max_weight=Inf, noise_levels=[1, 1.5,2.0], depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033, record=false)
    """
    Function that computes the training data for several noise levels.
    If record=true, stores full time evolution; else only final values.
    """

    if record
        exact_expvals = Array{Float64,3}(undef, length(noise_levels), length(training_thetas), ansatz.steps+1) # 3D array: (noise_levels, circuits, steps+1)
    else
        exact_expvals = Array{Float64,2}(undef, length(noise_levels), length(training_thetas)) # 2D array: (noise_levels, circuits)
    end

    for (idx, i) in enumerate(noise_levels) # use enumerate to get valid index
        noisy_training = training_trotter_time_evolution(ansatz, training_thetas; observable=observable, noise_kind=noise_kind, record=record, min_abs_coeff=min_abs_coeff, max_weight=max_weight, noise_level=i, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
        println("Noisy expval at noise level $(i): ", noisy_training)

        if record
            for j in 1:length(training_thetas)
                exact_expvals[idx, j, :] .= noisy_training[j] # full trajectory
            end
        else
            for j in 1:length(training_thetas)
                exact_expvals[idx, j] = noisy_training[j][end] # only final value
            end
        end
    end
    return exact_expvals
end

function final_noise_layer_circuit(ansatz; depol_strength=0.05, dephase_strength=0.05)
    """
    Function that adds a final layer of depolarizing and dephasing noise to the ansatz.
    """
    depol_noise_layer = [DepolarizingNoise(qind, depol_strength ) for qind in 1:ansatz.nqubits];
    dephase_noise_layer = [DephasingNoise(qind, dephase_strength) for qind in 1:ansatz.nqubits];
    noisy_circuit = deepcopy(ansatz.target_circuit)
    append!(noisy_circuit,depol_noise_layer)
    append!(noisy_circuit,dephase_noise_layer)

    return noisy_circuit
end

function gate_noise_circuit(ansatz; depol_strength=0.01, dephase_strength=0.01, start_with_ZZ=true, layer=false)
    #can also only generate one layer of noisy circuit (in case we want to record, we can calc the expecatation value after each step)
    """
    Noise model from the CPA paper, where we add a layer of depolarizing and dephasing noise after each step/layer of the ansatz.
    """
    circuit::Vector{Gate} = []
    if layer
        steps = 1
    else
        steps = ansatz.steps
    end

    # the function after this expects a circuit with at least one layer and will always append something
    if steps<1
        error("The number of steps should be at least 1 (steps=$steps).")
    end

    depol_noise_layer = [DepolarizingNoise(qind, depol_strength ) for qind in 1:ansatz.nqubits];
    phase_damp_layer = [DephasingNoise(qind, dephase_strength) for qind in 1:ansatz.nqubits];

    if start_with_ZZ
        rzzlayer!(circuit, ansatz.topology)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
    end

    for _ in 1:steps-1
        rxlayer!(circuit, ansatz.nqubits)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
        rzzlayer!(circuit, ansatz.topology)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
    end

    rxlayer!(circuit, ansatz.nqubits)
    append!(circuit, depol_noise_layer)
    append!(circuit, phase_damp_layer)

    if !start_with_ZZ
        rzzlayer!(circuit, ansatz.topology)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
    end

    return circuit
end

function kicked_gate_noise_circuit(ansatz; depol_strength=0.01, dephase_strength=0.01, start_with_ZZ=true, layer=false)
    #can also only generate one layer of noisy circuit (in case we want to record, we can calc the expecatation value after each step)
    """
    Noise model from the CPA paper, where we add a layer of depolarizing and dephasing noise after each step/layer of the ansatz.
    """
    circuit::Vector{Gate} = []
    if layer
        steps = 1
    else
        steps = ansatz.steps
    end

    # the function after this expects a circuit with at least one layer and will always append something
    if steps<1
        error("The number of steps should be at least 1 (steps=$steps).")
    end

    depol_noise_layer = [DepolarizingNoise(qind, depol_strength ) for qind in 1:ansatz.nqubits];
    phase_damp_layer = [DephasingNoise(qind, dephase_strength) for qind in 1:ansatz.nqubits];
    xlayer(circuit) = append!(circuit, (PauliRotation([:X], [qind]) for qind in 1:ansatz.nqubits))
    zzlayer(circuit) = append!(circuit, (PauliRotation([:Z, :Z], pair, π/2) for pair in ansatz.topology))

    
    if start_with_ZZ
        zzlayer(circuit)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
    end

    for _ in 1:steps-1
        xlayer(circuit)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
        zzlayer(circuit)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
    end

    xlayer(circuit)
    append!(circuit, depol_noise_layer)
    append!(circuit, phase_damp_layer)

    if !start_with_ZZ
        zzlayer(circuit)
        append!(circuit, depol_noise_layer)
        append!(circuit, phase_damp_layer)
    end

    return circuit
end

function realistic_gate_noise_circuit(ansatz; depol_strength_double=0.0033, dephase_strength_double=0.0033, depol_strength_single=0.00035, dephase_strength_single=0.00035, start_with_ZZ=true, layer=false)
    """
    Noise model from the CPA paper, where we add a layer of depolarizing and dephasing noise after each step/layer of the ansatz.
    """
    circuit::Vector{Gate} = []
    if layer
        steps = 1
    else
        steps = ansatz.steps
    end

    # the function after this expects a circuit with at least one layer and will always append something
    if steps<1
        error("The number of steps should be at least 1 (steps=$steps).")
    end

    depol_noise_layer_single = [DepolarizingNoise(qind, depol_strength_single ) for qind in 1:ansatz.nqubits];
    phase_damp_layer_single = [DephasingNoise(qind, dephase_strength_single) for qind in 1:ansatz.nqubits];
    depol_noise_layer_double = [DepolarizingNoise(qind, depol_strength_double ) for qind in 1:ansatz.nqubits];
    phase_damp_layer_double = [DephasingNoise(qind, dephase_strength_double) for qind in 1:ansatz.nqubits];

    if start_with_ZZ
        rzzlayer!(circuit, ansatz.topology)
        append!(circuit, depol_noise_layer_double)
        append!(circuit, phase_damp_layer_double)
        
    end

    for _ in 1:steps-1
        rxlayer!(circuit, ansatz.nqubits)
        append!(circuit, depol_noise_layer_single)
        append!(circuit, phase_damp_layer_single)
        rzzlayer!(circuit, ansatz.topology)
        append!(circuit, depol_noise_layer_double)
        append!(circuit, phase_damp_layer_double)
    end

    rxlayer!(circuit, ansatz.nqubits)
    append!(circuit, depol_noise_layer_single)
    append!(circuit, phase_damp_layer_single)

    if !start_with_ZZ
        rzzlayer!(circuit, ansatz.topology)
        append!(circuit, depol_noise_layer_double)
        append!(circuit, phase_damp_layer_double)
    end

    return circuit
end

function realistic_kicked_gate_noise_circuit(ansatz; depol_strength_double=0.0033, dephase_strength_double=0.0033, depol_strength_single=0.00035, dephase_strength_single=0.00035, start_with_ZZ=true, layer=false)
    """
    Noise model from the CPA paper, where we add a layer of depolarizing and dephasing noise after each step/layer of the ansatz.
    """
    circuit::Vector{Gate} = []
    if layer
        steps = 1
    else
        steps = ansatz.steps
    end

    # the function after this expects a circuit with at least one layer and will always append something
    if steps<1
        error("The number of steps should be at least 1 (steps=$steps).")
    end

    depol_noise_layer_single = [DepolarizingNoise(qind, depol_strength_single ) for qind in 1:ansatz.nqubits];
    phase_damp_layer_single = [DephasingNoise(qind, dephase_strength_single) for qind in 1:ansatz.nqubits];
    depol_noise_layer_double = [DepolarizingNoise(qind, depol_strength_double ) for qind in 1:ansatz.nqubits];
    phase_damp_layer_double = [DephasingNoise(qind, dephase_strength_double) for qind in 1:ansatz.nqubits];

    xlayer(circuit) = append!(circuit, (PauliRotation([:X], [qind]) for qind in 1:nq))
    zzlayer(circuit) = append!(circuit, (PauliRotation([:Z, :Z], pair, π/2) for pair in ansatz.topology))


    if start_with_ZZ
        zzlayer(circuit)
        append!(circuit, depol_noise_layer_double)
        append!(circuit, phase_damp_layer_double)
        
    end

    for _ in 1:steps-1
        xlayer(circuit)
        append!(circuit, depol_noise_layer_single)
        append!(circuit, phase_damp_layer_single)
        zzlayer(circuit)
        append!(circuit, depol_noise_layer_double)
        append!(circuit, phase_damp_layer_double)
    end

    xlayer(circuit)
    append!(circuit, depol_noise_layer_single)
    append!(circuit, phase_damp_layer_single)

    if !start_with_ZZ
        zzlayer(circuit)
        append!(circuit, depol_noise_layer_double)
        append!(circuit, phase_damp_layer_double)
    end

    return circuit
end

### 3 methods for CDR
# CDR only for last value (was correct)
function cdr(
    noisy_exp_values::Vector{Float64},
    exact_exp_values::Vector{Float64},
    noisy_target_exp_value::Float64;
    exact_target_exp_value::Union{Nothing, Float64}=nothing,
    use_target::Bool=true)
    training_data = DataFrame(x=noisy_exp_values, y=exact_exp_values)
    ols = lm(@formula(y ~ x), training_data)
    cdr_em(x) = coef(ols)[1] + coef(ols)[2] * x

    corrected = cdr_em(noisy_target_exp_value)

    if use_target && exact_target_exp_value !== nothing
        rel_error_after = abs(exact_target_exp_value - corrected) / abs(exact_target_exp_value)
        rel_error_before = abs(exact_target_exp_value - noisy_target_exp_value) / abs(exact_target_exp_value)
        return corrected, rel_error_after, rel_error_before
    else
        return corrected
    end
end

# CDR for all steps in time evolution
function cdr(
    noisy_exp_values::Vector{Vector{Float64}},
    exact_exp_values::Vector{Vector{Float64}},
    noisy_target_exp_value::Vector{Float64};
    exact_target_exp_value::Union{Nothing, Vector{Float64}}=nothing,
    use_target::Bool=true)
    nsteps = length(noisy_target_exp_value)
    corrected = Vector{Float64}(undef, nsteps)
    rel_errors_after = Float64[]
    rel_errors_before = Float64[]
    println("nsteps", nsteps)
    
    for i in 2:nsteps
        println("i", i)
        exact_exp_values_last  = [row[i] for row in exact_exp_values] # necessary for nested vecotr format
        noisy_exp_values_last  = [row[i] for row in noisy_exp_values]

        println("noisy_exp_values_last", i, noisy_exp_values_last)
        println("exact_exp_values_last", exact_exp_values_last)
        println("noisy_target_exp_value", noisy_target_exp_value[i])
        println("exact_target_exp_value", exact_target_exp_value[i])
        

        result = cdr(
            noisy_exp_values_last,
            exact_exp_values_last,
            noisy_target_exp_value[i];
            exact_target_exp_value = use_target ? exact_target_exp_value === nothing ? nothing : exact_target_exp_value[i] : nothing,
            use_target = use_target
        )
        if use_target && exact_target_exp_value !== nothing
            corrected[i], err_after, err_before = result
            push!(rel_errors_after, err_after)
            push!(rel_errors_before, err_before)
        else
            corrected[i] = result
        end
    end

    return use_target ? (corrected, rel_errors_after, rel_errors_before) : corrected
end

# CDR with weighted linear regression
function cdr(
    noisy_exp_values::Vector{Vector{Float64}},
    exact_exp_values::Vector{Vector{Float64}},
    noisy_target_exp_value::Vector{Float64},
    decay_weights::Vector{Vector{Float64}};
    exact_target_exp_value::Union{Nothing, Vector{Float64}}=nothing,
    use_target::Bool=true)
    nsteps = length(noisy_exp_values[1])
    ncircuits = length(noisy_exp_values)
    corrected = Vector{Float64}(undef, nsteps)
    rel_errors_after = Float64[]
    rel_errors_before = Float64[]

    for t in 2:nsteps
        x_all, y_all, w_all = Float64[], Float64[], Float64[]
        for c in 1:ncircuits, τ in 1:t
            push!(x_all, noisy_exp_values[c][τ])
            push!(y_all, exact_exp_values[c][τ])
            push!(w_all, decay_weights[t][τ])
        end
        df = DataFrame(x = x_all, y = y_all, w = w_all)

        ols = lm(@formula(y ~ x), df, wts = df.w)
        cdr_em(x) = coef(ols)[1] + coef(ols)[2] * x

        corrected[t] = cdr_em(noisy_target_exp_value[t])
        if use_target && exact_target_exp_value !== nothing
            err_after = abs(exact_target_exp_value[t] - corrected[t]) / abs(exact_target_exp_value[t])
            err_before = abs(exact_target_exp_value[t] - noisy_target_exp_value[t]) / abs(exact_target_exp_value[t])
            push!(rel_errors_after, err_after)
            push!(rel_errors_before, err_before)
        end

    end

    return use_target ? (corrected, rel_errors_after, rel_errors_before) : corrected
end

#### vnCDR optimization function ####

## 1st method: computes only final vnCDR corrected value
function vnCDR(
    noisy_exp_values::Array{Float64,2},        # size (m circuits, n+1 noise levels)
    exact_exp_values::Vector{Float64},         # size m
    noisy_target_exp_value::Vector{Float64};  # size n+1
    exact_target_exp_value::Union{Nothing, Float64}=nothing,
    use_target::Bool=true,
    lambda::Float64=0.0                         # regularization strength
)
    model = lambda == 0.0 ? LinearRegressor(fit_intercept = false) : RidgeRegressor(lambda=lambda,fit_intercept = false)


    # Convert input matrix to DataFrame
    X = DataFrame(noisy_exp_values', :auto)

    println("X", X)
    println("exact_exp_values", exact_exp_values)
    mach = machine(model, X, exact_exp_values)
    fit!(mach)
    params = fitted_params(mach)
    println("params", params)

    # Manually compute prediction
    coefs = [v for (_, v) in fitted_params(mach).coefs]
    println("coefs", coefs)
    println("noisy_target_exp_value", noisy_target_exp_value)   
    pred = coefs'* noisy_target_exp_value
    println("pred" , pred)
    
    if use_target && exact_target_exp_value !== nothing
        rel_error_after = abs(exact_target_exp_value - pred) / abs(exact_target_exp_value)
        rel_error_before = abs(exact_target_exp_value - noisy_target_exp_value[end]) / abs(exact_target_exp_value)
        return pred, rel_error_after, rel_error_before
    else
        return pred
    end
end

##2nd method: vnCDR for every step
function vnCDR(
    noisy_exp_values::Array{Float64,3},  # (matrix) from via vnCDR_training_trotter_time_evolution      # size (n+1 noise levels, m circuits, t+1 steps)
    exact_exp_values::Vector{Vector{Float64}},  # from  trotter_time_evolution!!
    noisy_target_exp_value::Array{Float64,2}; # (matrix) from zne_time_evolution
    exact_target_exp_value::Union{Nothing, Vector{Float64}}=nothing,
    use_target::Bool=true,
    lambda::Float64=0.0
)
    #println("size exact_exp_vals ",size(exact_exp_values))
    nsteps = size(noisy_exp_values, 3)

    println("nsteps" , nsteps)
    corrected = Vector{Float64}(undef, nsteps)
    rel_errors_after = Float64[] # vector 
    rel_errors_before = Float64[] 
    for i in 2:nsteps
        # println("inner function: size 1st argument ",size(transpose(noisy_exp_values[:, :, i])))
        # println("type of noisy_exp_values[:,:,i]",typeof(noisy_exp_values[:, :, i]))
        exact_exp_values_last  = [row[i] for row in exact_exp_values]

        # println("exact training ",size(exact_exp_values_last))
        # println("type of noisy target exp value ",typeof(noisy_target_exp_value[i]))
        # println("type of exact target exp value ",typeof(exact_target_exp_value[i]))

        # println("perm ",size(permutedims(noisy_exp_values[:, :, i], (2, 1))))
        result = vnCDR(
        #permutedims(noisy_exp_values[:, :, i], (2, 1)),
        noisy_exp_values[:, :, i],
        exact_exp_values_last,
        noisy_target_exp_value[:,i];
        exact_target_exp_value = use_target ? (exact_target_exp_value === nothing ? nothing : exact_target_exp_value[i]) : nothing,
        use_target = use_target,
        lambda = lambda
        )

        

        if use_target && exact_target_exp_value !== nothing
            corrected[i], err_after, err_before = result
            println("corrected[i]", corrected[i])
            println("err_after", err_after)
            println("err_before", err_before)
            push!(rel_errors_after, err_after)
            push!(rel_errors_before, err_before)
        else
            corrected[i] = result
        end
    end

    return use_target ? (corrected, rel_errors_after, rel_errors_before) : corrected
end


function full_run(ansatz, angle_definition::Float64, noise_kind::String;
    min_abs_coeff::Float64 = 0.0, min_abs_coeff_noisy=0.0, max_weight = Inf, training_set = nothing,
    observable = nothing, num_samples=10, non_replaced_gates=30,
    depol_strength=0.01, dephase_strength=0.01,
    depol_strength_double=0.0033, dephase_strength_double=0.0033,
    min_abs_coeff_target = 0.0, record = false,
    cdr_method = "end", use_target = true, real_qc_noisy_data = nothing)
    @logmsg SubInfo "ready to ruuuuuummmble"

    #name for the observable in the logger
    obs_62 = PauliSum(ansatz.nqubits)
    add!(obs_62, :Z, 62)
    obs_string = ""

    if observable === nothing || observable == obs_interaction(ansatz)
        observable = obs_interaction(ansatz)
        obs_string = "ZZ"
    elseif observable == obs_magnetization(ansatz)
        obs_string = "Z"
    elseif observable == obs_62
        obs_string = "Z_62"
    end

    if training_set === nothing
        training_set = training_set_generation_strict_perturbation(ansatz, angle_definition; num_samples=num_samples)
    end 

    if use_target # this is the expensive part of the computation
        time1 = time()
        exact_expval_target = trotter_time_evolution(ansatz; observable = observable, noise_kind="noiseless", min_abs_coeff = min_abs_coeff_target, record = record)
        timetmp1 = time()
        @logmsg SubInfo "exact_expval_target done in $(round(timetmp1 - time1; digits = 2)) s"

        noisy_expval_target = trotter_time_evolution(ansatz; observable = observable, noise_kind=noise_kind, min_abs_coeff = min_abs_coeff_target, record = record)
        timetmp2 = time()
        @logmsg SubInfo "noisy_expval_target done in $(round(timetmp2 - timetmp1; digits = 2)) s"
        timetmp1 = timetmp2
    else
        time1 = time()
        exact_expval_target = NaN
        if real_qc_noisy_data !== nothing
            noisy_expval_target = real_qc_noisy_data
        else
            noisy_expval_target = NaN
        time1 = timetmp1 = time()
        end
    end

    exact_expvals = training_trotter_time_evolution(ansatz, training_set; observable = observable, noise_kind="noiseless", min_abs_coeff=min_abs_coeff, max_weight = max_weight, record = record)
    timetmp2 = time()
    @logmsg SubInfo "exact_training_time_evolution done in $(round(timetmp2 - timetmp1; digits = 2)) s"
    timetmp1 = timetmp2

    noisy_expvals = training_trotter_time_evolution(ansatz, training_set; observable = observable, noise_kind=noise_kind, min_abs_coeff=min_abs_coeff_noisy, max_weight = max_weight, depol_strength=depol_strength, dephase_strength=dephase_strength, record = record)
    timetmp2 = time()
    @logmsg SubInfo "noisy_training_time_evolution done in $(round(timetmp2 - timetmp1; digits = 2)) s"
    timetmp1 = timetmp2

    if !use_target && cdr_method != "end"
        error("Use target is false and the cdr_method is not set to end.")
    end

    if !record && cdr_method == "end"
        corr_exp_result = cdr(noisy_expvals, exact_expvals, noisy_expval_target[end]; exact_target_exp_value = use_target ? exact_expval_target[end] : nothing, use_target = use_target)
    
    elseif record && cdr_method == "end"
        noisy_expvals_last = [noisy_expvals[i][end] for i in eachindex(noisy_expvals)]
        exact_expvals_last = [exact_expvals[i][end] for i in eachindex(exact_expvals)]
        corr_exp_result = cdr(noisy_expvals_last, exact_expvals_last, noisy_expval_target[end]; exact_target_exp_value = use_target ? exact_expval_target[end] : nothing, use_target = use_target)

    elseif cdr_method == "std_LR"
        corr_exp_result = cdr(noisy_expvals, exact_expvals, noisy_expval_target; exact_target_exp_value = use_target ? exact_expval_target : nothing, use_target = use_target)
    elseif cdr_method == "lin_WLR"
        decay_weights = [[τ / t for τ in 1:t] for t in 1:length(noisy_expvals[1])]
        corr_exp_result = cdr(noisy_expvals, exact_expvals, noisy_expval_target, decay_weights; exact_target_exp_value = use_target ? exact_expval_target : nothing, use_target = use_target)
    elseif cdr_method == "last1_WLR"
        decay_weights = [[τ == t ? 1.0 : (τ == t - 1 ? 0.5 : 0.0) for τ in 1:t] for t in 1:length(noisy_expvals[1])]
        corr_exp_result = cdr(noisy_expvals, exact_expvals, noisy_expval_target, decay_weights; exact_target_exp_value = use_target ? exact_expval_target : nothing, use_target = use_target)
    else
        error("cdr correction method $(cdr_method) unknown.")
    end

    if use_target
        corr_exp, rel_error_after, rel_error_before = corr_exp_result
    else
        corr_exp = corr_exp_result
        rel_error_after = isa(corr_exp, Vector) ? zeros(length(corr_exp)) : NaN
        rel_error_before = isa(corr_exp, Vector) ? zeros(length(corr_exp)) : NaN
    end

    timetmp2 = time()
    @logmsg SubInfo "cdr method $(cdr_method) done in $(round(timetmp2 - timetmp1; digits = 2)) s"

    log = noise_kind == "naive" ? open("trotter_naive.log", "a") :
          noise_kind == "gate_ising" ? open("trotter_gate_ising.log", "a") :
          noise_kind == "realistic_ising" ? open("trotter_realistic_ising.log", "a") :
          noise_kind == "gate_kickedising" ? open("trotter_gate_kickedising.log", "a") :
          noise_kind == "realistic_kickedising" ? open("trotter_realistic_kickedising.log", "a") :

          error("Noise kind $(noise_kind) unknown (for logger).")

    if record #we only record when there is a corrected exp value, so for cdr_method =="end", if record, we only get one value!
        for i in eachindex(corr_exp)
            str = format("{:>10s} {:>3n} {:>10s} {:>5n} {:>5n} {:>6.2e} {:>10.2e} {:>10.2e}{:>5n} {:>5n}{:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}{:>5n} {:>10.3e} {:>10.2e} {:>10.2e} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.2e}\n",
                cdr_method, i, obs_string, ansatz.nqubits, ansatz.steps, ansatz.time, ansatz.J, ansatz.h,
                non_replaced_gates, num_samples, angle_definition, min_abs_coeff,
                min_abs_coeff_noisy, min_abs_coeff_target, max_weight,
                use_target ? exact_expval_target[i] : NaN,
                noisy_expval_target[i], corr_exp[i], rel_error_before[i], rel_error_after[i],
                rel_error_before[i] / max(rel_error_after[i], 1e-12), timetmp2 - time1)
            write(log, str)
        end
    else
        ratio_rel_error = rel_error_before / max(rel_error_after, 1e-12)
        str = format("{:>10s} {:>3n} {:>2s} {:>5n} {:>5n} {:>6.2e} {:>10.2e} {:>10.2e}{:>5n} {:>5n}{:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}  {:>5n} {:>10.3e} {:>10.2e}{:>10.2e}  {:>10.3e} {:>10.3e} {:>10.3e} {:>10.2e}\n",
            cdr_method, ansatz.steps, obs_string, ansatz.nqubits, ansatz.steps, ansatz.time, ansatz.J, ansatz.h,
            non_replaced_gates, num_samples, angle_definition, min_abs_coeff,
            min_abs_coeff_noisy, min_abs_coeff_target, max_weight,
            use_target ? exact_expval_target[end] : 0.0,
            noisy_expval_target[1], corr_exp, rel_error_before, rel_error_after,
            ratio_rel_error, timetmp2 - time1)
        write(log, str)
    end

    close(log)

    return exact_expval_target, noisy_expval_target, corr_exp, rel_error_before, rel_error_after
end

function full_run_all_methods(ansatz::trotter_ansatz_tfim,
    angle_definition::Float64,
    noise_kind::String;
    min_abs_coeff::Float64=0.0,
    min_abs_coeff_noisy::Float64=0.0,
    max_weight=Inf,
    training_set=nothing,
    observable=nothing,
    num_samples::Int=10,
    depol_strength::Float64=0.01,
    dephase_strength::Float64=0.01,
    depol_strength_double::Float64=0.0033,
    dephase_strength_double::Float64=0.0033,
    min_abs_coeff_target::Float64=0.0,
    noise_levels::Vector{Float64}=[1.0,1.5,2.0],
    lambda::Float64=0.0,
    use_target::Bool=true,
    real_qc_noisy_data=nothing
)
    @logmsg SubInfo "→ Starting full_run_all_methods (noise_kind=$noise_kind, σ=$angle_definition)"

    # determine observable name
    obs_62 = PauliSum(ansatz.nqubits); add!(obs_62, :Z, 62)
    if observable === nothing || observable == obs_interaction(ansatz)
    observable, obs_string = obs_interaction(ansatz), "ZZ"
    elseif observable == obs_magnetization(ansatz)
    obs_string = "Z"
    elseif observable == obs_62
    obs_string = "Z_62"
    else
    obs_string = "Unknown"
    end
    @logmsg SubInfo "→ Observable: $obs_string"

    # generate training set if needed
    if training_set === nothing
    @logmsg SubInfo "→ Generating training set (n=$num_samples)…"
    training_set = training_set_generation_strict_perturbation(ansatz, angle_definition; num_samples=num_samples)
    end
    @logmsg SubInfo "→ Training set size: $(length(training_set))"

    # timestamp start
    time1 = time()

    # compute or inject final‐step target values
    if use_target
    exact_target = trotter_time_evolution(ansatz; observable=observable,
                    noise_kind="noiseless",
                    min_abs_coeff=min_abs_coeff_target,
                    record=false)
    timetmp = time()
    @logmsg SubInfo "→ exact_target done in $(round(timetmp - time1; digits=2)) s"

    noisy_target = trotter_time_evolution(ansatz; observable=observable,
                    noise_kind=noise_kind,
                    min_abs_coeff=min_abs_coeff_target,
                    record=false)
    timetmp = time()
    @logmsg SubInfo "→ noisy_target done in $(round(timetmp - time1; digits=2)) s"
    else
    @logmsg SubInfo "→ Skipping model targets; using real_qc_noisy_data=$(real_qc_noisy_data)"
    exact_target = NaN
    noisy_target = real_qc_noisy_data !== nothing ? real_qc_noisy_data : NaN
    timetmp = time()
    end
    @logmsg SubInfo "→ Targets: exact=$(exact_target), noisy=$(noisy_target)"

    # build training ensembles
    exact_train = training_trotter_time_evolution(ansatz, training_set;
                        observable=observable,
                        noise_kind="noiseless",
                        min_abs_coeff=min_abs_coeff,
                        max_weight=max_weight,
                        record=false)
    timetmp1 = time()
    @logmsg SubInfo "→ exact_train done in $(round(timetmp1 - timetmp; digits=2)) s"

    noisy_train = training_trotter_time_evolution(ansatz, training_set;
                        observable=observable,
                        noise_kind=noise_kind,
                        min_abs_coeff=min_abs_coeff_noisy,
                        max_weight=max_weight,
                        depol_strength=depol_strength,
                        dephase_strength=dephase_strength,
                        depol_strength_double=depol_strength_double,
                        dephase_strength_double=dephase_strength_double,
                        record=false)
    timetmp2 = time()
    @logmsg SubInfo "→ noisy_train done in $(round(timetmp2 - timetmp1; digits=2)) s"

    # --- ZNE ---
    zne_levels = zne_time_evolution(ansatz; observable=observable,
            noise_kind=noise_kind,
            min_abs_coeff=min_abs_coeff,
            max_weight=max_weight,
            noise_levels=noise_levels,
            depol_strength=depol_strength,
            dephase_strength=dephase_strength,
            depol_strength_double=depol_strength_double,
            dephase_strength_double=dephase_strength_double,
            record=false)
    result_zne = zne(zne_levels;
    noise_levels=noise_levels,
    fit_type="linear",
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target)
    timetmp3 = time()
    @logmsg SubInfo "→ ZNE done in $(round(timetmp3 - timetmp2; digits=2)) s"
    if use_target
    zne_corr, zne_err_after, zne_err_before = result_zne
    else
    zne_corr = result_zne; zne_err_before = NaN; zne_err_after = NaN
    end
    open("trotter_ZNE_$(noise_kind).log","a") do log
    str = format(
    "{:>10s} {:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.2f}\n",
    "ZNE", ansatz.steps,
    ansatz.time, ansatz.J, ansatz.h,
     obs_string,
      ansatz.nqubits,
    exact_target, 
    noisy_target,
    zne_err_before, zne_err_after,
    timetmp3 - time1
    )
    write(log, str)
    end

    # --- CDR ---
    result_cdr = cdr(noisy_train, exact_train, noisy_target;
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target)
    timetmp4 = time()
    @logmsg SubInfo "→ CDR done in $(round(timetmp4 - timetmp3; digits=2)) s"
    if use_target
    cdr_corr, cdr_err_after, cdr_err_before = result_cdr
    else
    cdr_corr = result_cdr; cdr_err_before = NaN; cdr_err_after = NaN
    end
    open("trotter_CDR_$(noise_kind).log","a") do log
    str = format(
    "{:>10s} {:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.2f}\n",
    "CDR", ansatz.steps,ansatz.time, ansatz.J, ansatz.h, obs_string, ansatz.nqubits,
    exact_target, noisy_target,
    cdr_err_before, cdr_err_after,
    timetmp4 - time1
    )
    write(log, str)
    end

    # --- vnCDR ---
    noisy_train_multi = vnCDR_training_trotter_time_evolution(ansatz, training_set;
                                    observable=observable,
                                    noise_kind=noise_kind,
                                    min_abs_coeff=min_abs_coeff,
                                    max_weight=max_weight,
                                    noise_levels=noise_levels,
                                    depol_strength=depol_strength,
                                    dephase_strength=dephase_strength,
                                    depol_strength_double=depol_strength_double,
                                    dephase_strength_double=dephase_strength_double,
                                    record=false)
    result_vn = vnCDR(noisy_train_multi, exact_train, zne_levels;
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target, lambda=lambda)
    timetmp5 = time()
    @logmsg SubInfo "→ vnCDR done in $(round(timetmp5 - timetmp4; digits=2)) s"
    if use_target
    vn_corr, vn_err_after, vn_err_before = result_vn
    else
    vn_corr = result_vn; vn_err_before = NaN; vn_err_after = NaN
    end
    open("trotter_vnCDR_$(noise_kind).log","a") do log
    str = format(
    "{:>10s} {:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.2f}\n",
    "vnCDR", ansatz.steps,ansatz.time, ansatz.J, ansatz.h, obs_string, ansatz.nqubits,
    exact_target, noisy_target,
    vn_err_before, vn_err_after,
    timetmp5 - time1
    )
    write(log, str)
    end

    @logmsg SubInfo "→ full_run_all_methods complete."
    return (exact_target, noisy_target,
    zne_corr, cdr_corr, vn_corr,
    zne_err_before, cdr_err_before, vn_err_before,
    zne_err_after,  cdr_err_after,  vn_err_after)
end


function plot_MSE_csv_data(filename::String, xaxis::String)
    data = CSV.read(filename, DataFrame)
    hasproperty(data, Symbol(xaxis)) || error("Column $xaxis not found in data")

    x = getproperty(data, Symbol(xaxis))

    plot(x, data.MSE_loose_CPA, yscale = :log10, marker = :x, label = "MSE loose CPA")
    plot!(x, data.MSE_loose_small, marker = :x, label = "MSE loose small")
    plot!(x, data.MSE_noise, marker = :x, label = "MSE noise")
    plot!(x, data.MSE_strict_CPA, marker = :x, label = "MSE strict CPA")
    plot!(x, data.MSE_strict_small, marker = :x, label = "MSE strict small")
    #plot!(x, data.MSE_brut, marker = :x, label = "MSE brut",
    #     xlabel = xaxis == "sigma_J" ? L"\theta_J" : L"\theta_h", ylabel = "MSE", legend = :bottomright)
    plot!(x, data.MSE_brut, marker = :x, label = "MSE brut",
         xlabel = xaxis == "sigma_J" ? L"\theta_J" : L"\theta_h", ylabel = "MSE", legend = xaxis == "sigma_J" ? :bottomright : :bottomleft)
    savefig(filename[1:end-4]*".png")
end



function run_method(trotter, training_set,sigma_star, noise_kind; min_abs_coeff = 0.0, min_abs_coeff_noisy =0.0, min_abs_coeff_target=0.0, num_samples=10, depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033)
    exact_expval_target, noisy_expval_target, corr_exp, rel_error_before, rel_error_after = full_run(trotter, sigma_star, noise_kind; min_abs_coeff = min_abs_coeff, min_abs_coeff_noisy = min_abs_coeff_noisy, observable = obs_magnetization(trotter), training_set = training_set, depol_strength=depol_strength, dephase_strength=dephase_strength,depol_strength_double = depol_strength_double, dephase_strength_double = dephase_strength_double)  
    MSE_ind =(exact_expval_target - corr_exp)^2
    return MSE_ind
end
