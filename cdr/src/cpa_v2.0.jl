
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
using LsqFit
using StatsModels
using Polynomials

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



function obs_interaction(nq, topology)

    """
    Returns the normalised interaction observable sum_{ij} Z_i Z_j.
    """

    interaction = PauliSum(nq)
    
    for i in 1:length(topology)
        q1 = topology[i][1]
        q2 = topology[i][2]
        add!(interaction, [:Z, :Z], [q1, q2])
    end
    return interaction/length(topology)
end

function obs_magnetization(nq)

    """
    Returns the normalised magnetization observable sum_i Z_i.
    """

    magnetization = PauliSum(nq)
    for i in 1:nq
        add!(magnetization,:Z,i)
    end
    magnetization = magnetization/nq
    return magnetization
end




function kickedisingcircuit(nq, nl; topology=nothing)

    """
    generates a circuit with nq qubits and nl layers where the RX angle is a variable and the RZZ angle is fixed to -π/2
    """

    if isnothing(topology)
        topology = bricklayertopology(nq)
    end
    
    xlayer(circuit) = append!(circuit, (PauliRotation([:X], [qind]) for qind in 1:nq))
    zzlayer(circuit) = append!(circuit, (PauliRotation([:Z, :Z], pair, -π/2) for pair in topology))
    #Question: why does Manuel have π/2 here? It's not the same physically
    circuit = Gate[]

    for _ in 1:nl
        zzlayer(circuit)
        xlayer(circuit)
    end

    return circuit
end

function tiltedisingcircuit(nq, nl; topology=nothing)

    """
    generates a circuit with nq qubits and nl layers represented tilted TFIM
    """

    if isnothing(topology)
        topology = bricklayertopology(nq)
    end
    
    xlayer(circuit) = append!(circuit, (PauliRotation([:X], [qind]) for qind in 1:nq))
    zlayer(circuit) = append!(circuit, (PauliRotation([:Z], [qind]) for qind in 1:nq))
    zzlayer(circuit) = append!(circuit, (PauliRotation([:Z, :Z], pair) for pair in topology))
    circuit = Gate[]

    for _ in 1:nl
        zzlayer(circuit)
        xlayer(circuit)
        zlayer(circuit)
    end

    return circuit
end


#################### Time evolution ####################

function define_thetas(circuit, dt, J=2.0, h=1.0, U=1.0)
    rzz_indices = getparameterindices(circuit, PauliRotation, [:Z, :Z])
    rx_indices = getparameterindices(circuit, PauliRotation, [:X])
    rz_indices = getparameterindices(circuit, PauliRotation, [:Z])

    nparams = countparameters(circuit)

    thetas = zeros(nparams)
    thetas[rzz_indices] .= - J * dt * 2
    thetas[rx_indices] .= h * dt * 2
    thetas[rz_indices] .= U * dt * 2

    return thetas
end 

function trotter_time_evolution(nq, nl, topology, layer; observable = nothing, special_thetas=nothing, noise_kind="none", min_abs_coeff=0.0, max_weight = Inf, noise_level = 1.0, depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033)

    """
    Function that computes the time evolution of the ansatz using the first order Trotter approximation exact time evolution operator.
    The function returns the overlap of the final state with the |0> state.
    """

    if observable===nothing
        obs = obs_interaction(nq, topology)
    else
        obs = deepcopy(observable)
    end

    if special_thetas===nothing
        error("not implemented")
        # thetas = constrain_params(ansatz)
    else
        thetas = special_thetas
    end


    
    # if noise_kind=="naive"
    #     if record
    #         error("Naive noise model doesn't support recording :(")
    #     else
    #         layer = final_noise_layer_circuit(ansatz; depol_strength = noise_level*depol_strength,dephase_strength = noise_level*dephase_strength)
    #     end


    # nparams = countparameters(ansatz.target_circuit)
    nparams = length(thetas)
    @assert nparams==countparameters(layer) "ERROR: nparams = $nparams != parameters required by the layer = $(countparameters(layer))"
    expvals_trotter = Float64[]   
    push!(expvals_trotter, overlapwithzero(obs))
    for i in 1:nl
        # psum = propagate!(layer, obs, thetas[Int(nparams/nl*(i-1)+1):Int(nparams/nl*i)];
        # min_abs_coeff=min_abs_coeff, max_weight = max_weight)
        psum = propagate!(layer, obs, thetas;
        min_abs_coeff=min_abs_coeff, max_weight = max_weight)
        # some functions replicating noise
        if noise_kind=="both"
            full_noise(psum, depol_strength, dephase_strength, noise_level)
        elseif noise_kind=="depolarizing"
            depol_noise(psum, depol_strength, noise_level)
        end
        push!(expvals_trotter, overlapwithzero(psum))
    end
    return expvals_trotter  
end


function full_noise(psum::PauliSum, p::Float64, q::Float64, noise_level::Float64=1.0)
    for (pstr, coeff) in psum
        set!(psum, pstr, coeff*(1-noise_level*p)^countweight(pstr)*(1-q)^countxy(pstr))
    end
end

function depol_noise(psum::PauliSum, p::Float64, noise_level::Float64=1.0)
    for (pstr, coeff) in psum
        set!(psum, pstr, coeff*(1-noise_level*p)^countweight(pstr))
    end
end

function dephas_noise(psum::PauliSum, q::Float64)
    for (pstr, coeff) in psum
        set!(psum, pstr, coeff*(1-q)^countxy(pstr))
    end
end



######### Training circuits related ##########
function training_circuit_generation_loose_perturbation(layer, dt, J, h, angle_definition::Float64=pi/20, U=0.0; sample_function = nothing, num_samples::Int = 10)

    """
    Generates a training set similar to the strict perturbation method, but keeps an angle if it is already in the correct interval. 
    """

    theta_J = -2*J*dt
    theta_h = 2*h*dt
    theta_U = 2*U*dt

    if !(0.0 <= theta_h <= angle_definition) && !(pi/2 - angle_definition <= theta_h <= pi/2)
        change_theta_h = true
    else
        change_theta_h = false
    end
    if !(-angle_definition <= theta_J <= 0.0) && !(-pi/2 <= theta_J <= -pi/2 + angle_definition)
        change_theta_J = true
    else
        change_theta_J = false
    end
    if !(0.0 <= theta_U <= angle_definition) && !(pi/2 - angle_definition <= theta_U <= pi/2)
        change_theta_U = true
    else
        change_theta_U = false
    end
    
    function sample_theta_CPA(angle_definition)
        # tht_h ∈ [0, angle_definition] ∪ [π/2 - angle_definition, π/2]
        tht_h = rand(Bool) ? rand(Uniform(0.0, angle_definition)) : rand(Uniform(π/2 - angle_definition, π/2))
    
        # tht_J ∈ [−angle_definition, 0] ∪ [−π/2, −π/2 + angle_definition]
        tht_J = rand(Bool) ? rand(Uniform(-angle_definition, 0.0)) : rand(Uniform(-π/2, -π/2 + angle_definition))

        # tht_U ∈ [0, angle_definition] ∪ [π/2 - angle_definition, π/2]
        tht_U = rand(Bool) ? rand(Uniform(0.0, angle_definition)) : rand(Uniform(π/2 - angle_definition, π/2))
    
        return tht_h, tht_J, tht_U
    end

    function sample_theta_small(angle_definition)
        # tht_h ∈ [0, angle_definition]
        tht_h =  rand(Uniform(0.0, angle_definition)) 
    
        # tht_J ∈ [−angle_definition, 0] ∪ [−π/2, −π/2 + angle_definition]
        tht_J = rand(Uniform(-angle_definition, 0.0))
        
        # tht_U ∈ [0, angle_definition]
        tht_U =  rand(Uniform(0.0, angle_definition)) 
    
        return tht_h, tht_J, tht_U
    end
    
    if sample_function === nothing
        sample_function = sample_theta_small
    elseif sample_function==="CPA"
        sample_function = sample_theta_CPA
    elseif sample_function==="small"
        sample_function = sample_theta_small
    end


    training_thetas_list = Vector{Vector{Float64}}()
    thetas = define_thetas(layer, dt, J, h, U)
    training_thetas = deepcopy(thetas)
    
    theta_J_indices = getparameterindices(layer, PauliRotation, [:Z,:Z]) 
    theta_h_indices = getparameterindices(layer, PauliRotation, [:X])
    theta_U_indices = getparameterindices(layer, PauliRotation, [:Z])

    for _ in 1:num_samples
        if change_theta_h
            tht_h_perturbed, _ = sample_function(angle_definition)
        else 
            tht_h_perturbed = theta_h
        end
        if change_theta_J
            _, tht_J_perturbed = sample_function(angle_definition)
        else 
            tht_J_perturbed = theta_J
        end
        if change_theta_U
            _, tht_U_perturbed = sample_function(angle_definition)
        else 
            tht_U_perturbed = theta_U
        end
        training_thetas[theta_h_indices] .= tht_h_perturbed
        training_thetas[theta_J_indices] .= tht_J_perturbed
        training_thetas[theta_U_indices] .= tht_U_perturbed
        push!(training_thetas_list, copy(training_thetas))
    
    end

    return training_thetas_list
end


function training_circuit_generation_strict_perturbation(layer, dt, J, h, angle_definition::Float64 = pi/20, U=0.0; sample_function = nothing, num_samples::Int = 10)

    """
    Generates a training set according to the CPA approach (http://arxiv.org/abs/2412.09518). We do not use data augmentation here (in the form of ZNE or PEC, then referred to as CPDR-ZNE or CPDR-PEC respectively), but stick to standard CPA.
    Their bound (Theorem 1) only holds if we replace all gates of the circuit, unlike the method in the original CDR paper (http://arxiv.org/abs/2005.10189).
    """

    function sample_theta_CPA(angle_definition)
        # tht_h ∈ [0, angle_definition] ∪ [π/2 - angle_definition, π/2]
        tht_h = rand(Bool) ? rand(Uniform(0.0, angle_definition)) : rand(Uniform(π/2 - angle_definition, π/2))
    
        # tht_J ∈ [−angle_definition, 0] ∪ [−π/2, −π/2 + angle_definition]
        tht_J = rand(Bool) ? rand(Uniform(-angle_definition, 0.0)) : rand(Uniform(-π/2, -π/2 + angle_definition))
    
        return tht_h, tht_J
    end

    function sample_theta_small(angle_definition)
        # tht_h ∈ [0, angle_definition] ∪ [π/2 - angle_definition, π/2]
        tht_h =  rand(Uniform(0.0, angle_definition)) 
    
        # tht_J ∈ [−angle_definition, 0] ∪ [−π/2, −π/2 + angle_definition]
        tht_J = rand(Uniform(-angle_definition, 0.0))
    
        return tht_h, tht_J
    end
    
    if sample_function === nothing
        sample_function = sample_theta_CPA
    elseif sample_function==="CPA"
        sample_function = sample_theta_CPA
    elseif sample_function==="small"
        sample_function = sample_theta_small
    end


    training_thetas_list = Vector{Vector{Float64}}()
    thetas = define_thetas(layer, dt, J, h)
    training_thetas = deepcopy(thetas)
    
    theta_J_indices = getparameterindices(layer, PauliRotation, [:Z,:Z]) 
    theta_h_indices = getparameterindices(layer, PauliRotation, [:X])
    
    for _ in 1:num_samples
        tht_h_perturbed, tht_J_perturbed = sample_function(angle_definition)
        training_thetas[theta_h_indices] .= tht_h_perturbed
        training_thetas[theta_J_indices] .= tht_J_perturbed
        push!(training_thetas_list, copy(training_thetas))
    
    end

    return training_thetas_list
end

function training_trotter_time_evolution(nq, nl, topology, layer, training_thetas::Vector{Vector{Float64}}; observable = nothing, noise_kind="none", min_abs_coeff=0.0, max_weight = Inf, noise_level = 1.0, depol_strength=0.02, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033)
   
    """
    Function that computes the time evolution of the ansatz using the first order Trotter approximation exact time evolution operator.
    The function returns the overlap of the final state with the |0> state (Heisenberg picture). 
    """

    exact_expvals = Array{Float64,2}(undef, length(training_thetas), nl+1)
    for (i, thetas) in enumerate(training_thetas)
        exact_expvals[i, :] = trotter_time_evolution(nq, nl, topology, layer; observable = observable, special_thetas=thetas, noise_kind=noise_kind, min_abs_coeff=min_abs_coeff, max_weight = max_weight, noise_level = noise_level, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
    end
    return exact_expvals
end


######### ZNE isolated implementation ##########
function zne_time_evolution(nq, nl, topology, layer, dt, J, h, U=0.0; observable = nothing, noise_kind="noiseless", min_abs_coeff=0.0, max_weight = Inf, noise_levels = [1,1.5,2.0], depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033)

    """
    Function that computes the time evolution of the ansatz using the first order Trotter approximation exact time evolution operator at different noise levels.
    """

    expvals = Array{Float64,2}(undef, length(noise_levels), nl+1)
    thetas = define_thetas(layer, dt, J, h, U)
    for (idx,i) in enumerate(noise_levels)
        expval_target = trotter_time_evolution(nq, nl, topology, layer; special_thetas=thetas, observable = observable, noise_kind=noise_kind, noise_level = i,min_abs_coeff=min_abs_coeff,max_weight = max_weight, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
        for j in 1:length(expval_target)
            expvals[idx,:] .= expval_target
        end
    end
    return expvals
end


# 1st method of ZNE
function zne(noisy_exp::Vector{Float64}; noise_levels = [1,1.5,2.0], fit_type = "linear", exact_target_exp_value::Union{Nothing, Float64}=nothing, use_target::Bool=true)

    """
    Function that computes the ZNE correction for a single noisy expectation value.
    """

    training_data = DataFrame(x=noise_levels, y= noisy_exp)
    if fit_type==="linear"
        ols = lm(@formula(y ~ x), training_data)
        cdr_em(x) = coef(ols)[1] + coef(ols)[2] * x
        corrected = cdr_em(0.0)


    elseif fit_type === "richardson"
        P = Polynomial
        fit_ = Polynomials.fit(P, noise_levels, noisy_exp)
        corrected = fit_(0.0)
    
    elseif fit_type === "exponential"
        model(x, p) = p[1] .+ p[2] .* exp.(-p[3] .* x)
        p0 = [training_data.y[end], training_data.y[1] - training_data.y[end], 1.0]
        fit_ = curve_fit(model, training_data.x, training_data.y, p0)
        corrected = model(0.0, fit_.param)
    else @error("Fit type $(fit_type) not supported.")
    
    end

    if use_target && exact_target_exp_value !== nothing
        abs_error_after = abs(exact_target_exp_value - corrected) #/ abs(exact_target_exp_value)
        abs_error_before = abs(exact_target_exp_value - noisy_exp[1]) #/ abs(exact_target_exp_value) #fixed to the first noise_level to be one
        return corrected, abs_error_after, abs_error_before
    else
        return corrected

    end
end

# 2nd method of ZNE
function zne(noisy_exp::Matrix{Float64}; noise_levels = [1,1.5,2.0], fit_type = "linear", exact_target_exp_value::Union{Nothing, Vector{Float64}}=nothing, use_target::Bool=true)

    """
    Function that computes the ZNE correction for a set of noisy expectation values to visualize the time evolution of the ZNE correction.
    """

    nsteps = size(noisy_exp,2) #this is one more than nsteps (includes time 0)
    corrected = Array{Float64, 1}(undef, nsteps) # undef allocates memory (set to 0)
    abs_errors_after = Float64[]
    abs_errors_before = Float64[]
    for i in 1:nsteps 
        result = zne(noisy_exp[:,i]; noise_levels = noise_levels, 
        fit_type = fit_type,
         exact_target_exp_value = use_target ? (exact_target_exp_value === nothing ? nothing : exact_target_exp_value[i]) : nothing,
         use_target = use_target)
         if use_target && exact_target_exp_value !== nothing
            corrected[i], err_after, err_before = result
            push!(abs_errors_after, err_after)
            push!(abs_errors_before, err_before)
         else
            corrected[i] = result
            #corrected[i] = max(corrected[i], 1e-16)
        end

        # ToDo: add polynomial fit (Richardson extrapolation) for comparison
    end

        return use_target ? (corrected, abs_errors_after, abs_errors_before) : corrected

end





###### vnCDR (ZNE and CDR combined) ##########

function vnCDR_training_trotter_time_evolution(nq, nl, topology, layer, training_thetas::Vector{Vector{Float64}}; observable=nothing, noise_kind="none", min_abs_coeff=0.0, max_weight=Inf, noise_levels=[1, 1.5,2.0], depol_strength=0.01, dephase_strength=0.01, depol_strength_double=0.0033, dephase_strength_double=0.0033)
    
    """
    Function that computes the training data for several noise levels.
    If record=true, stores full time evolution; else only final values.
    """

    exact_expvals = Array{Float64,3}(undef, length(noise_levels), length(training_thetas), nl+1) # 3D array: (noise_levels, circuits, nl+1)

    for (idx, i) in enumerate(noise_levels) # use enumerate to get valid index
        noisy_training = training_trotter_time_evolution(nq, nl, topology, layer, training_thetas; observable=observable, noise_kind=noise_kind, min_abs_coeff=min_abs_coeff, max_weight=max_weight, noise_level=i, depol_strength=depol_strength, dephase_strength=dephase_strength, depol_strength_double=depol_strength_double, dephase_strength_double=dephase_strength_double)
        for j in 1:length(training_thetas)
            exact_expvals[idx, j, :] .= noisy_training[j] # full trajectory
        end
    end
    return exact_expvals
end


### 3 methods for CDR

function cdr(
    noisy_exp_values::Array{Float64, 1},
    exact_exp_values::Array{Float64, 1},
    noisy_target_exp_value::Float64;
    exact_target_exp_value::Union{Nothing, Float64}=nothing,
    use_target::Bool=true)
    training_data = DataFrame(x=noisy_exp_values, y=exact_exp_values)

    """
    Function that computes the CDR correction for a single (last) noisy expectation value.
    """
    ols = lm(@formula(y ~ x), training_data)
    cdr_em(x) = coef(ols)[1] + coef(ols)[2] * x

    corrected = cdr_em(noisy_target_exp_value)

    if use_target && exact_target_exp_value !== nothing
        abs_error_after = abs(exact_target_exp_value - corrected) #/ abs(exact_target_exp_value)
        abs_error_before = abs(exact_target_exp_value - noisy_target_exp_value) #/ abs(exact_target_exp_value)
        return corrected, abs_error_after, abs_error_before
    else
        return corrected
    end
end


function cdr(
    noisy_exp_values::Array{Float64, 2},
    exact_exp_values::Array{Float64, 2},
    noisy_target_exp_value::Array{Float64, 1};
    exact_target_exp_value::Union{Nothing, Array{Float64, 1}}=nothing,
    use_target::Bool=true)

    """
    Function that computes the CDR correction for a set of noisy expectation values to visualize the time evolution of the CDR correction.
    """

    nsteps = length(noisy_target_exp_value)
    corrected = Vector{Float64}(undef, nsteps)
    abs_errors_after = Float64[]
    abs_errors_before = Float64[]

    for i in 1:nsteps

        exact_exp_values_last  = exact_exp_values[:, i] 
        noisy_exp_values_last  = noisy_exp_values[:, i]
        result = cdr(
            noisy_exp_values_last,
            exact_exp_values_last,
            noisy_target_exp_value[i];
            exact_target_exp_value = use_target ? exact_target_exp_value === nothing ? nothing : exact_target_exp_value[i] : nothing,
            use_target = use_target
        )
        if use_target && exact_target_exp_value !== nothing
            corrected[i], err_after, err_before = result
            push!(abs_errors_after, err_after)
            push!(abs_errors_before, err_before)
        else
            corrected[i] = result
        end
    end

    return use_target ? (corrected, abs_errors_after, abs_errors_before) : corrected
end

# CDR with weighted linear regression
function cdr(
    noisy_exp_values::Array{Float64, 2},
    exact_exp_values::Array{Float64, 2},
    noisy_target_exp_value::Array{Float64, 1},
    decay_weights::Vector{Vector{Float64}};
    exact_target_exp_value::Union{Nothing, Array{Float64, 1}}=nothing,
    use_target::Bool=true)

    """
    Function that computes the CDR correction for a set of noisy expectation values to visualize the time evolution of the CDR correction.
    Uses weighted linear regression to compute the correction.
    """
    nsteps = size(noisy_exp_values)[2]
    ncircuits = size(noisy_exp_values)[1]
    corrected = Array{Float64, 1}(undef, nsteps)
    abs_errors_after = Array{Float64, 1}(undef, nsteps)
    abs_errors_before = Array{Float64, 1}(undef, nsteps)

    for t in 1:nsteps
        x_all, y_all, w_all = Float64[], Float64[], Float64[]
        for c in 1:ncircuits, τ in 1:t
            push!(x_all, noisy_exp_values[c, τ])
            push!(y_all, exact_exp_values[c, τ])
            push!(w_all, decay_weights[t][τ])
        end
        df = DataFrame(x = x_all, y = y_all, w = w_all)

        ols = lm(@formula(y ~ x), df, wts = df.w)
        cdr_em(x) = coef(ols)[1] + coef(ols)[2] * x

        corrected[t] = cdr_em(noisy_target_exp_value[t])
        if use_target && exact_target_exp_value !== nothing
            err_after = abs(exact_target_exp_value[t] - corrected[t]) #/ abs(exact_target_exp_value[t])
            err_before = abs(exact_target_exp_value[t] - noisy_target_exp_value[t]) #/ abs(exact_target_exp_value[t])
            abs_errors_after[t] = err_after
            abs_errors_before[t] = err_before
        end

    end

    return use_target ? (corrected, abs_errors_after, abs_errors_before) : corrected
end

#### vnCDR optimization function ####

#1st method: vnCDR for final step
function vnCDR(
    noisy_exp_values::Array{Float64,2},        # size (m circuits, n+1 noise levels)
    exact_exp_values::Array{Float64,1},         # size m
    noisy_target_exp_value::Array{Float64, 1};  # size n+1
    exact_target_exp_value::Union{Nothing, Float64}=nothing,
    use_target::Bool=true,
    lambda::Float64=0.0, fit_type = "linear" , fit_intercept = false                    
)

    """
    Function that computes the vnCDR correction for a single noisy expectation value.
    """

    model = lambda===0.0 ? LinearRegressor(fit_intercept = fit_intercept) : RidgeRegressor(lambda=lambda,fit_intercept = fit_intercept)
    
    if use_target && exact_target_exp_value !== nothing
        abs_error_before = abs(exact_target_exp_value - noisy_target_exp_value[1]) 
    end 

    if fit_type === "exponential"
        # make sure we can fit even with values <0 but >-2
        offset = 2.0
        exact_exp_values = log.(exact_exp_values.+offset)
        noisy_exp_values = log.(noisy_exp_values.+offset)
        noisy_target_exp_value = log.(noisy_target_exp_value.+offset)
    end
    
    # Convert input matrix to DataFrame
    X = DataFrame(noisy_exp_values', :auto)
    mach = machine(model, X, exact_exp_values)
    fit!(mach)
    params = fitted_params(mach)
    @logmsg SubInfo "params , $(params)"

    # Manually compute prediction
    coefs = [v for (_, v) in params.coefs]
    @logmsg SubInfo "coefs , $(coefs)"
    @logmsg SubInfo "noisy_target_exp_value , $(noisy_target_exp_value)" 
    pred = coefs'* noisy_target_exp_value

    if fit_intercept
        pred += params.intercept
    end

    if fit_type === "exponential"
        pred = exp(pred).-offset
    end
    
    @logmsg SubInfo "pred , $(pred)"
        
    if use_target && exact_target_exp_value !== nothing
        abs_error_after = abs(exact_target_exp_value - pred) #/ abs(exact_target_exp_value)
        #abs_error_before = abs(exact_target_exp_value - noisy_target_exp_value[end]) #/ abs(exact_target_exp_value)
        return pred, abs_error_after, abs_error_before
    else
        return pred
    end
end


##2nd method: vnCDR for every step

function vnCDR(
    noisy_exp_values::Array{Float64,3},  # (matrix) from via vnCDR_training_trotter_time_evolution      # size (n+1 noise levels, m circuits, t+1 steps)
    exact_exp_values::Array{Float64, 2},  # from  trotter_time_evolution
    noisy_target_exp_value::Array{Float64,2}; # (matrix) from zne_time_evolution
    exact_target_exp_value::Union{Nothing, Array{Float64,1}}=nothing,
    use_target::Bool=true,
    lambda::Float64=0.0, fit_type = "linear", fit_intercept = false                     # regularization strength
)

    """
    Function that computes the vnCDR correction for a set of noisy expectation values to visualize the time evolution of the vnCDR correction.
    """

    nsteps = size(noisy_exp_values, 3) # one more than nsteps (includes time 0)
    corrected = Array{Float64, 1}(undef, nsteps)
    abs_errors_after = Array{Float64, 1}(undef, nsteps) # vector 
    abs_errors_before = Array{Float64, 1}(undef, nsteps) 
    for i in 1:nsteps
        result = vnCDR(
        noisy_exp_values[:, :, i],
        exact_exp_values[:, i],
        noisy_target_exp_value[:,i];
        exact_target_exp_value = use_target ? (exact_target_exp_value === nothing ? nothing : exact_target_exp_value[i]) : nothing,
        use_target = use_target,
        lambda = lambda, fit_type = fit_type, fit_intercept = fit_intercept
        )

        

        if use_target && exact_target_exp_value !== nothing
            corrected[i], err_after, err_before = result
            abs_errors_after[i] = err_after
            abs_errors_before[i] = err_before
        else
            corrected[i] = result
        end
    end

    return use_target ? (corrected, abs_errors_after, abs_errors_before) : corrected
end




function full_run_all_methods(nq, nl, topology, layer, J, h, dt,
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
    real_qc_noisy_data=nothing, record_fit_data = false,fit_type = "linear",
    fit_intercept = false
)

    """
    Function which runs the all error mitigation methods (CDR, ZNE, vnCDR) for the given ansatz.
    The function returns the CDR-corrected expectation value and the absative error before and after the correction (if available) and logs the results.
    """
    U = 0.0
    @logmsg SubInfo "→ Starting full_run_all_methods (noise_kind=$noise_kind, σ=$angle_definition)"
    T = dt*nl
    # determine observable name
    obs_62 = PauliSum(nq); add!(obs_62, :Z, 62)
    if observable === nothing || observable===obs_interaction(nq, topology)
        observable, obs_string = obs_interaction(nq, topology), "ZZ"
    elseif observable===obs_magnetization(nq)
        obs_string = "Z"
    elseif observable===obs_62
        obs_string = "Z_62"
    else
        obs_string = "Unknown"
    end
    @logmsg SubInfo "→ Observable: $obs_string"

    # generate training set if needed
    if training_set === nothing
    @logmsg SubInfo "→ Generating training set (n=$num_samples)…"
    training_set = training_circuit_generation_strict_perturbation(layer, dt, J, h, angle_definition, U; num_samples=num_samples)
    end
    @logmsg SubInfo "→ Training set size: $(length(training_set))"

    # timestamp start
    time1 = time()

    # compute or inject final‐step target values
    if use_target
        exact_target = trotter_time_evolution(nq, nl, topology, layer; observable=observable,
                        noise_kind="none",
                        min_abs_coeff=min_abs_coeff_target)
        timetmp = time()
        @logmsg SubInfo "→ exact_target done in $(round(timetmp - time1; digits=2)) s"

        noisy_target = trotter_time_evolution(nq, nl, topology, layer; observable=observable,
                        noise_kind=noise_kind,
                        min_abs_coeff=min_abs_coeff_target)
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
    exact_train = training_trotter_time_evolution(nq, nl, topology, layer, training_set;
                        observable=observable,
                        noise_kind="none",
                        min_abs_coeff=min_abs_coeff,
                        max_weight=max_weight)
    exact_train = [row[end] for row in exact_train]

    timetmp1 = time()
    @logmsg SubInfo "→ exact_train done in $(round(timetmp1 - timetmp; digits=2)) s"

    noisy_train = training_trotter_time_evolution(nq, nl, topology, layer, training_set;
                        observable=observable,
                        noise_kind=noise_kind,
                        min_abs_coeff=min_abs_coeff_noisy,
                        max_weight=max_weight,
                        depol_strength=depol_strength,
                        dephase_strength=dephase_strength,
                        depol_strength_double=depol_strength_double,
                        dephase_strength_double=dephase_strength_double)

    noisy_train = [row[end] for row in noisy_train]

    timetmp2 = time()
    @logmsg SubInfo "→ noisy_train done in $(round(timetmp2 - timetmp1; digits=2)) s"

    # --- ZNE ---
    zne_levels = zne_time_evolution(nq, nl, topology, layer, dt, J, h, U; observable=observable,
            noise_kind=noise_kind,
            min_abs_coeff=min_abs_coeff,
            max_weight=max_weight,
            noise_levels=noise_levels,
            depol_strength=depol_strength,
            dephase_strength=dephase_strength,
            depol_strength_double=depol_strength_double,
            dephase_strength_double=dephase_strength_double)
    
    
    zne_levels = zne_levels[:, end]

    result_zne = zne(zne_levels;
    noise_levels=noise_levels,
    fit_type=fit_type,
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target)


    #add linear for comparison
    result_zne_lin = zne(zne_levels;
    noise_levels=noise_levels,
    fit_type="linear",
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target)

    if record_fit_data
        # println("in full run all meths h=$h")
        fn = format("ZNE_noise_{}_hval_{:.3e}_nq={:d}_nl={:d}_angledef={:.3e}.dat",noise_kind, h*2*dt, nq, nl,angle_definition)
        mkpath(dirname(fn))
        CSV.write(fn,DataFrame(noise_levels=noise_levels, zne_levels=zne_levels))
    end

    timetmp3 = time()
    @logmsg SubInfo "→ ZNE $(fit_type) done in $(round(timetmp3 - timetmp2; digits=2)) s"
    if use_target
        zne_corr, zne_err_after, zne_err_before = result_zne
    else
        zne_corr = result_zne; zne_err_before = NaN; zne_err_after = NaN
    end

    if use_target
        zne_corr_lin, zne_err_after_lin, zne_err_before_lin = result_zne_lin
    else
        zne_corr_lin = result_zne_lin; zne_err_before_lin = NaN; zne_err_after_lin = NaN
    end
    
    tmp = "./trotter_ZNE_$(fit_type)_$(noise_kind).log"
    mkpath(dirname(tmp))
    open(tmp,"a") do log
        str = format(
        "{:>10s} {:>10.2e}{:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>5s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e}{:>10.3e} {:>10.3e} {:>8.2f}\n",
        "ZNE",angle_definition, nl,
        T, J, h,
        obs_string,
        nq,
        exact_target, 
        noisy_target,zne_corr,
        zne_err_before, zne_err_after,
        timetmp3 - time1
        )
        write(log, str)
    end
    
    tmp = "./trotter_ZNE_linear_$(noise_kind).log"
    mkpath(dirname(tmp))
    open(tmp,"a") do log
        str = format(
        "{:>10s} {:>10.2e}{:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e}{:>10.3e} {:>10.3e} {:>8.2f}\n",
        "ZNE",angle_definition, nl,
        T, J, h,
        obs_string,
        nq,
        exact_target, 
        noisy_target,zne_corr_lin,
        zne_err_before_lin, zne_err_after_lin,
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

    
    tmp = "./trotter_CDR_$(noise_kind).log"
    mkpath(dirname(tmp))
    open(tmp,"a") do log
        str = format(
        "{:>10s} {:>10.2e}{:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.2f}\n",
        "CDR", angle_definition, nl,T, J, h, obs_string, nq,
        exact_target, noisy_target,
        cdr_err_before, cdr_err_after,
        timetmp4 - time1
        )
        write(log, str)
    end

    # --- vnCDR ---
    noisy_train_multi = vnCDR_training_trotter_time_evolution(nq, nl, topology, layer, training_set;
                                    observable=observable,
                                    noise_kind=noise_kind,
                                    min_abs_coeff=min_abs_coeff,
                                    max_weight=max_weight,
                                    noise_levels=noise_levels,
                                    depol_strength=depol_strength,
                                    dephase_strength=dephase_strength,
                                    depol_strength_double=depol_strength_double,
                                    dephase_strength_double=dephase_strength_double)

    noisy_train_multi = noisy_train_multi[:, :, end]

    result_vn = vnCDR(noisy_train_multi, exact_train, zne_levels;
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target, lambda=lambda, fit_intercept=fit_intercept, fit_type = fit_type)
    timetmp5 = time()

    #add hyperplane for comparison
    result_vn_lin = vnCDR(noisy_train_multi, exact_train, zne_levels;
    exact_target_exp_value = use_target ? exact_target : nothing,
    use_target=use_target, lambda=0.0, fit_intercept=false, fit_type = "linear")


    @logmsg SubInfo "→ vnCDR done in $(round(timetmp5 - timetmp4; digits=2)) s"
    if use_target
        vn_corr, vn_err_after, vn_err_before = result_vn
    else
        vn_corr = result_vn; vn_err_before = NaN; vn_err_after = NaN
    end

    if use_target
        vn_corr_lin, vn_err_after_lin, vn_err_before_lin = result_vn_lin
    else
        vn_corr_lin = result_vn_lin; vn_err_before_lin = NaN; vn_err_after_lin = NaN
    end


    tmp = "./trotter_vnCDR_$(fit_type)_$(noise_kind).log"
    mkpath(dirname(tmp))
    open(tmp,"a") do log
        str = format(
        "{:>10s} {:>10.2e}{:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e}  {:>10.3e} {:>10.3e} {:>8.2f}\n",
        "vnCDR",angle_definition, nl,T, J, h, obs_string, nq,
        exact_target, noisy_target,vn_corr,
        vn_err_before, vn_err_after,
        timetmp5 - time1
        )
        write(log, str)
    end


    tmp = "./trotter_vnCDR_linear_$(noise_kind).log"
    mkpath(dirname(tmp))
    open(tmp,"a") do log
        str = format(
        "{:>10s} {:>10.2e}{:>3n}{:>6.2e}{:>10.2e}{:>10.2e} {:>2s} {:>5n} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e}  {:>10.3e} {:>8.2f}\n",
        "vnCDR",angle_definition, nl,T, J, h, obs_string, nq,
        exact_target, noisy_target,vn_corr_lin,
        vn_err_before_lin, vn_err_after_lin,
        timetmp5 - time1
        )
        write(log, str)
    end


    @logmsg SubInfo "→ full_run_all_methods complete."
    return (exact_target, noisy_target,
    zne_corr, zne_corr_lin, cdr_corr, vn_corr, vn_corr_lin,
    zne_err_before, zne_err_before_lin, cdr_err_before, vn_err_before, vn_err_before_lin,
    zne_err_after, zne_err_after_lin,  cdr_err_after, vn_err_after, vn_err_after_lin)
end
