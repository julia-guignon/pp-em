using PauliPropagation
using Plots
using Logging
using Statistics
using GLM
using DataFrames
using StatsModels
using LsqFit
using Distributions
using LaTeXStrings
using ProgressMeter
using LinearAlgebra
using Format
using DataFrames
using CSV

################# Logging Setup ####################
struct UnbufferedLogger <: Logging.AbstractLogger
    stream::IO
    level::Logging.LogLevel
end

const ImportantInfo = Base.CoreLogging.LogLevel(300)
const MainInfo = Base.CoreLogging.LogLevel(200)
const SubInfo = Base.CoreLogging.LogLevel(100)
const SubsubInfo = Base.CoreLogging.LogLevel(50)

const LOG_LEVEL_NAMES = Dict(
    Logging.Debug => "Debug",
    Logging.Info => "Info",
    Logging.Warn => "Warn",
    Logging.Error => "Error",
    SubInfo => "SubsubInfo",
    SubInfo => "SubInfo",
    MainInfo => "MainInfo",
    ImportantInfo => "ImportantInfo"
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
struct imag_trotter_ansatz_tfim
    target_circuit::Vector{Gate}
    target_circuit_layer::Vector{Gate}
    topology::Vector{Tuple{Int64, Int64}}
    nqubits::Integer
    steps::Integer #layers
    time::ComplexF64
    J::Float64
    h::Float64
    sigma_J::ComplexF64
    sigma_h::ComplexF64
    sigma_J_indices::Vector{Int64}
    sigma_h_indices::Vector{Int64}
    sigma_J_indices_layer::Vector{Int64}
    sigma_h_indices_layer::Vector{Int64}
end

struct trotter_ansatz_tfim
    target_circuit::Vector{Gate}
    target_circuit_layer::Vector{Gate}
    topology::Vector{Tuple{Int64, Int64}}
    nqubits::Integer
    steps::Integer #layers
    time::ComplexF64
    J::Float64
    h::Float64
    sigma_J::Float64
    sigma_h::Float64
    sigma_J_indices::Vector{Int64}
    sigma_h_indices::Vector{Int64}
    sigma_J_indices_layer::Vector{Int64}
    sigma_h_indices_layer::Vector{Int64}
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
    return wrapcoefficients(interaction/length(ansatz.topology), PauliFreqTracker)
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
    return wrapcoefficients(magnetization, PauliFreqTracker)
end


function trotter_time_evolution(
    ansatz;
    observable = nothing,
    special_thetas=nothing,
    noise_kind="noiseless",
    record=false,
    min_abs_coeff=0.0,
    max_weight=Inf,
    max_freq=Inf,
    max_sins=Inf,
    depol_strength=0.01,
    dephase_strength=0.01,
    depol_strength_double=0.0033,
    dephase_strength_double=0.0033,
    customtruncfunc=nothing,
    return_fmt="0") 

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
            circuit = final_noise_layer_circuit(ansatz; depol_strength, dephase_strength)
        end
    elseif noise_kind=="gate"
        circuit = gate_noise_circuit(ansatz; depol_strength, dephase_strength, layer=record)
    elseif noise_kind=="noiseless"
        if record
            circuit = ansatz.target_circuit_layer
        else
            circuit = ansatz.target_circuit
        end
    elseif noise_kind=="realistic"
        circuit = realistic_gate_noise_circuit(ansatz; depol_strength_single = depol_strength, dephase_strength_single = dephase_strength, depol_strength_double = depol_strength_double, dephase_strength_double = dephase_strength_double, layer=record)
        
    else
        error("Noise kind $(noise_kind) unknown.")
    end

    if record
        nparams = countparameters(ansatz.target_circuit)
        expvals_trotter = Float64[]   
        terms_nbr = Float64[]   
        push!(expvals_trotter, overlapwithzero(obs))
        for i in 1:ansatz.steps
            psum = propagate!(circuit, obs, thetas[Int(nparams/ansatz.steps*(i-1)+1):Int(nparams/ansatz.steps*i)];min_abs_coeff=min_abs_coeff)
            push!(expvals_trotter, overlapwithzero(psum))
            push!(terms_nbr, length(psum.terms))
        end
        if return_fmt=="0"
            return expvals_trotter
        elseif return_fmt=="countterms"
            return expvals_trotter, terms_nbr
        else
            error("Return format $return_fmt not supported (record mode used, doesn't allow psum).")
        end
    else 
        psum = propagate!(circuit, obs, thetas; min_abs_coeff=min_abs_coeff, max_weight=max_weight, max_freq=max_freq, max_sins=max_sins, customtruncfunc=customtruncfunc)
        if return_fmt=="0"
            return overlapwithzero(psum)
        elseif return_fmt=="psum"
            return psum
        elseif return_fmt=="countterms"
            return overlapwithzero(psum), length(psum.terms)
        else
            error("Return format $return_fmt not supported.")
        end
    end
end

#### exact solution for ITE on (an)isotropic XY model with transverse field ####
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


### expectation value calc for exact 
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


function getnewimaginarypaulistring(gate::MaskedPauliRotation, pstr::PauliStringType)
    new_pstr, sign = pauliprod(gate.generator_mask, pstr, gate.qinds)
    return new_pstr, -1im * sign
end


function PauliPropagation.applytoall!(gate::PauliRotation, theta::ComplexF64, psum, aux_psum; kwargs...)
    # NOTE: This is for imaginary time evolution!
    if real(theta) > 0.0
        throw(ArgumentError("Parameter `theta` needs to be fully imaginary. Got theta=$theta"))
    end

    
    # turn the PauliRotation gate into a MaskedPauliRotation gate
    # this allows for faster operations
    gate = PauliPropagation._tomaskedpaulirotation(gate, paulitype(psum))

    # pre-compute the sinh and cosh values because they are used for every Pauli string that does not commute with the gate
    cosh_val = cos(theta)
    sinh_val = sin(theta)
    # loop over all Pauli strings and their coefficients in the Pauli sum
    for (pstr, coeff) in psum

        if !commutes(gate, pstr)
            # if the gate does not commute with the pauli string, do nothing
            continue
        end

        # else we know the gate will split the Pauli string into two
        coeff1 = real(coeff * cosh_val)
        new_pstr, sign = getnewimaginarypaulistring(gate, pstr)
        coeff2 = real(coeff * sinh_val * sign)

        # set the coefficient of the original Pauli string
        set!(psum, pstr, coeff1)

        # set the coefficient of the new Pauli string in the aux_psum
        # we can set the coefficient because PauliRotations create non-overlapping new Pauli strings
        set!(aux_psum, new_pstr, coeff2)
    end

    return
end


function PauliPropagation.applymergetruncate!(gate, psum, aux_psum, thetas, param_idx; max_weight=Inf, min_abs_coeff=1e-10, max_freq=Inf, max_sins=Inf, customtruncfunc=nothing, normalization=false, kwargs...)

    # Pick out the next theta if gate is a ParametrizedGate.
    # Else set the paramter to nothing for clarity that theta is not used.
    if gate isa ParametrizedGate
        theta = thetas[param_idx]
        # If the gate is parametrized, decrement theta index by one.
        param_idx -= 1
    else
        theta = nothing
    end
    # Apply the gate to all Pauli strings in psum, potentially writing into auxillary aux_psum in the process.
    # The pauli sums will be changed in-place
    applytoall!(gate, theta, psum, aux_psum; kwargs...)
    
    # Any contents of psum and aux_psum are merged into the larger of the two, which is returned as psum.
    # The other is emptied and returned as aux_psum.
    psum, aux_psum = mergeandempty!(psum, aux_psum)

    nq = psum.nqubits    
    if normalization
        min_abs_coeff = min_abs_coeff * getcoeff(psum, :I, 1)
    end
    
    # Check truncation conditions on all Pauli strings in psum and remove them if they are truncated.
    PauliPropagation.checktruncationonall!(psum; max_weight, min_abs_coeff, max_freq, max_sins, customtruncfunc)

    return psum, aux_psum, param_idx
end

function thermalStateComparison(H, circuit, beta, theta, nq; max_weight=nq, min_abs_coeff=1e-10)
    analyticResult = imaginaryTimeEvolution(H, -beta)
    pstr = PauliString(nq, :I, 1)
    totalAngle = 0
    psum = PauliSum(pstr)
    while totalAngle < beta
        # psum = PauliSum(pstr)
        psum = propagate!(circuit, psum; max_weight, min_abs_coeff, normalization=true)
        totalAngle += imag(theta)
    end
    pstr = topaulistrings(psum)
    distance = computeTwoNorm(pstr, analyticResult, nq)
    return distance
end

function thermalStateExpectation(circuit, nl, nq, operator; max_weight=nq, min_abs_coeff=0, return_fmt="0")
    pstr = PauliString(nq, :I, 1)
    psum = PauliSum(pstr)

    expectations = Float64[]
    terms_nbr = Float64[]

    @showprogress for layers in 1:nl
        psum = propagate!(circuit, psum; max_weight, min_abs_coeff, normalization=true)
        
        # the extra division is the also rescale expectations without continually dividing trough
        expectation = getcoeff(psum, operator) / getcoeff(psum, pstr)
        expectations = push!(expectations, expectation)
        terms_nbr = push!(terms_nbr, length(psum.terms))
    end
    if return_fmt=="0"
        return expectations
    elseif return_fmt=="countterms"
        return expectations, terms_nbr
    else
        error("Return format $return_fmt not supported.")
    end
    
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

function imaginaryTimeEvolution(h::Matrix, beta::Float64)
    operator = exp(-beta*h)
    return operator/tr(exp(-beta*h))
end

function imaginaryTimeEvolutionState(h::Matrix, beta::Float64, state::Vector)
    operator = exp(-beta*h)*state
    return operator/LinearAlgebra.norm(operator)
end

function comparison(P::Matrix, Q::Matrix, beta::Float64)
    num = exp(-beta/2*P)*Q*exp(-beta/2*P)
    C = commutator2(P, Q)
    if (LinearAlgebra.norm(C) < 1e-3)
        analytic = cosh(beta)*Q-sinh(beta)*P*Q
    else
        analytic = Q
    end 
    println("Numerical: ", num)
    println("Analytic: ", analytic)
    println(LinearAlgebra.norm(num-analytic))
end

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

function generatePauliBasis(nqubits::Int)
    basis = Vector{String}(undef, 4^nqubits)
    for i in 1:4^nqubits
        basis[i] = join(rand(['I', 'X', 'Y', 'Z'], nqubits))
    end
    return basis
end

function computeTwoNorm(pstr::Vector, rho::Matrix, nq::Int)
    A = 0
    B = LinearAlgebra.norm(rho)^2
    mix = 0
    for i in 1:size(pstr)[1]
        A += abs(pstr[i].coeff)^2
        string = (inttostring(pstr[i].term,nq))
        mix += pstr[i].coeff*tr(rho*interpretPauli(string))
    end
    A *= 2^nq
    return A+B-2*mix
end

function expectationValue(rho::Matrix, pauli::String)
    mat = interpretPauli(pauli)
    return tr(rho*mat)
end

function run_or_read(model, layer, nq, nl, tstep, obs_i, obs_j, max_weight, min_abs_coeff, fn; run=false)
    """
     Runs the simulation or uses existing data if present. 
    """
    min_abs_coeff_exponent = log2(min_abs_coeff)
    fn = format("data/{}_minabscoeff=2^{}_max_weight_{}_termcount.dat", fn, min_abs_coeff_exponent, max_weight)
    expectation = 0
    terms_nbr = 0
    observable = PauliString(nq, [:Z,:Z], [obs_i,obs_j]);
    time = range(tstep, tstep*nl, nl)
    try
        @assert run==false
        open(fn, "r") do log
            data = CSV.read(log, DataFrame)
            #time_file = data[:,1]
            expectation = data[:,2]
            terms_nbr = data[:,3]
        end
    catch e
        expectation, terms_nbr = thermalStateExpectation(layer, nl, nq, observable; min_abs_coeff=min_abs_coeff, max_weight=max_weight, return_fmt="countterms")
        data = DataFrame(time=time, expectation=expectation, terms_nbr=terms_nbr)
        CSV.write(fn, data)
    end
    return expectation, terms_nbr
end

function zte_exp(expvals, terms_nbr, times, trunc_coeffs, max_weights; plotting=true, exact_expvals=nothing, abs_threshold=1e-16, fn="")
        try
        @assert length(expvals)==length(terms_nbr)
    catch e
        error("The length of input arguments don't match (length(expvals)=$(length(expvals)), length(terms_nbr)=$(length(terms_nbr)))")
    end
    corrected_expvals = Vector{Float64}()
    for idx=1:length(expvals[1])
        expvals_t = [row[idx] for row in expvals]
        terms_nbr_t = [row[idx] for row in terms_nbr]
        push!(corrected_expvals, exp_fit_2(terms_nbr_t, expvals_t; abs_threshold=abs_threshold, fn="plot_fits/"*fn*"_step=$idx.png"))
    end

    @logmsg SubInfo "corrected_expvals: $corrected_expvals"

    if plotting
        truncated_most_precise = expvals[end]
        try
            @assert length(times)==length(corrected_expvals)==length(truncated_most_precise)
        catch e
            error("The lengths don't match for plotting (length(times)=$(length(times)), length(corrected_expvals)=$(length(corrected_expvals)))")
        end
        corrected_expvals_cleaned = replace_zeros(corrected_expvals)
        truncated_most_precise_cleaned = replace_zeros(truncated_most_precise)
        p = plot(times, corrected_expvals_cleaned, label="Corrected", marker=:circle, ms = 6)
        plot!(times, truncated_most_precise_cleaned, label="Truncated most precise", marker=:diamond, guidefontsize=16, tickfontsize=12, legendfontsize=12,titlefontsize = 12, ms = 6, dpi= 300, legend=:topleft, alpha = 0.5)
        # plot!(yscale=:log10)
        if exact_expvals!=nothing 
            plot!(times, exact_expvals, label="Exact", marker=:circle, tickfontsize = 12, ms=6)
        end
        # plot!(yscale=:log10)
    
        xlabel!("Imag. Time")
        ylabel!("Observable")
        
        if exact_expvals!=nothing
            exact_expvals_cleaned = replace_zeros(exact_expvals)
            # plot!(times, exact_expvals_cleaned, label="exact", marker=:circle)
            rel_err_corr = replace_zeros(abs.((corrected_expvals-exact_expvals)./exact_expvals))
            q = plot(times, rel_err_corr, label="Corrected", marker=:circle, tickfontsize = 12, ms=6)
            rel_err_most_prec = replace_zeros(abs.((truncated_most_precise-exact_expvals)./exact_expvals))
            plot!(times, rel_err_most_prec, label="Truncated most precise", marker=:circle, guidefontsize=16, tickfontsize=12, legendfontsize=12,titlefontsize = 12, ms = 6, dpi= 300, legend=:bottomright)

            plot!(yscale=:log10)
            xlabel!("Time")
            ylabel!("Relative error")
            title!("Evolution over time (trunc coeffs %$(log2(trunc_coeffs[1])) to %$(log2(trunc_coeffs[end]))")
            display(q)
        end
        savefig(p, "evo_plot/"*fn*"_$(log2(trunc_coeffs[1]))_to_$(log2(trunc_coeffs[end]))_max_weights_$(max_weights[1])_to_$(max_weights[end])_correction.png")
        display(p)
    end
    return corrected_expvals
end

function exp_fit_2(xdata, ydata; abs_threshold=1e-16, fn="")
    
    # 0) mask small x as before
    mask = xdata .>= abs_threshold
    x = xdata[mask]; y = ydata[mask]

    # 1) determine scale S from smallest x’s order of magnitude
    S = 10.0 ^ floor(Int, log10(minimum(x)))
    x_scaled = x ./ S

    # 2) define model in scaled units: y = A − B exp(−C' x_scaled)
    model(x, p) = p[1] .- p[2] .* exp.(-p[3] .* x)

    # 3) initial guesses A₀≈y[end], B₀≈A₀−y[1], C'₀≈1
    p0 = [y[end], y[end] - y[1], 1.0]

    # 4) perform fit
    fit = curve_fit(model, x_scaled, y, p0)
    A_fit, B_fit, Cp_fit = fit.param

    # recover physical decay constant C = C'/S
    C_fit = Cp_fit / S

    # 5) build fit curve back on original scale
    x_fit = range(minimum(x), stop=maximum(x), length=100)
    y_fit = A_fit .- B_fit .* exp.(-C_fit .* x_fit)

    @logmsg SubInfo "exp fit params: A=$A_fit, B=$B_fit, C=$C_fit"
    if fn != ""
    scatter(x, y, label="data", marker=:circle, ms=6)
    array = range(minimum(x), maximum(x), 50)
    plot!(x_fit, y_fit, label="fit", 
        guidefontsize=16, tickfontsize=12, legendfontsize=12, titlefontsize = 16, ms = 6)
    if minimum(x) < 1; plot!(xscale=:log10); end
    title = match(r"step=\d+", fn)
    xlabel!("Number of Pauli Strings"); ylabel!("Expectation Value")
    savefig(fn)
    end
    return A_fit
end

function replace_zeros(vec, val=1e-16)
    vec[abs.(vec).<=val] .= val
    return vec
end