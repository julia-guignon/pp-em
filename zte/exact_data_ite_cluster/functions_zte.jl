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

################# Exact calculations ####################


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

function thermalStateExpectation(circuit, nl, nq, operator; max_weight=nq, min_abs_coeff=0, return_fmt="0")
    pstr = PauliString(nq, :I, 1)
    psum = PauliSum(pstr)

    println(circuit)

    expectations = Float64[]
    terms_nbr = Float64[]

    @time for layers in 1:nl
        psum = propagate!(circuit, psum; max_weight=max_weight, min_abs_coeff=min_abs_coeff, normalization=true)
        
        # the extra division is the also rescale expectations without continually dividing trough
        expectation = getcoeff(psum, operator) / getcoeff(psum, pstr)
        expectations = push!(expectations, expectation)
        terms_nbr = push!(terms_nbr, length(psum.terms))
        println(getcoeff(psum, operator), " ", getcoeff(psum, pstr))
    end
    println("Expectations: ", expectations)
    println("Terms number: ", terms_nbr)
    if return_fmt=="0"
        return expectations
    elseif return_fmt=="countterms"
        return expectations, terms_nbr
    else
        error("Return format $return_fmt not supported.")
    end
    
end

############# ZTE fit function #############
function zte_exp(expvals, truncs, times; plotting=true, exact_expvals=nothing, abs_threshold=1e-16, fn="")
    try
        @assert length(expvals)==length(truncs)
    catch e
        error("The length of input arguments don't match (length(expvals)=$(length(expvals)), length(truncs)=$(length(truncs)))")
    end
    corrected_expvals = Vector{Float64}()
    for idx=1:length(expvals[1])
        expvals_t = [row[idx] for row in expvals]
        truncs_t = [row[idx] for row in truncs]

        # data = DataFrame(trunc=truncs, expval=expvals_t)
        push!(corrected_expvals, exp_fit(truncs_t, expvals_t; abs_threshold=abs_threshold, fn="plot_fits/"*fn*"_step=$idx.png"))
    end
    if plotting
        truncated_most_precise = expvals[end]
        try
            @assert length(times)==length(corrected_expvals)==length(truncated_most_precise)
        catch e
            error("The lengths don't match for plotting (length(times)=$(length(times)), length(corrected_expvals)=$(length(corrected_expvals)))")
        end
        corrected_expvals_cleaned = replace_zeros(corrected_expvals)
        truncated_most_precise_cleaned = replace_zeros(truncated_most_precise)
        p = plot(times, corrected_expvals_cleaned, label="corrected", marker=:circle)
        plot!(times, truncated_most_precise_cleaned, label="truncated most precise", marker=:circle)
        # plot!(yscale=:log10)
        xlabel!("Time")
        ylabel!("Observable")
        title!("Evolution over time")
        if exact_expvals!=nothing
            exact_expvals_cleaned = replace_zeros(exact_expvals)
            # plot!(times, exact_expvals_cleaned, label="exact", marker=:circle)
            rel_err_corr = replace_zeros(abs.((corrected_expvals-exact_expvals)./exact_expvals))
            q = plot(times, rel_err_corr, label="corrected", marker=:circle)
            rel_err_most_prec = replace_zeros(abs.((truncated_most_precise-exact_expvals)./exact_expvals))
            plot!(times, rel_err_most_prec, label="truncated most precise", marker=:circle)
            plot!(yscale=:log10)
            xlabel!("Timestep")
            ylabel!("Relative error")
            title!("Evolution over time")
            display(q)
        end
        display(p)
    end
    return corrected_expvals
end

function exp_fit(xdata, ydata; abs_threshold=1e-16, fn="")
    mask = xdata .>= abs_threshold
    xdata = xdata[mask]
    ydata = ydata[mask]
    model(x, p) = p[1] .+ p[2] .* exp.(-p[3] .* x)
    p0 = [ydata[end], ydata[1] - ydata[end], 0.01] 
    @logmsg SubInfo "initial parameters exp fit $p0 "
    # smaller initial rate
    fit = curve_fit(model, xdata, ydata, p0)
    @logmsg SubInfo "exp fit params: $fit.param"
    if fn!=""
        scatter(xdata, ydata, label="data", marker=:circle)
        plot!(xdata, model(xdata, fit.param), label="fit")
        if minimum(xdata)<1
            plot!(xscale=:log10)
        end
        title = match(r"step=\d+", fn)
        title!(title.match)
        xlabel!("truncation level")
        ylabel!("expectation value")
        savefig(fn)
    end

    return model(0.0, fit.param)
end

function replace_zeros(vec, val=1e-16)
    vec[abs.(vec).<=val] .= val
    return vec
end

function run_or_read(model, circuit, nq, nl, tstep, obs_i, obs_j, max_weight, min_abs_coeff; run=false)
    """
    It's in the title. Run the simulation or use existing data if present. 
    """
    if max_weight == Inf
        max_weight_str = "Inf"
    else
        max_weight_str = string(max_weight)
    end
    fn = format("data/{}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}_maxweight={}_minabscoeff={:.4e}_termcount.dat", model, nq, nl, tstep, obs_i, obs_j, max_weight_str, min_abs_coeff)
    expectation = 0
    terms_nbr = 0
    observable = PauliString(nq, [:Z,:Z], [obs_i,obs_j]);
    time = range(tstep, tstep*nl, nl)
    try
        @assert run==false
        open(fn, "r") do log
            data = CSV.read(log, DataFrame)
            time_file = data[:,1]
            expectation = data[:,2]
            terms_nbr = data[:,3]
        end
    catch e
        expectation, terms_nbr = thermalStateExpectation(circuit, nl, nq, observable; min_abs_coeff=min_abs_coeff, max_weight=max_weight, return_fmt="countterms")
        data = DataFrame(time=time, expectation=expectation, terms_nbr=terms_nbr)
        CSV.write(fn, data)
    end
    return expectation, terms_nbr
end
