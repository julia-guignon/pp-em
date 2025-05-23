include("functions_zte.jl")
global_logger(UnbufferedLogger(stdout, MainInfo)); 

function run_zte()
    model = "TFIM"

    theta = 0.01im
    θ = theta
    nq = 30
    nl = 20

    topology = bricklayertopology(nq)

    # the circuit
    circuitTFIM = Gate[]
    for i in 1:nl
        append!(circuitTFIM, PauliRotation(:Z, ii, θ) for ii in 1:nq);
        append!(circuitTFIM, PauliRotation([:X, :X], pair, θ) for pair in topology);
    end

    # system parameters
    obs_i = 21
    obs_j = 20
    # observable = PauliString(nq, [:Z,:Z], [obs_i,obs_j])

    max_weight = Inf
    trunc_coeffs = 2.0 .^ (-10:-1:-20)

    tstep = 0.01
    time = range(tstep, tstep*nl, nl)
    if max_weight == Inf
        max_weight_str = "Inf"
    else
        max_weight_str = string(max_weight)
    end

    fn = format("{}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}_maxweight={}", model, nq, nl, tstep, obs_i, obs_j, max_weight_str)    

    # initialize the plot
    scatter(ylabel=L"$\langle Z_{20}Z_{21} \rangle$", xlabel=L"time", title = L"$H = \sum_{i=1} -Z_i - X_iX_{i+1}$")

    # exact expectation values calculations
    M = jordan_wigner_tfim(nq, 1.0, 0.0, 1.0)
    zz = zeros(length(time))
    for i in 1:length(time)
        zz[i] = zz_correlation(M, time[i], obs_i, obs_j)
    end

    # get truncated expectation values
    expvals = Vector{Vector{Float64}}()
    terms_nbrs = Vector{Vector{Float64}}()
    for j in trunc_coeffs
        expectation, terms_nbr = run_or_read("TFIM", circuitTFIM, nq, nl, tstep, obs_i, obs_j, max_weight, j; run=true)
        plot!(time, expectation, label = L"$2^{%$(round(log2(j), digits=2))}$", legend=:topleft, lw=2, markersize=4, marker=:circle, alpha=0.5)
        push!(expvals, expectation)
        push!(terms_nbrs, terms_nbr)
    end

    #save the overview plot
    plot!(yscale=:log10)
    plot!(time, zz, label = "exact", color="red", legend=:topleft, lw=2, markersize=4, marker=:circle, alpha=0.5)
    fn = format("{}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}_maxweight={}", model, nq, nl, tstep, obs_i, obs_j, max_weight_str)
    savefig(fn*".png")

    display(scatter!())
    zte_exp(expvals, terms_nbrs, time; exact_expvals=zz, fn=fn);
end

run_zte()
println("I'm done")