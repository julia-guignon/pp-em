include("zte_all_functions.jl")
global_logger(UnbufferedLogger(stdout, MainInfo)); 

function exp_fit_2(xdata, ydata; abs_threshold=1e-16, fn="")
    mask = xdata .>= abs_threshold
    x = xdata[mask]; y = ydata[mask]

    # Weights: 0.(ooM - 1) = (ooM - 1) / 10
    ooM = floor.(Int, log10.(x))
    w = (ooM .- 1) ./ 10
    sqrtw = sqrt.(w)

    S = 10.0 ^ floor(Int, log10(minimum(x)))
    x_scaled = x ./ S

    model(x, p) = p[1] .- p[2] .* exp.(-p[3] .* x)
    weighted_model(x, p) = sqrtw .* model(x, p)
    y_weighted = sqrtw .* y
    p0 = [y[end], y[end] - y[1], 1.0]

    fit = curve_fit(weighted_model, x_scaled, y_weighted, p0)
    A_fit, B_fit, Cp_fit = fit.param
    C_fit = Cp_fit / S

    x_fit = range(minimum(x), stop=maximum(x), length=100)
    y_fit = A_fit .- B_fit .* exp.(-C_fit .* x_fit)

    @logmsg SubInfo "exp fit params: A=$A_fit, B=$B_fit, C=$C_fit"
    if fn != ""
        scatter(x, y, label="data", marker=:circle, ms=6)
        plot!(x_fit, y_fit, label="fit", 
            guidefontsize=16, tickfontsize=12, legendfontsize=12, titlefontsize = 16, ms = 6)
        if minimum(x) < 1; plot!(xscale=:log10); end
        title = match(r"step=\d+", fn)
        xlabel!("Number of Pauli Strings"); ylabel!("Expectation Value")
        savefig(fn)
    end
    return A_fit
end



function run_zte()
    model = "XX_YY_Z"

    Jx = 1.0
    Jy = 1.0
    h = 3.0

    # time resolution
    tstep = 0.01
    theta = tstep * 1im
    nq = 20
    nl = 30
    time = range(tstep, tstep*nl, nl)


    topology  = bricklayertopology(nq)
    topology_string = "1d"

    # circuit layer (isotropic XY model w. transverse field)
    layer = Gate[]
    append!(layer, PauliRotation(:Z, ii, h*theta) for ii in 1:nq);
    append!(layer, PauliRotation([:X, :X], pair,Jx*theta) for pair in topology);
    append!(layer, PauliRotation([:Y, :Y], pair, Jy*theta) for pair in topology);


    # system parameters
    obs_i = div(nq,2)
    obs_j = div(nq,2)+1
    println("obs_i: ", obs_i, " obs_j: ", obs_j) # change run_or_read for different type of observable


    max_weights = 2:1:7
    trunc_coeff = 2^(-18)
 
    fn = format("{}_Jx={}_Jy={}_h={}_nq={:d}_nl={:d}__topo_{}_theta={:.3e}i_obsi={:d}_obsj={:d}_trunc_coeff={:.3e}", model, Jx,Jy,h,nq, nl, topology_string, tstep, obs_i, obs_j, trunc_coeff)
    
    # init plot
    scatter(ylabel=L"$\langle Z_{%$obs_i} Z_{%$obs_j} \rangle$", xlabel=L"\textrm{imaginary\ time}", title = L"$H = \sum_{i=1} - %$h Z_i  - (%$Jx X_iX_{i+1} + %$Jy Y_iY_{i+1})$")

    # exact expectation values calculations for 1d
    M = jordan_wigner_tfim(nq, Jx, Jy, h)
    zz = zeros(length(time))
    for i in 1:length(time)
        zz[i] = zz_correlation(M, time[i], obs_i, obs_j)
    end

    # get truncated expectation values
    expvals = Vector{Vector{Float64}}()
    terms_nbrs = Vector{Vector{Float64}}()
    @time for j in max_weights
        expectation, terms_nbr = run_or_read(model, layer, nq, nl, tstep, obs_i, obs_j, j, trunc_coeff, fn)#; run=true)
        expectation[expectation .== 0] .= 1e-16
        plot!(time, expectation, label = L"%$j", legend=:topleft, lw=2, markersize=4, marker=:circle, alpha=0.5)
        push!(expvals, expectation)
        push!(terms_nbrs, terms_nbr)
    end


    #save the overview plot
    #plot!(yscale=:log10)
    plot!(time, zz, label = "exact", color="red", legend=:topleft, lw=2, markersize=4, marker=:circle, alpha=0.5)
    #fn = format("{}_Jx={}_Jy={}_h={}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}_maxweight={}", model, Jx,Jy,h,nq, nl, tstep, obs_i, obs_j, max_weight)
    display(scatter!())
    savefig("evo_plot/$(fn)_max_weights_$(max_weights[1])_to_$(max_weights[end]).png")	
    zte_exp(expvals, terms_nbrs, time, trunc_coeff, max_weights; exact_expvals=zz, fn=fn);
end

run_zte()