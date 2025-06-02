include("zte_all_functions.jl")
global_logger(UnbufferedLogger(stdout, MainInfo)); 


function run_zte()
    model = "XX_YY_Z"

    Jx = 1.0
    Jy = 1.0
    h = 1.0

    # time resolution
    tstep = 0.01
    theta = tstep * 1im
    nx = 5
    ny = 5
    nq = nx*ny
    nl = 30
    time = range(tstep, tstep*nl, nl)


    topology  = rectangletopology(nx,ny)
    topology_string = "rect"

    # circuit layer (isotropic XY model w. transverse field)
    layer = Gate[]
    append!(layer, PauliRotation(:Z, ii, h*theta) for ii in 1:nq);
    append!(layer, PauliRotation([:X, :X], pair,Jx*theta) for pair in topology);
    append!(layer, PauliRotation([:Y, :Y], pair, Jy*theta) for pair in topology);


    # system parameters
    obs_i = div(nq,2)
    obs_j = div(nq,2)+1
    println("obs_i: ", obs_i, " obs_j: ", obs_j) # change run_or_read for different type of observable


    max_weight = 12
    trunc_coeffs = 2.0 .^ (-10:-1:-18)
 
    fn = format("{}_Jx={}_Jy={}_h={}_nq={:d}_nl={:d}__topo_{}_theta={:.3e}i_obsi={:d}_obsj={:d}_maxweight={}", model, Jx,Jy,h,nq, nl, topology_string, tstep, obs_i, obs_j, max_weight)
    
    # init plot
    scatter(ylabel=L"$\langle Z_{%$obs_i} Z_{%$obs_j} \rangle$", xlabel=L"\textrm{imaginary\ time}", title = L"$H = \sum_{i=1} - %$h Z_i  - (%$Jx X_iX_{i+1} + %$Jy Y_iY_{i+1})$")

    # exact expectation values calculations
    M = jordan_wigner_tfim(nq, Jx, Jy, h)
    zz = zeros(length(time))
    for i in 1:length(time)
        zz[i] = zz_correlation(M, time[i], obs_i, obs_j)
    end

    # get truncated expectation values
    expvals = Vector{Vector{Float64}}()
    terms_nbrs = Vector{Vector{Float64}}()
    @time for j in trunc_coeffs
        expectation, terms_nbr = run_or_read(model, layer, nq, nl, tstep, obs_i, obs_j, max_weight, j, fn)#; run=true)
        expectation[expectation .== 0] .= 1e-16
        plot!(time, expectation, label = L"$2^{%$(round(log2(j), digits=2))}$", legend=:topleft, lw=2, markersize=4, marker=:circle, alpha=0.5)
        push!(expvals, expectation)
        push!(terms_nbrs, terms_nbr)
    end


    #save the overview plot
    #plot!(yscale=:log10)
    plot!(time, zz, label = "exact", color="red", legend=:topleft, lw=2, markersize=4, marker=:circle, alpha=0.5)
    #fn = format("{}_Jx={}_Jy={}_h={}_nq={:d}_nl={:d}_theta={:.3e}_obsi={:d}_obsj={:d}_maxweight={}", model, Jx,Jy,h,nq, nl, tstep, obs_i, obs_j, max_weight)
    display(scatter!())
    savefig("evo_plot/$(fn)_trunc_$(log2(trunc_coeffs[1]))_to_$(log2(trunc_coeffs[end])).png")	
    zte_exp(expvals, terms_nbrs, time, trunc_coeffs; exact_expvals=zz, fn=fn);
end

run_zte()