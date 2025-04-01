include("cpa.jl")

### Full Run Test ###
global_logger(UnbufferedLogger(stdout, SubInfo))

nq = 20 #20
steps = 7 #8
sigma_star = pi/20
T = 0.5
J = 2.0 #J > 0 in ferromagnetic phase, J < 0 in antiferromagnetic phase
h = 1.0 #abs(h) < abs(J) in ordered phase

trotter = trotter_setup(nq, steps, T, J, h);
noise_kind = "realistic" #takes the default realistic values
min_abs_coeff = 1e-5;
min_abs_coeff_noisy = min_abs_coeff;
training_set = training_set_generation_loose_perturbation(trotter, sigma_star; sample_function = "small", num_samples=10);
full_run(trotter, sigma_star, noise_kind, min_abs_coeff, min_abs_coeff_noisy; observable = obs_magnetization(trotter))

