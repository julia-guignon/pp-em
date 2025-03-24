##import cdr.jl
include("cdr.jl")

nqubits = 4
nlayers = 2
num_acc_samples = 10
burn_in_samples = 10
n_sweeps = 0
npairs = 2
x_sigma = 0.1
x_0 = -2.1
max_freq = 30
max_weight = 5
depol_strength = 0.05
dephase_strength = 0.00
verbose = false

main(nqubits, nlayers, num_acc_samples, burn_in_samples, n_sweeps, npairs, x_sigma, x_0, max_freq, max_weight, depol_strength, dephase_strength, verbose)