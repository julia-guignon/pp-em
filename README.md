# Error Mitigation on Quantum Dynamicsusing Pauli Propagation
This repo can be used to perform error mitigation using the classical simulation library $`\texttt{PauliPropagation.jl}`$ in Julia. The central implemented error mitigation schemes include:
- Zero Noise Extrapolation (ZNE)
- Clifford Data Regression (CDR)
- variable noise CDR (vnCDR)
- Clifford Perturbation Approximation (CPA) and Clifford Perturbation Data Regression- ZNE (CPDR-ZNE)

To gain an overview of the concept of error mitigation for trotterized circuits, refer to the $`\texttt{introduction-example-error-mitigation.ipynb}`$.
In the notebook $`\texttt{advanced-example-error-mitigation.ipynb}`$, we show how to use our code base for the error mitigation techniques metioned above.
We compared our error mitigation results to the ZNE error mitigation of IBM's utility experiment (2023), which can be reproduced with $`\texttt{IBM_utility_exp_4b_data_generation.jl}`$.

 This repo was created as part of a research project during our Master's degrees. 
