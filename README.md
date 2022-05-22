# Experiments with "Dragonnet"—Neural Network for Causal Inferences

Masashi Yoshioka

March 16, 2022

## Abstract
Following Shi et al. (2019), I have applied the neural network called "Dragonnet" to see how well it can perform for causal inferences. I have simulated a fixed treatment effect model with some confounders that affect both treatment status and outcome nonlinealy. According to my simulation, neural network methods outperform the other methods such as linear regression and Random Forest regression in terms of both bias and RMSE. This might imply that neural network methods can perform well for causal inferences especially when treatment status and outcome depend on confounders nonlinearly. The results also imply that the "Dragonnet" outperforms the existing neural network method.

## Reference
Shi, C., Blei, D. M. & Veitch, V. (2019, December 10). *Adapting neural networks for the estimation of treatment effects* [Paper presentation]. 33rd Annual Conference on Neural Information Processing Systems (NeurIPS 2019), Vancouver, Canada. https://arxiv.org/abs/1906.02120
