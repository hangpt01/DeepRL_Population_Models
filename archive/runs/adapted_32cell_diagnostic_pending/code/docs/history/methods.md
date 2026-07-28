# Implemented methods

## MOPO

Bootstrap action-conditional continuous transition members are fitted to shared filtered state trajectories. The ridge design includes action-specific linear and quadratic log-abundance responses, allowing harvest and stocking effects to vary with state. Particle MPC subtracts return dispersion and ensemble disagreement. The known emission is applied after latent prediction.

## RefPlan

Maintains a posterior over continuous bootstrap members using observation likelihood. Shared action sequences are scored under every member by posterior mean minus return uncertainty.

## BA-MCTS

Root-samples latent state and model member, performs UCB tree search, and aggregates continuous states with finer log-state keys near the safety threshold. Ensemble disagreement supplies pessimism.

## MOOR

Fits one pooled misspecified Ricker `(r,K)` model to filtered states by bounded deterministic search and solves it with continuous fitted value iteration over physical abundance features. It remains structurally Ricker on every stress environment.

## PLUS

Maintains 21 candidate Ricker growth rates and a Rao-Blackwellized state-filter bank. Candidate evidence comes from predictive observation likelihood; continuous candidate fitted-Q values are posterior-weighted.

## Delphic-CQL

Compatible worlds vary latent dimensionality, scale, and projection while sharing the same public data and deterministic optimizer. Their latent variation is multiplied by posterior log-state ambiguity, so the worlds coincide when observation noise is zero. Variation in counterfactual world values is the Delphic penalty in a discrete CQL-style fitted-Q head; behavior-NLL and Bellman-error ranges are reported as compatibility diagnostics. Ordinary data-bootstrap disagreement is deliberately excluded from this axis.

## OGSRL

Fits a continuous dynamics ensemble, a kNN support guardian, and a categorical actor. Model rollouts start from samples reconstructed from cached posterior moments and optimize reward under separate dual variables for first-entry safety cost and OOD visitation. At deployment, actor probabilities, OOD probability, and next-step unsafe occupancy are integrated over belief particles; unsupported/unsafe actions are masked, with a least-violating hard fallback when the feasible set is empty. Linear reward/safety/OOD critics are fitted for diagnostics. The implementation does not transfer the paper's theorem to learned belief features.

These are transparent benchmark-native implementations. External paper repositories are not imported.
