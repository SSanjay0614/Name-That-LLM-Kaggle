# V7 Candidate — Independent Multi-Scale Centering

V7 starts from V5 and changes only the multi-scale centring calculation. Both tight- and broad-cluster means are computed from the original embedding matrix `Z`; the two corrections are then applied to the residual. This matches the documented 70% tight / 30% broad formula and removes the order dependence in V5.

All other V5 settings remain unchanged: ensemble weights, stylometric features, typographic priors, cluster thresholds, and grouped-shift mixture.

## Validation status

- Candidate is not yet pushed or scored.
- No competition submission has been made.
- Public-preview smoke tests can verify execution and format only, not accuracy. A scored submission is needed to measure leaderboard performance.
