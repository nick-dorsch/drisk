# monte

Composable tools for quick Monte Carlo modelling, with an emphasis on distribution elicitation and model composition.

Monte provides a compact, package-friendly API for probability distributions, elicitation workflows, and composable Monte Carlo models.

## Current scaffold

```text
src/monte/
  distributions/
    base.py                  # shared distribution interfaces
    types.py                 # shared typing helpers
    univariate/
      base.py                # univariate distribution interface
      continuous/            # continuous domain interfaces + normal, lognormal, logitnormal, beta, StretchedBeta, PERT
  random.py                  # seed/RNG helpers
```
