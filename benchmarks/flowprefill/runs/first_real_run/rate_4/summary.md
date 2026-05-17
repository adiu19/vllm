# FlowPrefill benchmark — rate_4

## Attainment by policy and tier

| policy | tier | trials | median % | min % | max % |
|---|---|---|---|---|---|
| control | urgent | 5 | 38.0 | 32.8 | 39.1 |
| control | generous | 5 | 94.3 | 92.8 | 95.0 |
| conservative | urgent | 5 | 37.1 | 32.8 | 39.7 |
| conservative | generous | 5 | 94.2 | 92.8 | 94.9 |
| aggressive | urgent | 5 | 38.0 | 32.8 | 40.7 |
| aggressive | generous | 5 | 94.3 | 92.7 | 95.1 |

## Paired differences vs control (urgent)

Per-trial difference: `attainment[policy] - attainment[control]`.

| policy | trials paired | median Δpp | min Δpp | max Δpp |
|---|---|---|---|---|
| conservative | 5 | -0.8 | -1.6 | +0.8 |
| aggressive | 5 | +0.0 | -0.4 | +1.6 |

## Counts

- total rows: 19884
- measure-window rows: 18050
- error rows: 1
