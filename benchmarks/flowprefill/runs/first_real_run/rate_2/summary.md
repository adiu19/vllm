# FlowPrefill benchmark — rate_2

## Attainment by policy and tier

| policy | tier | trials | median % | min % | max % |
|---|---|---|---|---|---|
| control | urgent | 5 | 48.1 | 46.6 | 55.7 |
| control | generous | 5 | 99.2 | 98.1 | 99.6 |
| conservative | urgent | 5 | 50.9 | 42.3 | 54.2 |
| conservative | generous | 5 | 99.2 | 98.1 | 99.6 |
| aggressive | urgent | 5 | 49.1 | 46.4 | 56.6 |
| aggressive | generous | 5 | 99.2 | 98.1 | 99.6 |

## Paired differences vs control (urgent)

Per-trial difference: `attainment[policy] - attainment[control]`.

| policy | trials paired | median Δpp | min Δpp | max Δpp |
|---|---|---|---|---|
| conservative | 5 | -0.9 | -5.8 | +3.6 |
| aggressive | 5 | +0.9 | -0.9 | +4.2 |

## Counts

- total rows: 10110
- measure-window rows: 9135
- error rows: 0
