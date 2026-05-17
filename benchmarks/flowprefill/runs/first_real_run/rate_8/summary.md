# FlowPrefill benchmark — rate_8

## Attainment by policy and tier

| policy | tier | trials | median % | min % | max % |
|---|---|---|---|---|---|
| control | urgent | 5 | 2.9 | 1.9 | 4.8 |
| control | generous | 5 | 39.6 | 35.9 | 48.5 |
| conservative | urgent | 5 | 2.8 | 2.1 | 4.3 |
| conservative | generous | 5 | 37.7 | 36.4 | 46.8 |
| aggressive | urgent | 5 | 2.9 | 1.5 | 4.8 |
| aggressive | generous | 5 | 37.4 | 36.0 | 48.0 |

## Paired differences vs control (urgent)

Per-trial difference: `attainment[policy] - attainment[control]`.

| policy | trials paired | median Δpp | min Δpp | max Δpp |
|---|---|---|---|---|
| conservative | 5 | +0.0 | -0.6 | +0.2 |
| aggressive | 5 | -0.2 | -0.4 | +0.0 |

## Counts

- total rows: 39576
- measure-window rows: 35847
- error rows: 3
