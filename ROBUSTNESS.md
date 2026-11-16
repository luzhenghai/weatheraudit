# Added robustness checks and conservative detector

These experiments address limitations identified during final review on 2026-11-16. They reuse the eight selected stations and cached observations; they do not create an independent station test set. Results include failures. The original results and original detector remain reproducible.

## What improved and what remains uncertain

Rolling correction retains lower mean error than raw forecasts under longer reporting-lag assumptions. Local civil-day aggregation makes little difference on matched dates. Calendar-week resampling and Holm adjustment narrow the statistical claims. A conservative detector reduces false events in specified synthetic controls, but correctly matches fewer real provenance switches. It is an investigation aid, not an operational alarm.

## Reporting-delay sensitivity

Each row uses the same valid test dates and full daily calendar. The 30-day mean residual is shifted by 1, 8, 14, 30 or 60 calendar days. No future target enters a correction. The last column freezes the mean offset using only pre-test observations. All values are temperature MAE in °C. Every assumed lag retains lower mean error than raw at all eight stations; no significance claim is made for each lag separately. Frozen corrections can be worse, particularly at Quzhou and Shanghai. Actual per-record publication timestamps remain unknown.

| Station | Raw MAE | Lag 1 | Lag 8 | Lag 14 | Lag 30 | Lag 60 | Frozen pre-test offset |
|---|---|---|---|---|---|---|---|
| Quzhou | 1.858 | 1.141 | 1.196 | 1.202 | 1.188 | 1.386 | 2.900 |
| Shanghai | 0.980 | 0.839 | 0.862 | 0.886 | 0.889 | 0.897 | 1.865 |
| Osaka | 1.525 | 0.667 | 0.684 | 0.696 | 0.722 | 0.761 | 0.671 |
| Hotan | 1.175 | 0.855 | 0.922 | 0.911 | 0.979 | 1.054 | 1.495 |
| Chuncheon | 1.578 | 1.068 | 1.116 | 1.139 | 1.246 | 1.373 | 1.192 |
| Gangneung | 1.245 | 1.014 | 1.023 | 1.042 | 1.094 | 1.217 | 1.081 |
| Pohang | 1.700 | 0.905 | 0.963 | 0.986 | 0.996 | 0.995 | 0.967 |
| Busan | 1.896 | 1.030 | 1.053 | 1.071 | 1.095 | 1.143 | 1.093 |

## Local day and station-label sensitivity

Previous Runs hourly day-1 forecasts are reaggregated into maxima, requiring at least 20 valid hours per day. China retains UTC+8; Japan/Korea use UTC+9. Corrections use an eight-day lag. All four definitions for a station are scored on their common target-date intersection, including the ±1-day label perturbations. Thus counts and scores differ from the original day-ahead table. A label shift of +1 moves a station reading's index one day forward; −1 moves it backward.

| Station | Common dates | UTC+8 raw | Local raw | UTC+8 corrected | Local corrected | Station labels −1 day raw | Station labels +1 day raw |
|---|---|---|---|---|---|---|---|
| Quzhou | 225 | 2.060 | 2.060 | 1.507 | 1.507 | 3.083 | 3.400 |
| Shanghai | 231 | 1.118 | 1.118 | 1.030 | 1.030 | 2.437 | 2.462 |
| Osaka | 297 | 1.743 | 1.737 | 1.303 | 1.311 | 2.521 | 2.468 |
| Hotan | 237 | 1.401 | 1.401 | 1.256 | 1.256 | 2.466 | 2.697 |
| Chuncheon | 297 | 1.824 | 1.824 | 1.414 | 1.415 | 2.918 | 2.974 |
| Gangneung | 294 | 1.533 | 1.528 | 1.339 | 1.336 | 2.881 | 2.908 |
| Pohang | 297 | 1.781 | 1.780 | 1.308 | 1.320 | 2.871 | 2.676 |
| Busan | 297 | 2.021 | 2.023 | 1.295 | 1.289 | 2.617 | 2.597 |

The UTC+9 change shifts raw MAE by at most 0.006 °C and corrected MAE by at most 0.012 °C in the displayed rounded results. Whole-day shifts increase raw error at every station. These are sensitivity checks, not proof of the actual NOAA 24-hour measurement window or forecast issue-time harmonization. Boundary days with insufficient hourly data are excluded.

## Calendar-week resampling and multiple comparisons

Paired daily absolute-error gains are grouped by calendar weeks ending Sunday, retaining gaps. Each draw resamples whole week sums and counts; 9,999 draws estimate a percentile interval and an approximate centered, two-sided p-value. Holm adjustment covers a fixed family of 16 comparisons: eight rolling-vs-raw and eight boosting-vs-rolling. Positive gain means the named correction has lower error. This is approximate exploratory inference and assumes week blocks adequately represent dependence; it is not distribution-free.

| Station | Rolling gain vs raw | Holm p | Boosting gain vs rolling | Holm p |
|---|---|---|---|---|
| Quzhou | 0.6618 | 0.0016 | -0.1164 | 0.2640 |
| Shanghai | 0.1185 | 0.1526 | -0.0746 | 0.9544 |
| Osaka | 0.8412 | 0.0016 | 0.0354 | 0.2640 |
| Hotan | 0.2532 | 0.0100 | -0.1283 | 0.0225 |
| Chuncheon | 0.4619 | 0.0016 | 0.0382 | 1.0000 |
| Gangneung | 0.2224 | 0.0016 | -0.0035 | 1.0000 |
| Pohang | 0.7366 | 0.0016 | 0.1232 | 0.0528 |
| Busan | 0.8437 | 0.0016 | 0.0413 | 0.9573 |

At family alpha 0.05, rolling gains retain support at seven stations; Shanghai is inconclusive. No positive boosting gain over rolling retains support. Hotan's negative boosting gain remains supported. The original seven-available-observation intervals are retained as unadjusted descriptive analyses. This correction family does not include every exploratory rain, heat, model or detector comparison.

## Signed and multi-onset real-background injections

The original detector is unchanged. Steps of −2, −0.5, +0.5 and +2 °C are placed into each of three sources at 35%, 50% and 65% of the available records before 2024-11-30. Three sites meet the minimum history length; Osaka is too sparse. The ±45-day match tolerance is unchanged. There are 108 trials; they share background data and are not independent replicates.

| Injected step °C | Trials | Nearby detection | Correct source | New nearby detection | Control already nearby |
|---|---|---|---|---|---|
| -2.0 | 27 | 26 | 18 | 20 | 6 |
| -0.5 | 27 | 9 | 0 | 5 | 6 |
| 0.5 | 27 | 12 | 2 | 7 | 6 |
| 2.0 | 27 | 26 | 19 | 20 | 6 |

A nearby natural control event can inflate apparent recovery. Six trials per amplitude already have a nearby event; the new-detection column excludes them. Correct-source counts are inclusive and must not be read as incremental recovery. Small steps remain especially difficult: zero of 27 negative 0.5 °C steps and two of 27 positive 0.5 °C steps are attributed correctly.

## Conservative detector implementation

The original detector produced events in 16/50 strong-AR null runs in the first simulation batch. The alternative was then fixed once: 28-observation permutation blocks; 999 permutations; only one root CUSUM search in each pair, without recursive splitting; per-pair alpha 0.05/3; and local t-like screening with effective block size 28. Observed and permuted statistics use the same complete-block data, omitting at most 27 trailing observations from candidate search. The 0.5 °C magnitude and t-like threshold 2 remain unchanged. Fewer searches and stricter dependence allowances trade sensitivity for fewer alarms.

Settings were written to `detector_improvement_protocol.json` before the accepted validation batch. New null seeds start at 20301116 and are checked programmatically to be disjoint from the first batch; synthetic step seeds use a separate offset. This is a development response to observed failures, followed by separately seeded simulation checks; it is not preregistration or independent real-world validation. No further settings were selected from validation results.

## Known-null validation and detection tradeoff

Each 730-day simulated sequence has a common seasonal signal and stationary Gaussian AR(1) source errors. Forecast and reanalysis share a correlated noise component. No step or time-varying bias is present in null runs. Both detector versions receive identical simulated data. Any returned event, including an ambiguous event, is false under this generator. Named false events are also counted separately.

| Detector | AR coefficient | Runs | Runs with false event | Runs with named false event | Rate | 95% Wilson interval |
|---|---|---|---|---|---|---|
| original | 0.3 | 50 | 0 | 0 | 0% | 0.0%–7.1% |
| conservative | 0.3 | 50 | 0 | 0 | 0% | 0.0%–7.1% |
| original | 0.8 | 50 | 20 | 13 | 40% | 27.6%–53.8% |
| conservative | 0.8 | 50 | 0 | 0 | 0% | 0.0%–7.1% |

Under strong autocorrelation, the alternative reduces runs with any false event from 20/50 to 0/50. Zero observed events still has a 95% Wilson upper bound of 7.1%; it is not a zero-risk guarantee. These rates apply only to this generator. Longer memory, trends, non-Gaussian errors, gaps and simultaneous changes remain untested here.

Separate strong-AR sequences receive ±2 °C midpoint steps, ten trials per source and sign:

| Detector | Step °C | Trials | Detected | Correct source |
|---|---|---|---|---|
| original | -2.0 | 30 | 30 | 30 |
| original | 2.0 | 30 | 30 | 29 |
| conservative | -2.0 | 30 | 30 | 30 |
| conservative | 2.0 | 30 | 30 | 30 |

The conservative version detects and correctly attributes 60/60 of these selected large synthetic steps. This does not establish sensitivity to small steps or natural changes.

The same four real provenance switches provide a direct cost of conservatism:

| Station | Conservative candidate | Distance days | Verdict | Correct source |
|---|---|---|---|---|
| Quzhou | 2024-12-22 | 10 | ambiguous | False |
| Shanghai | 2024-12-31 | 1 | station changed | True |
| Osaka | none | — | missed | False |
| Hotan | 2025-01-22 | 21 | station changed | True |

Correct real-switch matches fall from the original 3/4 to 2/4. Quzhou becomes ambiguous, Shanghai and Hotan remain correct, and Osaka remains missed. The improved synthetic false-event behavior does not justify replacing the original 3/4 headline with an unqualified claim of overall superiority. Original and conservative modes are both exposed in the notebook.

## Reproduce and remaining work

Run all cells with `weatheraudit.py`, `robustness_checks.py` and `cache.zip` alongside the notebook. Tables 12–19 reproduce the checks; `robustness_results.json` and `detector_improvement_results.json` retain trial-level records. Protocol JSON files are written before each batch. The full release run denies all HTTP requests and uses only the existing cache.

Remaining work requires evidence beyond this local cache: verify NOAA measurement windows and actual release times, test fixed settings on new stations or a future period, investigate simultaneous source changes, and measure usefulness for a real heat-planning decision. No user, health or emissions benefit is claimed.
