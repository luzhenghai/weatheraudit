# WeatherAudit
**Track:** AI + Climate · **Team:** solo entry
## The problem
A local climate-service analyst needs to know when a temperature correction becomes stale. We focus on Quzhou, China, with seven comparison stations across China, Japan and South Korea. This retrospective prototype audits forecasts for local heat planning; actual user impact remains untested.
## What we built
A notebook takes NOAA station readings and archived model forecasts, returns correction errors and heat-alert counts, and generates a historical replay card. A three-source diagnostic suggests which record to investigate after a bias shift.
## What we added
Our own idea: compare learned corrections with simple rules, then use station, forecast and reanalysis differences to investigate unstable calibration, checking against NOAA provenance flags withheld from the detector's input.
## How it works
Gradient boosting learns forecast error from weather, season and delayed past errors. Training precedes the final 306 dates (254–300 observed test days per station). Rolling methods use observations at least eight days old; this delay is assumed. Comparisons include fixed/monthly offsets, rolling bias and calendar-day persistence. Added inference resamples calendar weeks and applies Holm correction across 16 comparisons.
## Results
Daily maximum temperature MAE in °C; each station's full test set:
| Station | Raw | Rolling | Boosting |
|---|---:|---:|---:|
| Quzhou | 1.86 | 1.20 | 1.31 |
| Shanghai | 0.98 | 0.86 | 0.94 |
| Osaka | 1.52 | 0.68 | 0.65 |
| Hotan | 1.18 | 0.92 | 1.05 |
| Chuncheon | 1.58 | 1.12 | 1.08 |
| Gangneung | 1.25 | 1.02 | 1.03 |
| Pohang | 1.70 | 0.96 | 0.84 |
| Busan | 1.90 | 1.05 | 1.01 |
Rolling mean MAE improves at all eight stations; seven gains retain support after Holm correction. No boosting gain over rolling retains support. Yesterday-persistence MAE is 1.94–2.88 °C on matched dates. A separate true day-ahead check reduces Quzhou MAE from 2.06 to 1.48 °C. Rolling mean MAE remains below raw with reporting lags up to 60 days. Local UTC+9 aggregation changes corrected day-ahead MAE by at most 0.012 °C on matched dates.

The identity S−F = (S−E) − (F−E) links station S, forecast F and ERA5 E. Under a single-source-change assumption, two differences step. The original detector matches 3/4 documented provenance switches within ±45 days in a flag-blind retrospective check; development was not independently blinded. NOAA flags establish a processing/source change, not a sensor replacement. Added signed injections expose poor 0.5 °C attribution. A conservative detector reduces false-event runs from 20/50 to 0/50 in separately seeded strong-autocorrelation simulations (zero-event 95% upper bound 7.1%) and attributes 60/60 synthetic ±2 °C steps, but real-switch matches fall to 2/4.

Simulation rates do not establish real-world false-positive control. NOAA observation windows and actual publication times remain unverified. The main archive stitches short-lead forecasts; sources are dependent. Nominal 80% intervals cover 76–81% empirically. Selected sites, gaps and small heat/rain samples limit generalization; Quzhou has only two hot days.
## What we'd do next
Verify observation windows and release times, test fixed settings on new stations, then evaluate an alarm-gated correction and a real heat-planning decision prospectively.
## Sources
[NOAA GHCN-Daily and flags](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt); [Open-Meteo forecast and ERA5 APIs](https://open-meteo.com/) (CC BY 4.0); [scikit-learn](https://scikit-learn.org/); [interval background](https://arxiv.org/abs/2107.07511). AI assistance: [Claude](https://claude.ai/), [WorkBuddy](https://www.codebuddy.cn/work/) and [OpenAI Codex](https://openai.com/codex/) were used extensively for coding, review, drafting and demo production. Full data, model/library credits, limitations and reproduction steps are in the README and executed notebook. Optional TimesFM results are exploratory and excluded from the main comparison.
