# Reproduction and release checks

## Main experiment

1. Keep `WeatherAudit.ipynb`, `weatheraudit.py`, `robustness_checks.py` and `cache.zip` together. Use Python 3.13 for the tested environment; Colab already supplies a notebook environment.
2. Install the main dependencies with `python -m pip install -r requirements.txt`. To match final-review fitted values exactly, use `requirements-lock.txt` in a Python 3.13 environment. An existing notebook kernel must use that environment.
3. Run all notebook cells, or run `python weatheraudit.py` from this directory. On first use, the code extracts `cache.zip` into `cache/`. Do not place unrelated cached responses in that directory when checking reproducibility.
4. The notebook contains all saved tables and 10 embedded figures and can be read without running it. `results_v4.json` holds the original results. `robustness_results.json` and `detector_improvement_results.json` hold the additional checks; `ROBUSTNESS.md` explains protocols and outcomes.

Final review on 2026-11-16 executed **24 cells, 124 outputs, 10 embedded figures, zero errors**, in **89.08 seconds**. It copied only the notebook, both Python files and cache archive into a new directory and denied every HTTP request at the requests session layer. The complete run succeeded without downloading anything. This proves the shipped cache is sufficient for the main and added experiments in the tested environment; it does not validate external site availability or model accuracy beyond the reported experiment.

`cache.zip` contains **436** files. The archive CRC check and each SHA-256 hash in `cache_manifest.json` were verified. `python verify_release.py` checks archive integrity, notebook execution status, source synchronization, the principal quoted results and one-page write-up table. It requires only the Python standard library and does not execute the model.

## Scope and qualifications

The final temperature errors and provenance-check values match the pre-review results. A bug in the rain-history feature was corrected: absent precipitation is preserved as missing instead of counted as a dry day. Rain scores therefore changed and the write-up uses the rerun values.

Additional checks cover reporting lags through 60 days, local UTC+9 days, calendar-week resampling with Holm adjustment, signed multi-onset injections, known-null controls and a conservative detector. Original and conservative detector results are kept separate. The latter reduces strong-AR simulated false-event runs from 20/50 to 0/50 on new seeds but real provenance matches fall from 3/4 to 2/4. Zero events has a 95% Wilson upper bound of 7.1%; this is not a real-world error guarantee.

The persistence comparison uses exact calendar lags and complete paired dates, reporting the sample count for each reference. Yesterday persistence assumes prompt access; eight-day persistence observes the project's assumed archive lag. Main forecast rows use a stitched short-lead archive; day-ahead results are separately computed from Previous Runs. Forecast issue time, daily observation windows and record release times are not fully harmonized.

The TimesFM exploration is separate and requires extra dependencies and weights. Its old values are explicitly marked legacy; the repaired model run was not executed. See `TIMESFM_NOTE.md`. It is not needed to reproduce the main experiment.

The one-page `WeatherAudit_writeup.pdf` is exported from the accompanying Word source. It has one page, working source hyperlinks, no personal name or school metadata, and was rendered and visually inspected. The README supplies supporting detail and full credits.
