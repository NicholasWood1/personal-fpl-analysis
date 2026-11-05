# FPL sample exploration

This folder contains a starter notebook for exploring gameweek performance and rank movement across the sampled FPL cohorts.

## Start

Open `fpl_cohort_exploration.ipynb` in VS Code and select the project's `.venv` Python interpreter. The notebook loads the newest matching files from `../data/processed/` and `../data/raw/`.

If Jupyter or plotting libraries are missing from the environment, install them into the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyter ipykernel matplotlib
```

## What the notebook explores

- Sample coverage by rank stratum and gameweek
- Points and overall-rank distributions by gameweek
- Manager-level rank trajectories
- Gameweek-to-gameweek rank changes (negative means a better rank; positive means a worse rank)
- Cohort summaries grouped by the sampled standings stratum

The current pilot contains histories for one manager, so its charts and distributions are only an end-to-end preview. Increase `HISTORY_MANAGER_LIMIT` in the collection script or set it to `None` before interpreting cohort-level results.

The notebook joins the processed history to the raw standings sample by `manager_id` to recover the sampled rank stratum and page metadata.
