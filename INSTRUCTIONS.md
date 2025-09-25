# Week 3 Mini-Project Instructions

## Step 1 – MLOps Overview & Conda
- Create environment from `conda.yaml`
- Run `python src/train.py` (baseline, no MLflow)
- Confirm `metrics.json` is created

## Step 2 – Experiment Tracking with MLflow
- Run training with MLflow enabled:
  - macOS/Linux: `USE_MLFLOW=1 python src/train.py`
  - Windows (PowerShell): `$env:USE_MLFLOW=1; python src/train.py`
- Change hyperparameters in code and run at least 3 experiments
- Launch UI: `mlflow ui --backend-store-uri ./mlruns`
- Compare runs (params, metrics, artifacts)

## Step 3 – Git & GitHub
- `git init && git add . && git commit -m "init"`
- Create a new branch, modify `train.py`, commit
- Push to GitHub and open a PR; include MLflow screenshots + short explanation

## Step 4 – DVC for Data & Pipeline (no params.yaml)
- `dvc init`
- Track dataset: `dvc add data/raw/wine.csv` and commit the resulting `.dvc` file
- Define `dvc.yaml` with two stages:

```
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - src/preprocess.py
      - data/raw/wine.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - data/processed/train.csv
      - data/processed/test.csv
    outs:
      - artifacts/
    metrics:
      - metrics.json
```

- Run: `dvc repro`

## Step 5 – Group Challenge & Review
- Modify model/parameters or add a simple preprocessing step
- Re-run pipeline via `dvc repro`
- Track experiments in MLflow and compare results
- Submit via GitHub PR with MLflow screenshots and a short Markdown report
