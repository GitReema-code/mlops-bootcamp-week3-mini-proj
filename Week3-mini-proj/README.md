# Week 3 Mini-Project: Reproducible ML Workflow with MLflow, Git, and DVC

This starter kit demonstrates environment reproducibility with **Conda**, experiment tracking with **MLflow**, code collaboration with **Git/GitHub**, and pipeline/data reproducibility with **DVC**.

## Project Structure
```
mlops-week3/
├─ conda.yaml
├─ src/
│  ├─ train.py
│  └─ preprocess.py
├─ data/
│  ├─ raw/wine.csv   # provided dataset
│  └─ processed/     # created by preprocess.py
└─ README.md
```

## Quickstart
1) Create the environment
```bash
conda env create -f conda.yaml
conda activate mlops-week3
```

2) Baseline training (no MLflow)
```bash
python src/train.py
```

3) Training with MLflow
```bash
# macOS/Linux
USE_MLFLOW=1 python src/train.py
# Windows (PowerShell)
$env:USE_MLFLOW=1; python src/train.py
mlflow ui --backend-store-uri ./mlruns
```

4) Preprocess CSV (Day 4+)
```bash
python src/preprocess.py
```

5) Wire up DVC (Day 4+)
```bash
dvc init
# you will define dvc.yaml in class, then:
dvc repro
```
