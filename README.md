# Week 3 Mini Project – MLOps Bootcamp  

This project was completed as part of **Week 3 – Generative AI Bootcamp (SDA x WeCloudData)**.  

## 🚀 Steps  
1. **Environment Setup**  
   - Created Conda env from `conda.yaml`  
   - Ran `src/train.py` baseline → generated `metrics.json`  

2. **Experiment Tracking with MLflow**  
   - Enabled MLflow logging (`$env:USE_MLFLOW=1; python src/train.py`)  
   - Ran multiple experiments, tracked accuracy & f1  

3. **Git & GitHub**  
   - Created branch `experiment-mlflow`  
   - Committed code changes and MLflow screenshots  
   - Opened PR → merged into `main`  

4. **DVC Pipeline**  
   - `dvc init`, tracked dataset with `dvc add data/raw/wine.csv`  
   - Defined stages in `dvc.yaml` (preprocess, train)  
   - Reproduced pipeline with `dvc repro`  

5. **Final Review**  
   - Experiments reproducible with DVC + MLflow  
   - Accuracy = **1.0** on wine dataset  
