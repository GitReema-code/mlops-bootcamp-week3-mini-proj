import os, json
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow

def load_data():
    proc_train = "data/processed/train.csv"
    proc_test  = "data/processed/test.csv"
    if os.path.exists(proc_train) and os.path.exists(proc_test):
        X_train = pd.read_csv(proc_train).drop(columns=["target"])
        y_train = pd.read_csv(proc_train)["target"]
        X_test  = pd.read_csv(proc_test).drop(columns=["target"])
        y_test  = pd.read_csv(proc_test)["target"]
        return X_train, X_test, y_train, y_test

    ds = load_wine(as_frame=True)
    X, y = ds.data, ds.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main(use_mlflow=True):
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)

    if use_mlflow:
        os.environ.setdefault("MLFLOW_TRACKING_URI", os.path.abspath("./mlruns"))
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 5)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)
            mlflow.sklearn.log_model(model, "model")
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        print({"accuracy": acc, "f1_macro": f1})

    metrics = {"accuracy": float(acc), "f1_macro": float(f1)}
    os.makedirs("artifacts", exist_ok=True)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    use_mlflow = os.environ.get("USE_MLFLOW", "0") == "1"
    main(use_mlflow=use_mlflow)
