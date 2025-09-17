import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/raw/wine.csv")
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

os.makedirs("data/processed", exist_ok=True)
pd.concat([X_train, y_train.rename("target")], axis=1).to_csv("data/processed/train.csv", index=False)
pd.concat([X_test, y_test.rename("target")], axis=1).to_csv("data/processed/test.csv", index=False)
print("Processed data created.")
