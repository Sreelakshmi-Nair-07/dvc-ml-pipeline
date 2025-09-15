#!/usr/bin/env python3
import os, yaml, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_model(params):
    model_cfg = params["model"]
    # Random Forest is the only supported model
    return RandomForestClassifier(
        **model_cfg["random_forest"]["params"], random_state=42
    )
def main():
    params = load_params()
    train_df = pd.read_csv(params["paths"]["processed_train"])
    target = params["dataset"]["target_col"]

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    model = get_model(params)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(params["paths"]["model"]), exist_ok=True)
    joblib.dump(model, params["paths"]["model"])

if __name__ == "__main__":
    main()
