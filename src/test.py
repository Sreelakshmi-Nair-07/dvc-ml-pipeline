#!/usr/bin/env python3
import json, yaml, joblib, os, csv
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score,  precision_score, recall_score

def main():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # Paths
    results_path = params["paths"]["results"]
    model_path = params["paths"]["model"]
    preprocessor_path = params["paths"]["preprocessor"]
    test_data_path = params["paths"]["processed_test"]
    metrics_path = "metrics.json"   # For DVC

    # Loading  the artifacts
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    test_df = pd.read_csv(test_data_path)

    # splitting the data
    X_test = test_df.drop(columns=[params["dataset"]["target_col"]])
    y_test = test_df[params["dataset"]["target_col"]]

    # Predict
    y_pred = model.predict(X_test)

    # Metrics config
    metrics_cfg = params["test"]["metrics"]
    average = metrics_cfg.get("average", "binary")
    pos_label = metrics_cfg.get("pos_label", 1)
    
    # Random Forest is the only supported model
    model_type = "random_forest"
    hyperparam = {"n_estimators": params["model"]["random_forest"]["params"]["n_estimators"]}
  
    # Compute metrics
    metrics = {
        "n_estimators": hyperparam["n_estimators"],
        "accuracy": accuracy_score(y_test, y_pred),
    }
    
    #saving metrics.json for DVC
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Building flat result row for DVC plots 
    result = {
        "n_estimators": hyperparam["n_estimators"],
        "accuracy": accuracy_score(y_test, y_pred),
    }
    
    # Read existing results if present, flatten any nested historical entries to the flat schema
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            history_raw = json.load(f)
        history: list = []
        for entry in history_raw:
            # Accept already-flat entries
            if isinstance(entry, dict) and all(k in entry for k in ["n_estimators", "accuracy"]):
                # If metrics were nested previously, lift them up
                if "metrics" in entry and isinstance(entry["metrics"], dict):
                    flat = {**{k: v for k, v in entry.items() if k != "metrics"}, **entry["metrics"]}
                    # Prefer top-level hyperparams if present; otherwise keep what's in metrics
                    history.append(flat)
                else:
                    history.append(entry)
            elif isinstance(entry, dict) and "metrics" in entry:
                # Old schema with nested metrics and possibly model_params
                metrics_dict = entry.get("metrics", {})
                params_dict = entry.get("model_params", {})
                flat_entry = {
                    # Bring common hyperparameters up
                    "n_estimators": metrics_dict.get("n_estimators", params_dict.get("n_estimators")),
                    "accuracy": metrics_dict.get("accuracy"),
                }
                # Remove None keys to keep clean records
                flat_entry = {k: v if v is not None else v for k, v in flat_entry.items() if v is not None}
                history.append(flat_entry)
            else:
                # Unknown schema; skip to avoid corrupting plots
                continue
    else:
        history = []

    # Append current RF-only result
    if "n_estimators" in result:
        history.append(result)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    # Filter any pre-existing entries to RF-only with n_estimators present
    history_rf = [e for e in history if isinstance(e, dict) and set(["n_estimators", "accuracy"]) <= set(e.keys())]
    with open(results_path, "w") as f:
        json.dump(history_rf, f, indent=4)

if __name__ == "__main__":
    main()