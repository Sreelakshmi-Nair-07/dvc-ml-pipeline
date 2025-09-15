import os
import warnings
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
import yaml
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset(params):
    df = pd.read_csv(params["dataset"]["raw_csv_path"])
    return df

def split_data(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
   split_cfg = params["split"]
   y = df[params["dataset"]["target_col"]]
   X = df.drop(columns=[params["dataset"]["target_col"]])
   stratify = y if split_cfg.get("stratify", True) else None
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_cfg.get("test_size", 0.2),
                                                       random_state=split_cfg.get("random_state", 42),stratify=stratify,)
   return X_train, X_test, y_train, y_test 

def build_preprocessor(X, params):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=params["preprocess"]["numeric_imputer"])),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=params["preprocess"]["categorical_imputer"])),
            ("ohe", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
def main():
    params = load_params()
    os.makedirs("data/processed", exist_ok=True)

    df = load_dataset(params)
    df = df.replace("?", np.nan)

    X_train, X_test, y_train, y_test = split_data(df, params)
    preprocessor = build_preprocessor(X_train, params)
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    try:
        feat_names = preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_t, columns=feat_names)
        X_test_df = pd.DataFrame(X_test_t, columns=feat_names)
    except:
        X_train_df = pd.DataFrame(X_train_t)
        X_test_df = pd.DataFrame(X_test_t)
        
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    target_col = params["dataset"]["target_col"]
    X_train_df[target_col] = y_train_enc
    X_test_df[target_col] = y_test_enc

    # Saving  the combined files
    X_train_df.to_csv(params["paths"]["processed_train"], index=False)
    X_test_df.to_csv(params["paths"]["processed_test"], index=False)
    
    #saving the preprocessor
    if params["preprocess"]["save_preprocessor"]:
        joblib.dump(preprocessor, params["paths"]["preprocessor"])

if __name__ == "__main__":
    main()