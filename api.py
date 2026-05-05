"""
Fraud Detection REST API
Run: uvicorn api:app --reload

Endpoints
---------
GET  /health                    health check
GET  /models                    list all available model names by category
POST /train                     upload CSV + params → train model, return metrics + model_id
POST /predict/{model_id}        JSON feature rows → fraud predictions
POST /synthetic                 upload CSV + params → augmented CSV download
"""

import io
import uuid
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    IsolationForest, RandomForestClassifier, StackingClassifier,
    VotingClassifier,
)
from sklearn.covariance import EllipticEnvelope
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LogisticRegression, PassiveAggressiveClassifier, Perceptron,
    RidgeClassifier, SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler,
)
from sklearn.svm import LinearSVC, NuSVC, OneClassSVM, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.utils.multiclass import type_of_target

HAS_XGB = HAS_LGBM = HAS_CATBOOST = HAS_IMBLEARN = False
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    pass
try:
    from lightgbm import LGBMClassifier; HAS_LGBM = True
except ImportError:
    pass
try:
    from catboost import CatBoostClassifier; HAS_CATBOOST = True
except ImportError:
    pass
try:
    from imblearn.ensemble import BalancedRandomForestClassifier; HAS_IMBLEARN = True
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────

ANOMALY_MODELS  = {"Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope"}
MINMAX_MODELS   = {"Bernoulli Naive Bayes", "Complement Naive Bayes"}
ESTIMATOR_MODELS = {
    "Random Forest", "Extra Trees", "Gradient Boosting", "Histogram Gradient Boosting",
    "AdaBoost", "Bagging", "XGBoost", "LightGBM", "CatBoost",
    "Isolation Forest", "Balanced Random Forest",
}
MODEL_CATEGORIES: dict[str, list[str]] = {
    "Ensemble": [
        "Random Forest", "Extra Trees", "Gradient Boosting",
        "Histogram Gradient Boosting", "AdaBoost", "Bagging",
        "Voting Classifier", "Stacking Classifier",
    ] + (["XGBoost"] if HAS_XGB else [])
      + (["LightGBM"] if HAS_LGBM else [])
      + (["CatBoost"] if HAS_CATBOOST else [])
      + (["Balanced Random Forest"] if HAS_IMBLEARN else []),
    "Linear & Discriminant": [
        "Logistic Regression", "Ridge Classifier", "SGD Classifier",
        "Passive Aggressive", "Perceptron",
        "Linear Discriminant Analysis", "Quadratic Discriminant Analysis",
    ],
    "Tree":           ["Decision Tree", "Extra Tree (Single)"],
    "Naive Bayes":    ["Gaussian Naive Bayes", "Bernoulli Naive Bayes", "Complement Naive Bayes"],
    "Neighbors":      ["K-Nearest Neighbors", "Nearest Centroid"],
    "SVM":            ["SVM (RBF)", "SVM (Linear)", "SVM (Poly)", "Nu-SVC", "Linear SVC"],
    "Neural Network": ["MLP Neural Network"],
    "Anomaly Detection": [
        "Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Elliptic Envelope",
    ],
}

MODELS_DIR = Path("trained_models")
MODELS_DIR.mkdir(exist_ok=True)

# ── ML helpers (mirrors updated.py, no Streamlit dependency) ──────────────────

def make_preprocessor(X: pd.DataFrame, model_name: str = "") -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if X[c].nunique() <= 50]
    scaler = MinMaxScaler() if model_name in MINMAX_MODELS else StandardScaler()
    transformers = [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", scaler)]), num_cols),
    ]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", max_categories=10)),
        ]), cat_cols))
    return ColumnTransformer(transformers=transformers)


def build_model(name: str, n: int = 150):
    if name == "Random Forest":               return RandomForestClassifier(n_estimators=n, random_state=42, class_weight="balanced")
    if name == "Extra Trees":                 return ExtraTreesClassifier(n_estimators=n, random_state=42, class_weight="balanced")
    if name == "Gradient Boosting":           return GradientBoostingClassifier(n_estimators=n, random_state=42)
    if name == "Histogram Gradient Boosting": return HistGradientBoostingClassifier(max_iter=n, random_state=42)
    if name == "AdaBoost":                    return AdaBoostClassifier(n_estimators=n, random_state=42, algorithm="SAMME")
    if name == "Bagging":                     return BaggingClassifier(n_estimators=n, random_state=42)
    if name == "Voting Classifier":
        return VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
            ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ], voting="soft")
    if name == "Stacking Classifier":
        return StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
            ],
            final_estimator=GradientBoostingClassifier(n_estimators=50, random_state=42), cv=3,
        )
    if name == "XGBoost" and HAS_XGB:         return XGBClassifier(n_estimators=n, random_state=42, eval_metric="logloss", verbosity=0)
    if name == "LightGBM" and HAS_LGBM:       return LGBMClassifier(n_estimators=n, random_state=42, verbose=-1)
    if name == "CatBoost" and HAS_CATBOOST:   return CatBoostClassifier(iterations=n, random_seed=42, verbose=0)
    if name == "Balanced Random Forest" and HAS_IMBLEARN: return BalancedRandomForestClassifier(n_estimators=n, random_state=42)
    if name == "Logistic Regression":         return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "Ridge Classifier":            return RidgeClassifier(class_weight="balanced")
    if name == "SGD Classifier":              return SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42, class_weight="balanced")
    if name == "Passive Aggressive":          return PassiveAggressiveClassifier(max_iter=1000, random_state=42, class_weight="balanced")
    if name == "Perceptron":                  return Perceptron(max_iter=1000, random_state=42, class_weight="balanced")
    if name == "Linear Discriminant Analysis":   return LinearDiscriminantAnalysis()
    if name == "Quadratic Discriminant Analysis": return QuadraticDiscriminantAnalysis()
    if name == "Decision Tree":              return DecisionTreeClassifier(random_state=42, class_weight="balanced")
    if name == "Extra Tree (Single)":        return ExtraTreeClassifier(random_state=42, class_weight="balanced")
    if name == "Gaussian Naive Bayes":       return GaussianNB()
    if name == "Bernoulli Naive Bayes":      return BernoulliNB()
    if name == "Complement Naive Bayes":     return ComplementNB()
    if name == "K-Nearest Neighbors":        return KNeighborsClassifier(n_neighbors=5)
    if name == "Nearest Centroid":           return NearestCentroid()
    if name == "SVM (RBF)":    return SVC(kernel="rbf",    probability=True, class_weight="balanced", random_state=42)
    if name == "SVM (Linear)": return SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
    if name == "SVM (Poly)":   return SVC(kernel="poly",   probability=True, class_weight="balanced", random_state=42)
    if name == "Nu-SVC":       return NuSVC(probability=True, random_state=42)
    if name == "Linear SVC":   return LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)
    if name == "MLP Neural Network":   return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    if name == "Isolation Forest":     return IsolationForest(n_estimators=n, random_state=42, contamination="auto")
    if name == "One-Class SVM":        return OneClassSVM(nu=0.1, kernel="rbf")
    if name == "Local Outlier Factor": return LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    if name == "Elliptic Envelope":    return EllipticEnvelope(contamination=0.1, random_state=42)
    return RandomForestClassifier(n_estimators=100, random_state=42)


def encode_y(y: pd.Series) -> pd.Series:
    if y.dtype == object or str(y.dtype) == "bool":
        return pd.Series(LabelEncoder().fit_transform(y.astype(str)))
    if pd.api.types.is_float_dtype(y):
        non_null = y.dropna()
        if len(non_null) > 0 and (non_null == non_null.round()).all():
            return pd.Series(y.astype(int).values).reset_index(drop=True)
    return pd.Series(y.values).reset_index(drop=True)


def generate_synthetic_fraud(fraud_df: pd.DataFrame, n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = len(fraud_df)
    fraud_reset = fraud_df.reset_index(drop=True)
    rows = []
    for _ in range(n_samples):
        tmpl = fraud_reset.iloc[rng.integers(0, n_fraud)].copy()
        row = {}
        for col in fraud_reset.columns:
            if pd.api.types.is_numeric_dtype(fraud_reset[col]):
                std = fraud_reset[col].std()
                row[col] = tmpl[col] + (rng.normal(0, std * 0.05) if std > 0 else 0)
            else:
                row[col] = fraud_reset[col].iloc[rng.integers(0, n_fraud)]
        rows.append(row)
    synth = pd.DataFrame(rows)
    for col in fraud_df.columns:
        try:
            synth[col] = synth[col].astype(fraud_df[col].dtype)
        except Exception:
            pass
    return synth


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="REST API for the Fraud Detection ML Platform. Train models, run predictions, and generate synthetic fraud data.",
    version="1.0.0",
)


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.get("/models", summary="List all available models grouped by category")
def list_models():
    return {
        "categories": MODEL_CATEGORIES,
        "total": sum(len(v) for v in MODEL_CATEGORIES.values()),
        "optional_packages": {
            "xgboost": HAS_XGB,
            "lightgbm": HAS_LGBM,
            "catboost": HAS_CATBOOST,
            "imbalanced-learn": HAS_IMBLEARN,
        },
    }


@app.post("/train", summary="Train a model on an uploaded CSV")
async def train(
    file: UploadFile = File(..., description="CSV dataset"),
    target_col: str = Form(..., description="Name of the label/target column"),
    model_name: str = Form("Random Forest", description="Model name (see GET /models)"),
    test_size: float = Form(0.2, description="Fraction for test split (0.1–0.4)"),
    n_estimators: int = Form(150, description="Number of estimators/trees (ignored for non-ensemble models)"),
):
    # Validate model name
    all_models = [m for models in MODEL_CATEGORIES.values() for m in models]
    if model_name not in all_models:
        raise HTTPException(400, f"Unknown model '{model_name}'. GET /models for the full list.")
    if not (0.1 <= test_size <= 0.4):
        raise HTTPException(400, "test_size must be between 0.1 and 0.4.")

    # Read CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if target_col not in df.columns:
        raise HTTPException(400, f"Column '{target_col}' not found. Available: {df.columns.tolist()}")

    is_anomaly = model_name in ANOMALY_MODELS
    y = encode_y(df[target_col])
    X = df.drop(columns=[target_col]).reset_index(drop=True)

    if X.shape[1] == 0:
        raise HTTPException(400, "Dataset must have at least one feature column.")

    if not is_anomaly:
        tt = type_of_target(y.dropna())
        if tt in ("continuous", "continuous-multioutput"):
            raise HTTPException(400, f"'{target_col}' contains continuous values. Classifiers need discrete labels (e.g. 0/1).")
        if y.nunique() < 2:
            raise HTTPException(400, "Target column must have at least 2 distinct classes.")

    # Train
    pre = make_preprocessor(X, model_name)
    clf = build_model(model_name, n=n_estimators)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=None if is_anomaly else y,
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    try:
        pipe.fit(X_tr, y_tr)
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")

    y_pred = pipe.predict(X_te)
    if is_anomaly:
        y_pred = np.where(y_pred == -1, 1, 0)

    y_prob = None
    if not is_anomaly and y_te.nunique() == 2:
        try:
            y_prob = pipe.predict_proba(X_te)[:, 1]
        except Exception:
            pass

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    auc  = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_te, y_prob))
        except Exception:
            pass

    cm = confusion_matrix(y_te, y_pred).tolist()
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)

    # Persist model + feature columns so /predict can use them
    model_id = str(uuid.uuid4())
    model_path = MODELS_DIR / f"{model_id}.joblib"
    joblib.dump({"pipeline": pipe, "feature_cols": X.columns.tolist(), "is_anomaly": is_anomaly}, model_path)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "is_anomaly": is_anomaly,
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }


@app.post("/predict/{model_id}", summary="Run fraud predictions using a trained model")
async def predict(
    model_id: str,
    file: UploadFile = File(..., description="CSV file with feature columns (no label column needed)"),
):
    model_path = MODELS_DIR / f"{model_id}.joblib"
    if not model_path.exists():
        raise HTTPException(404, f"Model '{model_id}' not found. Train one first via POST /train.")

    saved = joblib.load(model_path)
    pipe: Pipeline = saved["pipeline"]
    feature_cols: list[str] = saved["feature_cols"]
    is_anomaly: bool = saved["is_anomaly"]

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Keep only the columns the model was trained on (ignore extras, fill missing with NaN)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Input CSV is missing columns that the model needs: {missing}")

    X = df[feature_cols]

    try:
        preds = pipe.predict(X)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    if is_anomaly:
        preds = np.where(preds == -1, 1, 0)

    probs = None
    try:
        prob_matrix = pipe.predict_proba(X)
        # Return the probability of the positive class (index 1 for binary)
        probs = prob_matrix[:, 1].tolist() if prob_matrix.shape[1] == 2 else prob_matrix.tolist()
    except Exception:
        pass

    return {
        "model_id": model_id,
        "n_rows": len(preds),
        "predictions": preds.tolist(),
        "fraud_probabilities": probs,
    }


@app.post("/synthetic", summary="Generate synthetic fraud rows and return augmented CSV")
async def synthetic(
    file: UploadFile = File(..., description="Original CSV dataset"),
    target_col: str = Form(..., description="Name of the fraud label column"),
    fraud_value: str = Form(..., description="The value in target_col that represents fraud (e.g. '1' or 'fraud')"),
    n_samples: int = Form(500, description="Number of synthetic fraud rows to generate"),
):
    if n_samples < 1 or n_samples > 10_000:
        raise HTTPException(400, "n_samples must be between 1 and 10 000.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if target_col not in df.columns:
        raise HTTPException(400, f"Column '{target_col}' not found. Available: {df.columns.tolist()}")

    # Match fraud_value against the actual dtype of the column
    col_vals = df[target_col]
    try:
        typed_fraud_value = col_vals.dtype.type(fraud_value)
    except (ValueError, TypeError):
        typed_fraud_value = fraud_value

    fraud_rows = df[col_vals == typed_fraud_value]
    if len(fraud_rows) == 0:
        raise HTTPException(400, f"No rows found where '{target_col}' == '{fraud_value}'. "
                                  f"Unique values: {col_vals.unique().tolist()[:20]}")

    synth = generate_synthetic_fraud(fraud_rows, n_samples=n_samples)
    augmented = pd.concat([df, synth], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    csv_bytes = augmented.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=augmented_fraud_dataset.csv",
            "X-Original-Rows": str(len(df)),
            "X-Synthetic-Rows": str(n_samples),
            "X-Total-Rows": str(len(augmented)),
        },
    )
