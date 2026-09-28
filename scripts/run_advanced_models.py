from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features.feature_engineering import prepare_model_data

RANDOM_STATE = 42
CV_SPLITS = 3


def classification_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def make_model(name, scale_pos_weight):
    if name == "Logistic-balanced":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=RANDOM_STATE,
        )
    raise ValueError(f"Modelo desconocido: {name}")


def manual_stratified_cv(model_name, X, y, scale_pos_weight):
    """Validación cruzada manual para mantener compatibilidad uniforme,
    incluida CatBoost, con las versiones actuales de scikit-learn.
    """
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []

    for train_idx, valid_idx in cv.split(X, y):
        model = make_model(model_name, scale_pos_weight)
        model.fit(X[train_idx], y.iloc[train_idx])
        prob = model.predict_proba(X[valid_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_results.append(classification_metrics(y.iloc[valid_idx], pred, prob))

    return {
        metric: float(np.mean([fold[metric] for fold in fold_results]))
        for metric in fold_results[0]
    }


def tune_xgboost(X, y, scale_pos_weight, n_iter=12):
    param_dist = {
        "n_estimators": [300, 500, 700],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.04, 0.06, 0.08],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.70, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for params in ParameterSampler(param_dist, n_iter=n_iter, random_state=RANDOM_STATE):
        fold_results = []
        for train_idx, valid_idx in cv.split(X, y):
            model = XGBClassifier(
                **params,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            model.fit(X[train_idx], y.iloc[train_idx])
            prob = model.predict_proba(X[valid_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_results.append(classification_metrics(y.iloc[valid_idx], pred, prob))

        row = {
            metric: float(np.mean([fold[metric] for fold in fold_results]))
            for metric in fold_results[0]
        }
        row.update(params)
        rows.append(row)

    tuning = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    best = tuning.iloc[0].to_dict()
    best_params = {key: best[key] for key in param_dist}
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    return tuning, best_params


def evaluate_test_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    result = classification_metrics(y_test, pred, prob)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    result.update({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return result, model, prob


def run_advanced_models(df):
    prepared = prepare_model_data(df)
    X_train = prepared["X_train_t"]
    X_test = prepared["X_test_t"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model_names = [
        "Logistic-balanced",
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "CatBoost",
    ]

    cv_rows = []
    for name in model_names:
        started = time.time()
        scores = manual_stratified_cv(name, X_train, y_train, scale_pos_weight)
        scores.update({"model": name, "seconds": time.time() - started})
        cv_rows.append(scores)

    cv_results = pd.DataFrame(cv_rows).sort_values(["roc_auc", "pr_auc"], ascending=False)

    tuning, best_xgb_params = tune_xgboost(
        X_train, y_train, scale_pos_weight=scale_pos_weight, n_iter=12
    )

    test_rows = []
    fitted = {}
    probabilities = {}

    for name in model_names:
        result, model, prob = evaluate_test_model(
            name,
            make_model(name, scale_pos_weight),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        test_rows.append(result)
        fitted[name] = model
        probabilities[name] = prob

    tuned_xgb = XGBClassifier(
        **best_xgb_params,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    result, model, prob = evaluate_test_model(
        "XGBoost-tuned", tuned_xgb, X_train, y_train, X_test, y_test
    )
    test_rows.append(result)
    fitted["XGBoost-tuned"] = model
    probabilities["XGBoost-tuned"] = prob

    ensemble_prob = np.mean(
        np.column_stack(
            [
                probabilities["GradientBoosting"],
                probabilities["XGBoost"],
                probabilities["CatBoost"],
            ]
        ),
        axis=1,
    )
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ensemble_result = classification_metrics(y_test, ensemble_pred, ensemble_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, ensemble_pred).ravel()
    ensemble_result.update(
        {"model": "Ensemble-GB-XGB-CAT", "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    )
    test_rows.append(ensemble_result)

    test_results = pd.DataFrame(test_rows).sort_values(["roc_auc", "pr_auc"], ascending=False)

    return {
        "prepared": prepared,
        "cv_results": cv_results,
        "tuning": tuning,
        "best_xgb_params": best_xgb_params,
        "test_results": test_results,
        "fitted_models": fitted,
        "ensemble_prob": ensemble_prob,
    }


if __name__ == "__main__":
    data_path = ROOT / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(data_path)

    result = run_advanced_models(df)

    print("ADVANCED MODELS OK")
    print("\nCross-validation (3-fold):")
    print(
        result["cv_results"][["model", "roc_auc", "pr_auc", "recall", "f1"]]
        .round(4)
        .to_string(index=False)
    )

    print("\nBest XGBoost params:")
    print(result["best_xgb_params"])

    print("\nTest set:")
    print(
        result["test_results"][[
            "model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
        ]]
        .round(4)
        .to_string(index=False)
    )

    assert result["cv_results"]["roc_auc"].max() > 0.90
    assert result["test_results"]["roc_auc"].max() > 0.91
    assert result["test_results"]["pr_auc"].max() > 0.53
    assert result["test_results"]["recall"].max() > 0.90
