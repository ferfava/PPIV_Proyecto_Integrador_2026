from pathlib import Path
import sys
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features.feature_engineering import prepare_model_data

RANDOM_STATE = 42


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }, model


def run_baselines(df):
    prepared = prepare_model_data(df)

    X_train = prepared["X_train_t"]
    X_test = prepared["X_test_t"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]

    candidates = {
        "Dummy-most-frequent": DummyClassifier(strategy="most_frequent"),
        "Logistic": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Logistic-balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted = {}

    for name, model in candidates.items():
        metrics, fitted_model = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        fitted[name] = fitted_model

    metrics_df = pd.DataFrame(results).sort_values(
        ["roc_auc", "pr_auc"], ascending=False
    )

    coef_df = pd.DataFrame(
        {
            "feature": prepared["feature_names"],
            "coefficient": fitted["Logistic-balanced"].coef_[0],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    return prepared, metrics_df, coef_df


if __name__ == "__main__":
    data_path = ROOT / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(data_path)

    prepared, metrics, coefficients = run_baselines(df)

    print("BASELINES OK")
    print(metrics.round(4).to_string(index=False))
    print("\nTop 10 coeficientes | Logistic-balanced")
    print(coefficients.head(10).round(4).to_string(index=False))

    assert metrics["roc_auc"].notna().all()
    assert metrics["pr_auc"].notna().all()
    assert (
        metrics.loc[metrics["model"] == "Dummy-most-frequent", "recall"].iloc[0] == 0
    )
    assert (
        metrics.loc[metrics["model"] == "Logistic-balanced", "recall"].iloc[0] > 0.65
    )
