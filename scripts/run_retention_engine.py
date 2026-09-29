from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features.feature_engineering import prepare_model_data

RANDOM_STATE = 42
SEGMENT_FEATURES = [
    "total_spent",
    "last_3_month_purchase_freq",
    "recency_days",
    "satisfaction_score",
    "support_tickets",
]


def _percentile_against(train_series, target_series):
    train = np.sort(pd.Series(train_series).dropna().values)
    fill = float(np.nanmedian(train))
    values = pd.Series(target_series).fillna(fill).values
    return np.searchsorted(train, values, side="right") / len(train)


def fit_segments(data, train_index, test_index):
    train = data.loc[train_index, SEGMENT_FEATURES].copy()
    test = data.loc[test_index, SEGMENT_FEATURES].copy()

    imputer = SimpleImputer(strategy="median")
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=SEGMENT_FEATURES, index=train_index)
    test_imp = pd.DataFrame(imputer.transform(test), columns=SEGMENT_FEATURES, index=test_index)

    for col in ["total_spent", "last_3_month_purchase_freq", "recency_days", "support_tickets"]:
        lo, hi = train_imp[col].quantile([0.01, 0.99])
        train_imp[col] = train_imp[col].clip(lo, hi)
        test_imp[col] = test_imp[col].clip(lo, hi)

    scaler = StandardScaler().fit(train_imp)
    X_train = scaler.transform(train_imp)
    X_test = scaler.transform(test_imp)

    model = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    train_labels = model.fit_predict(X_train)
    test_labels = model.predict(X_test)

    profile = []
    for cluster in range(4):
        idx = train_index[train_labels == cluster]
        row = {
            "segment_id": cluster,
            "customers": len(idx),
            "churn_rate": float(data.loc[idx, "churn"].mean()),
        }
        for col in SEGMENT_FEATURES:
            row[col] = float(data.loc[idx, col].mean())
        profile.append(row)

    profile = pd.DataFrame(profile)

    names = {}
    for _, row in profile.iterrows():
        cluster = int(row["segment_id"])
        if row["satisfaction_score"] < 2.5:
            names[cluster] = "Insatisfechos en riesgo"
        elif row["support_tickets"] > 3:
            names[cluster] = "Alta fricción de soporte"
        elif row["last_3_month_purchase_freq"] < 5:
            names[cluster] = "Inactivos satisfechos"
        else:
            names[cluster] = "Activos satisfechos"

    return test_labels, profile, names


def fit_ensemble(prepared):
    X_train = prepared["X_train_t"]
    X_test = prepared["X_test_t"]
    y_train = prepared["y_train"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    gb = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)
    xgb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
        scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=RANDOM_STATE,
    )
    cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss",
        eval_metric="AUC", auto_class_weights="Balanced", verbose=False,
        random_seed=RANDOM_STATE,
    )

    models = [gb, xgb, cat]
    probs = []
    for model in models:
        model.fit(X_train, y_train)
        probs.append(model.predict_proba(X_test)[:, 1])

    return np.mean(np.column_stack(probs), axis=1), models


def recommended_action(row):
    if pd.notna(row["satisfaction_score"]) and row["satisfaction_score"] <= 2:
        return "Recuperación de experiencia: contacto personalizado + resolución de causa"
    if row["support_tickets"] >= 5:
        return "Escalamiento de soporte y seguimiento preventivo"
    if row["last_3_month_purchase_freq"] <= 3:
        return "Reactivación con oferta personalizada"
    if row["value_score"] >= 0.75:
        return "Retención VIP / beneficio de fidelización"
    return "Seguimiento preventivo multicanal"


def build_retention_engine(df):
    prepared = prepare_model_data(df)
    data = prepared["data"]
    train_index = prepared["X_train"].index
    test_index = prepared["X_test"].index
    y_test = prepared["y_test"]

    probabilities, fitted_models = fit_ensemble(prepared)
    test_labels, segment_profile, segment_names = fit_segments(data, train_index, test_index)

    result = data.loc[test_index, [
        "customer_id", "total_spent", "avg_order_value", "support_tickets",
        "satisfaction_score", "last_3_month_purchase_freq", "recency_days", "churn",
    ]].copy()

    result["churn_probability"] = probabilities
    result["segment_id"] = test_labels
    result["segment"] = result["segment_id"].map(segment_names)

    spent_score = _percentile_against(data.loc[train_index, "total_spent"], result["total_spent"])
    order_score = _percentile_against(data.loc[train_index, "avg_order_value"], result["avg_order_value"])
    result["value_score"] = 0.70 * spent_score + 0.30 * order_score

    segment_churn = dict(zip(segment_profile["segment_id"], segment_profile["churn_rate"]))
    rates = np.array(list(segment_churn.values()))
    low, high = rates.min(), rates.max()
    normalized_segment_risk = {key: (value - low) / (high - low) for key, value in segment_churn.items()}
    result["segment_risk_score"] = result["segment_id"].map(normalized_segment_risk)

    result["priority_score"] = 100 * (
        0.65 * result["churn_probability"]
        + 0.25 * result["value_score"]
        + 0.10 * result["segment_risk_score"]
    )

    result["priority_band"] = pd.cut(
        result["priority_score"], bins=[-1, 40, 60, 75, 101], labels=["Baja", "Media", "Alta", "Crítica"]
    )
    result["recommended_action"] = result.apply(recommended_action, axis=1)
    result = result.sort_values("priority_score", ascending=False)

    base_rate = float(y_test.mean())
    evaluation = []
    for fraction in [0.05, 0.10, 0.20, 0.30]:
        n = max(1, int(len(result) * fraction))
        top = result.head(n)
        churn_rate = float(top["churn"].mean())
        evaluation.append({
            "top_fraction": fraction,
            "customers": n,
            "churn_rate": churn_rate,
            "lift": churn_rate / base_rate,
            "capture_rate": float(top["churn"].sum() / y_test.sum()),
        })
    evaluation = pd.DataFrame(evaluation)

    model_metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "base_churn_rate": base_rate,
    }

    assert result["customer_id"].is_unique
    assert result["priority_score"].between(0, 100).all()
    assert result["churn_probability"].between(0, 1).all()
    assert evaluation.loc[evaluation["top_fraction"] == 0.20, "capture_rate"].iloc[0] > 0.60
    assert evaluation.loc[evaluation["top_fraction"] == 0.10, "lift"].iloc[0] > 2.5
    assert model_metrics["roc_auc"] > 0.91
    assert model_metrics["pr_auc"] > 0.53

    return {
        "ranking": result,
        "evaluation": evaluation,
        "segment_profile": segment_profile,
        "model_metrics": model_metrics,
        "fitted_models": fitted_models,
    }


if __name__ == "__main__":
    data_path = ROOT / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(data_path)
    output = build_retention_engine(df)

    print("RETENTION ENGINE OK")
    print("\nMétricas del ensemble:")
    print(pd.Series(output["model_metrics"]).round(4).to_string())
    print("\nEvaluación del ranking:")
    print(output["evaluation"].round(4).to_string(index=False))
    print("\nBandas de prioridad:")
    print(output["ranking"]["priority_band"].value_counts().sort_index().to_string())
