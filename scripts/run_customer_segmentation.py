from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

CLUSTER_FEATURES = [
    "total_spent",
    "last_3_month_purchase_freq",
    "recency_days",
    "satisfaction_score",
    "support_tickets",
]


def prepare_segmentation_table(df: pd.DataFrame):
    data = df.copy()
    for col in ["signup_date", "last_purchase_date"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")

    reference_date = max(
        data["signup_date"].max(),
        data["last_purchase_date"].max(),
    ) + pd.Timedelta(days=1)

    data["recency_days"] = (reference_date - data["last_purchase_date"]).dt.days

    X = data[CLUSTER_FEATURES].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X),
        columns=CLUSTER_FEATURES,
        index=X.index,
    )

    capped_cols = [
        "total_spent",
        "last_3_month_purchase_freq",
        "recency_days",
        "support_tickets",
    ]
    caps = {}
    for col in capped_cols:
        lo, hi = X_imp[col].quantile([0.01, 0.99])
        caps[col] = (float(lo), float(hi))
        X_imp[col] = X_imp[col].clip(lo, hi)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return data, X_imp, X_scaled, reference_date, imputer, scaler, caps


def compare_k(X_scaled, min_k=2, max_k=8):
    rows = []
    models = {}

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=min(3000, len(X_scaled)),
                    random_state=RANDOM_STATE,
                ),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
                "inertia": model.inertia_,
            }
        )
        models[k] = (model, labels)

    return pd.DataFrame(rows), models


def clustering_stability(X_scaled, reference_labels, k=4):
    rows = []
    for seed in [1, 7, 21, 42, 99]:
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed,
        ).fit_predict(X_scaled)
        rows.append(
            {
                "seed": seed,
                "adjusted_rand_index": adjusted_rand_score(reference_labels, labels),
            }
        )
    return pd.DataFrame(rows)


def profile_segments(data, labels):
    result = data.copy()
    result["segment"] = labels

    profile = (
        result.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            churn_rate=("churn", "mean"),
            total_spent=("total_spent", "mean"),
            purchase_freq=("last_3_month_purchase_freq", "mean"),
            recency_days=("recency_days", "mean"),
            satisfaction=("satisfaction_score", "mean"),
            support_tickets=("support_tickets", "mean"),
        )
        .reset_index()
    )
    profile["share"] = profile["customers"] / len(result)

    return result, profile


def pca_projection(X_scaled, labels):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    projected = pd.DataFrame(
        {
            "pca_1": coords[:, 0],
            "pca_2": coords[:, 1],
            "segment": labels,
        }
    )
    return projected, pca.explained_variance_ratio_


def run_segmentation(df):
    (
        data,
        X_imp,
        X_scaled,
        reference_date,
        imputer,
        scaler,
        caps,
    ) = prepare_segmentation_table(df)

    comparison, models = compare_k(X_scaled)

    selected_k = 4
    model, labels = models[selected_k]

    stability = clustering_stability(X_scaled, labels, k=selected_k)
    segmented, profile = profile_segments(data, labels)
    projection, explained_variance = pca_projection(X_scaled, labels)

    assert len(segmented) == len(df)
    assert segmented["segment"].nunique() == selected_k
    assert segmented["segment"].isna().sum() == 0
    assert stability["adjusted_rand_index"].mean() > 0.90
    assert "churn" not in CLUSTER_FEATURES

    return {
        "reference_date": reference_date,
        "features": CLUSTER_FEATURES,
        "caps": caps,
        "comparison": comparison,
        "selected_k": selected_k,
        "model": model,
        "labels": labels,
        "stability": stability,
        "segmented": segmented,
        "profile": profile,
        "projection": projection,
        "pca_explained_variance": explained_variance,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(data_path)

    result = run_segmentation(df)

    print("SEGMENTATION OK")
    print("Fecha de referencia:", result["reference_date"].date())
    print("Features:", result["features"])
    print("\nComparación de k:")
    print(result["comparison"].round(4).to_string(index=False))
    print("\nEstabilidad:")
    print(result["stability"].round(4).to_string(index=False))
    print("\nPerfil de segmentos:")
    print(result["profile"].round(4).to_string(index=False))
