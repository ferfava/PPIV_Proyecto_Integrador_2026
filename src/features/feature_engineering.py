from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def build_model_table(df: pd.DataFrame):
    data = df.copy()

    for col in ["signup_date", "last_purchase_date"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")

    reference_date = max(
        data["signup_date"].max(),
        data["last_purchase_date"].max()
    ) + pd.Timedelta(days=1)

    data["age_invalid"] = data["age"].notna() & ~data["age"].between(18, 100)
    data.loc[data["age_invalid"], "age"] = np.nan

    data["temporal_inconsistency"] = (
        data["signup_date"].notna()
        & data["last_purchase_date"].notna()
        & (data["last_purchase_date"] < data["signup_date"])
    )

    data["customer_tenure_days"] = (reference_date - data["signup_date"]).dt.days
    data["recency_days"] = (reference_date - data["last_purchase_date"]).dt.days
    data["has_coupon"] = data["coupon_code"].notna().astype(int)

    data["email_engagement_ratio"] = np.where(
        data["email_open_rate"].fillna(0) > 0,
        data["email_click_rate"] / data["email_open_rate"],
        0.0,
    )

    data["spend_per_visit"] = np.where(
        data["total_visits"].fillna(0) > 0,
        data["total_spent"] / data["total_visits"],
        np.nan,
    )

    data["support_ticket_rate"] = np.where(
        data["total_visits"].fillna(0) > 0,
        data["support_tickets"] / data["total_visits"],
        np.nan,
    )

    data["purchase_frequency_per_month"] = data["last_3_month_purchase_freq"] / 3.0

    for col in ["age", "total_spent", "gender", "satisfaction_score"]:
        data[f"{col}_missing"] = data[col].isna().astype(int)

    return data, reference_date


def prepare_model_data(df: pd.DataFrame):
    data, reference_date = build_model_table(df)

    target = "churn"
    excluded = [
        "customer_id",
        "city",
        "signup_date",
        "last_purchase_date",
        "coupon_code",
        "lifetime_value",
        "age_invalid",
        "temporal_inconsistency",
        target,
    ]

    features = [col for col in data.columns if col not in excluded]
    X = data[features].copy()
    y = data[target].astype(int)

    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    assert X_train.shape[0] + X_test.shape[0] == len(data)
    assert np.isnan(X_train_t).sum() == 0
    assert np.isnan(X_test_t).sum() == 0
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert X_train_t.shape[1] == X_test_t.shape[1]

    return {
        "data": data,
        "reference_date": reference_date,
        "features": features,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "feature_names": preprocessor.get_feature_names_out(),
    }


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(data_path)
    result = prepare_model_data(df)

    print("VALIDACIÓN OK")
    print("Fecha de referencia:", result["reference_date"].date())
    print("Train:", result["X_train"].shape, "| churn:", round(result["y_train"].mean(), 4))
    print("Test :", result["X_test"].shape, "| churn:", round(result["y_test"].mean(), 4))
    print("Features originales:", len(result["features"]))
    print("Features transformadas:", len(result["feature_names"]))
