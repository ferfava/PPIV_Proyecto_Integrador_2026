from pathlib import Path
import sys
import pandas as pd

REQUIRED_COLUMNS = {
    "customer_id",
    "signup_date",
    "last_purchase_date",
    "churn",
}


def find_dataset() -> Path:
    candidates = [
        Path("data/raw/sales_marketing_customer_raw.csv"),
        Path("../data/raw/sales_marketing_customer_raw.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No se encontró data/raw/sales_marketing_customer_raw.csv"
    )


def main() -> int:
    path = find_dataset()
    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        print(f"ERROR: faltan columnas requeridas: {sorted(missing_columns)}")
        return 1

    errors = []

    if df.empty:
        errors.append("El dataset está vacío.")

    if df["customer_id"].isna().any():
        errors.append("customer_id contiene valores nulos.")

    if df["customer_id"].duplicated().any():
        errors.append("customer_id contiene duplicados.")

    churn_values = set(df["churn"].dropna().unique())
    if not churn_values.issubset({0, 1}):
        errors.append(f"churn contiene valores inesperados: {sorted(churn_values)}")

    if errors:
        print("VALIDACIÓN FALLIDA")
        for error in errors:
            print(f"- {error}")
        return 1

    print("VALIDACIÓN OK")
    print(f"Archivo: {path.resolve()}")
    print(f"Filas: {len(df):,}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Tasa de churn: {df['churn'].mean() * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
