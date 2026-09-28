from pathlib import Path
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.features.feature_engineering import prepare_model_data

RANDOM_STATE = 42


def build_explainable_xgb(scale_pos_weight):
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


def run_shap_explainability(df, save_outputs=False):
    prepared = prepare_model_data(df)
    X_train_t = prepared["X_train_t"]
    X_test_t = prepared["X_test_t"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    feature_names = np.asarray(prepared["feature_names"])

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_explainable_xgb(scale_pos_weight)
    model.fit(X_train_t, y_train)

    probabilities = model.predict_proba(X_test_t)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test_t)
    shap_values = explanation.values

    global_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # Elegimos un verdadero positivo de riesgo alto para una explicación local reproducible.
    positive_positions = np.where(y_test.to_numpy() == 1)[0]
    local_position = int(positive_positions[np.argmax(probabilities[positive_positions])])
    original_index = prepared["X_test"].index[local_position]
    customer_id = prepared["data"].loc[original_index, "customer_id"]
    actual_churn = int(y_test.iloc[local_position])

    local = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[local_position],
        "transformed_value": X_test_t[local_position],
    })
    local["abs_shap"] = local["shap_value"].abs()
    local["direction"] = np.where(local["shap_value"] > 0, "sube riesgo", "baja riesgo")

    raw_row = prepared["X_test"].iloc[local_position]

    def raw_value_for_feature(transformed_name):
        if transformed_name.startswith("num__"):
            original = transformed_name.replace("num__", "", 1)
            return raw_row.get(original, np.nan)
        if transformed_name.startswith("cat__"):
            return "one-hot"
        return np.nan

    local["raw_value"] = [raw_value_for_feature(name) for name in local["feature"]]
    local = local.sort_values("abs_shap", ascending=False)

    # Validación de aditividad: base + SHAP debe reconstruir el margen del modelo.
    base_value = float(np.asarray(explanation.base_values[local_position]).reshape(-1)[0])
    reconstructed_margin = base_value + float(shap_values[local_position].sum())
    model_margin = float(model.predict(X_test_t[[local_position]], output_margin=True)[0])
    assert np.isclose(reconstructed_margin, model_margin, atol=1e-4)

    summary = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "customer_id": str(customer_id),
        "actual_churn": actual_churn,
        "predicted_probability": float(probabilities[local_position]),
        "base_value": base_value,
        "model_margin": model_margin,
    }

    if save_outputs:
        output_dir = ROOT / "reports" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        tables_dir = ROOT / "reports" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        global_importance.to_csv(tables_dir / "shap_global_importance.csv", index=False)
        local.head(15).to_csv(tables_dir / "shap_local_example.csv", index=False)

        shap.summary_plot(
            shap_values,
            X_test_t,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()

        top = global_importance.head(15).sort_values("mean_abs_shap")
        plt.figure(figsize=(9, 6))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Media de |SHAP|")
        plt.title("Importancia global SHAP")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_global_importance.png", dpi=160, bbox_inches="tight")
        plt.close()

        for original_feature in ["satisfaction_score", "total_spent", "support_tickets"]:
            transformed_feature = f"num__{original_feature}"
            feature_idx = int(np.where(feature_names == transformed_feature)[0][0])
            raw_values = prepared["X_test"][original_feature].to_numpy()
            plt.figure(figsize=(7, 5))
            plt.scatter(raw_values, shap_values[:, feature_idx], alpha=0.25)
            plt.axhline(0, linewidth=1)
            plt.xlabel(original_feature)
            plt.ylabel("Valor SHAP")
            plt.title(f"Efecto SHAP de {original_feature}")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_dependence_{original_feature}.png", dpi=160, bbox_inches="tight")
            plt.close()

    return {
        "prepared": prepared,
        "model": model,
        "explainer": explainer,
        "explanation": explanation,
        "probabilities": probabilities,
        "global_importance": global_importance,
        "local_explanation": local,
        "summary": summary,
    }


if __name__ == "__main__":
    data_path = ROOT / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(data_path)
    result = run_shap_explainability(df, save_outputs=True)

    print("SHAP EXPLAINABILITY OK")
    print("ROC-AUC:", round(result["summary"]["roc_auc"], 4))
    print("PR-AUC :", round(result["summary"]["pr_auc"], 4))
    print("\nTop 10 variables globales:")
    print(result["global_importance"].head(10).round(4).to_string(index=False))
    print("\nEjemplo local:")
    print(result["summary"])
    print(result["local_explanation"].head(10).round(4).to_string(index=False))

    assert result["summary"]["roc_auc"] > 0.91
    assert result["summary"]["pr_auc"] > 0.53
    assert len(result["global_importance"]) == result["prepared"]["X_test_t"].shape[1]
