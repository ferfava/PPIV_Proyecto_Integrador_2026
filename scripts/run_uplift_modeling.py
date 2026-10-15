from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features.feature_engineering import prepare_model_data

RANDOM_STATE = 42


def simulate_retention_experiment(X_df, y_base, random_state=RANDOM_STATE):
    """
    Simula un experimento controlado de retención (A/B Test):
    - 50% de clientes asignados a Control (T=0, sin intervención adicional).
    - 50% asignados a Tratamiento (T=1, intervención activa de retención).

    Efecto causal fundamentado en las señales del negocio:
    1. Persuadibles (baja satisfacción <= 2 o alta fricción de soporte >= 3):
       La intervención proactiva resuelve causas de fondo y reduce el riesgo de churn en ~65%.
    2. Perros dormidos (alta satisfacción >= 4.5, 0 tickets, pocas visitas):
       Clientes pasivos donde un contacto intrusivo reactiva la intención de baja en ~8%.
    3. Seguros y causas perdidas: la intervención no altera sustancialmente su decisión.
    """
    rng = np.random.RandomState(random_state)
    n = len(X_df)
    t_arr = rng.binomial(1, 0.5, size=n)
    y_obs = y_base.copy().values

    persuadable = (X_df["satisfaction_score"] <= 2.5) | (X_df["support_tickets"] >= 3)
    sleeping_dog = (
        (X_df["satisfaction_score"] >= 4.5)
        & (X_df["support_tickets"] == 0)
        & (X_df["total_visits"] <= 10)
    )

    for i in range(n):
        if t_arr[i] == 1:
            if persuadable.iloc[i]:
                if y_obs[i] == 1 and rng.rand() < 0.65:
                    y_obs[i] = 0
            elif sleeping_dog.iloc[i]:
                if y_obs[i] == 0 and rng.rand() < 0.08:
                    y_obs[i] = 1

    return t_arr, y_obs


def fit_uplift_t_learner(X_train_t, y_train_obs, t_train, X_test_t, random_state=RANDOM_STATE):
    """
    Implementa un T-Learner (Two-Model Approach) con Gradient Boosting:
    - M0: modela P(Churn | X, Control)
    - M1: modela P(Churn | X, Tratamiento)
    - Uplift individual = P(Churn | Control) - P(Churn | Tratamiento)
      (reducción neta de abandono gracias a la intervención)
    """
    # Modelo para grupo Control (T=0)
    m0 = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        random_state=random_state,
    )
    m0.fit(X_train_t[t_train == 0], y_train_obs[t_train == 0])

    # Modelo para grupo Tratamiento (T=1)
    m1 = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.08,
        random_state=random_state,
    )
    m1.fit(X_train_t[t_train == 1], y_train_obs[t_train == 1])

    p_churn_control = m0.predict_proba(X_test_t)[:, 1]
    p_churn_treated = m1.predict_proba(X_test_t)[:, 1]
    uplift_score = p_churn_control - p_churn_treated

    return {
        "m0": m0,
        "m1": m1,
        "p_churn_control": p_churn_control,
        "p_churn_treated": p_churn_treated,
        "uplift_score": uplift_score,
    }


def evaluate_uplift_deciles(test_eval_df):
    """
    Evalúa el ranking de uplift dividiendo el test en deciles de uplift estimado.
    """
    df = test_eval_df.copy()
    df["decil"] = pd.qcut(df["uplift_score"], q=10, labels=False)
    # Invertir para que decil 1 sea el de mayor uplift
    df["decil_prioridad"] = 10 - df["decil"]

    summary = []
    for d, grp in df.groupby("decil_prioridad"):
        n_total = len(grp)
        grp_c = grp[grp["treatment"] == 0]
        grp_t = grp[grp["treatment"] == 1]

        rate_c = grp_c["churn_observed"].mean() if len(grp_c) > 0 else 0.0
        rate_t = grp_t["churn_observed"].mean() if len(grp_t) > 0 else 0.0
        actual_uplift = rate_c - rate_t
        predicted_uplift = grp["uplift_score"].mean()

        summary.append({
            "decil": int(d),
            "clientes": n_total,
            "clientes_control": len(grp_c),
            "clientes_tratados": len(grp_t),
            "churn_control": rate_c,
            "churn_tratados": rate_t,
            "uplift_real": actual_uplift,
            "uplift_predicho": predicted_uplift,
        })

    return pd.DataFrame(summary).sort_values("decil").reset_index(drop=True)


def assign_quadrants(row):
    """
    Clasifica a un cliente en los 4 cuadrantes estándar de Uplift:
    - Persuadibles
    - Seguros (Sure Things)
    - Causas Perdidas (Lost Causes)
    - Perros Dormidos (Sleeping Dogs)
    """
    uplift = row["uplift_score"]
    base_risk = row["p_churn_control"]

    if uplift >= 0.10:
        return "Persuadible"
    elif uplift < -0.02:
        return "Perro dormido"
    elif base_risk < 0.25:
        return "Seguro"
    else:
        return "Causa perdida"


def recommend_action(quadrant, satisfaction, tickets):
    if quadrant == "Persuadible":
        if satisfaction <= 2:
            return "Intervención prioritaria: resolución de insatisfacción y beneficio comercial"
        elif tickets >= 3:
            return "Escalamiento a soporte dedicado con seguimiento VIP"
        else:
            return "Oferta proactiva de retención / fidelización"
    elif quadrant == "Seguro":
        return "Fidelización pasiva: comunicación de valor sin descuentos innecesarios"
    elif quadrant == "Causa perdida":
        return "No intervenir comercialmente (ahorro de presupuesto)"
    else: # Perro dormido
        return "Exclusión estricta de campañas (Do Not Disturb)"


def compare_churn_vs_uplift(test_eval_df, fractions=[0.05, 0.10, 0.20, 0.30]):
    """
    Compara la efectividad comercial entre:
    1. Política Tradicional (priorizar por probabilidad de churn descendente)
    2. Política Uplift (priorizar por score de uplift descendente)
    """
    df = test_eval_df.copy()
    n = len(df)
    results = []

    for frac in fractions:
        k = int(n * frac)

        # Selección Tradicional (mayor riesgo de churn bajo control)
        top_trad = df.sort_values("p_churn_control", ascending=False).head(k)
        trad_c = top_trad[top_trad["treatment"] == 0]
        trad_t = top_trad[top_trad["treatment"] == 1]
        rate_trad_c = trad_c["churn_observed"].mean() if len(trad_c) > 0 else 0
        rate_trad_t = trad_t["churn_observed"].mean() if len(trad_t) > 0 else 0
        uplift_trad = rate_trad_c - rate_trad_t
        clientes_salvados_trad = uplift_trad * k

        # Selección Uplift (mayor reducción estimada de churn)
        top_uplift = df.sort_values("uplift_score", ascending=False).head(k)
        up_c = top_uplift[top_uplift["treatment"] == 0]
        up_t = top_uplift[top_uplift["treatment"] == 1]
        rate_up_c = up_c["churn_observed"].mean() if len(up_c) > 0 else 0
        rate_up_t = up_t["churn_observed"].mean() if len(up_t) > 0 else 0
        uplift_up = rate_up_c - rate_up_t
        clientes_salvados_uplift = uplift_up * k

        ganancia_incremental = clientes_salvados_uplift - clientes_salvados_trad
        eficiencia_vs_tradicional = (clientes_salvados_uplift / max(clientes_salvados_trad, 1e-6) - 1) * 100

        results.append({
            "fraccion_capacidad": f"Top {int(frac*100)}%",
            "clientes_contactados": k,
            "salvados_tradicional": round(clientes_salvados_trad, 1),
            "salvados_uplift": round(clientes_salvados_uplift, 1),
            "clientes_adicionales_salvados": round(ganancia_incremental, 1),
            "mejora_porcentual": f"+{round(eficiencia_vs_tradicional, 1)}%",
        })

    return pd.DataFrame(results)


def run_uplift_pipeline(df: pd.DataFrame):
    """
    Ejecuta el pipeline completo de Uplift Modeling.
    """
    prep = prepare_model_data(df)
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    X_train_t = prep["X_train_t"]
    X_test_t = prep["X_test_t"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    # Simulación de asignación y respuesta
    t_train, y_train_obs = simulate_retention_experiment(X_train, y_train, random_state=RANDOM_STATE)
    t_test, y_test_obs = simulate_retention_experiment(X_test, y_test, random_state=RANDOM_STATE + 1)

    # Ajuste T-Learner
    models = fit_uplift_t_learner(X_train_t, y_train_obs, t_train, X_test_t, random_state=RANDOM_STATE)

    test_eval = pd.DataFrame({
        "customer_id": df.loc[X_test.index, "customer_id"].values,
        "satisfaction_score": X_test["satisfaction_score"].values,
        "support_tickets": X_test["support_tickets"].values,
        "total_spent": X_test["total_spent"].values,
        "treatment": t_test,
        "churn_base": y_test.values,
        "churn_observed": y_test_obs,
        "p_churn_control": models["p_churn_control"],
        "p_churn_treated": models["p_churn_treated"],
        "uplift_score": models["uplift_score"],
    })

    test_eval["cuadrante"] = test_eval.apply(assign_quadrants, axis=1)
    test_eval["accion_sugerida"] = test_eval.apply(
        lambda r: recommend_action(r["cuadrante"], r["satisfaction_score"], r["support_tickets"]),
        axis=1,
    )

    deciles = evaluate_uplift_deciles(test_eval)
    comparacion = compare_churn_vs_uplift(test_eval)

    return {
        "models": models,
        "test_eval": test_eval,
        "deciles": deciles,
        "comparacion": comparacion,
        "cuadrantes_dist": test_eval["cuadrante"].value_counts(normalize=True).round(4) * 100,
    }


if __name__ == "__main__":
    raw_path = ROOT / "data" / "raw" / "sales_marketing_customer_raw.csv"
    df = pd.read_csv(raw_path)
    res = run_uplift_pipeline(df)
    print("\n--- DECILES DE UPLIFT ---")
    print(res["deciles"][["decil", "clientes", "churn_control", "churn_tratados", "uplift_real", "uplift_predicho"]].to_string(index=False))
    print("\n--- DISTRIBUCIÓN DE CUADRANTES ---")
    print(res["cuadrantes_dist"])
    print("\n--- COMPARACIÓN POLÍTICA TRADICIONAL VS UPLIFT ---")
    print(res["comparacion"].to_string(index=False))
