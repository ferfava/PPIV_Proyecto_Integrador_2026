from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

DATA_PATH = Path('data/raw/sales_marketing_customer_raw.csv')


def prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    df.loc[df['age'] < 18, 'age'] = np.nan
    return df


def evaluate_binary_hypothesis(data, group_col, label):
    tab = pd.crosstab(data[group_col], data['churn'])
    if True not in tab.index or False not in tab.index:
        raise ValueError(f'{group_col}: se requieren grupos True y False')

    pos = np.array([tab.loc[True, 1], tab.loc[False, 1]])
    nobs = np.array([tab.loc[True].sum(), tab.loc[False].sum()])
    rate_true = pos[0] / nobs[0]
    rate_false = pos[1] / nobs[1]

    z_stat, p_one_sided = proportions_ztest(pos, nobs, alternative='larger')
    chi2, p_chi2, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * min(tab.shape[0] - 1, tab.shape[1] - 1)))

    rr = rate_true / rate_false
    odds_true = pos[0] / (nobs[0] - pos[0])
    odds_false = pos[1] / (nobs[1] - pos[1])
    odds_ratio = odds_true / odds_false

    return {
        'hipotesis': label,
        'n_grupo_riesgo': int(nobs[0]),
        'n_comparacion': int(nobs[1]),
        'churn_grupo_riesgo': rate_true,
        'churn_comparacion': rate_false,
        'diferencia_pp': (rate_true - rate_false) * 100,
        'risk_ratio': rr,
        'odds_ratio': odds_ratio,
        'z_stat': z_stat,
        'p_value_unilateral': p_one_sided,
        'chi2': chi2,
        'p_value_chi2': p_chi2,
        'cramers_v': cramers_v
    }


def main():
    df = prepare()
    results = []

    # H1: satisfacción baja (1–2) vs satisfacción media/alta (3–5).
    # Los faltantes se excluyen del contraste para no asignarles una categoría artificial.
    h1 = df[df['satisfaction_score'].isin([1, 2, 3, 4, 5])].copy()
    h1['grupo_riesgo'] = h1['satisfaction_score'].isin([1, 2])
    results.append(evaluate_binary_hypothesis(
        h1, 'grupo_riesgo',
        'H1: satisfacción baja (1–2) > satisfacción 3–5'
    ))

    # H2: 5 o más tickets vs 0–4 tickets.
    h2 = df.copy()
    h2['grupo_riesgo'] = h2['support_tickets'] >= 5
    results.append(evaluate_binary_hypothesis(
        h2, 'grupo_riesgo',
        'H2: 5+ tickets > 0–4 tickets'
    ))

    # H3: quintil inferior de gasto vs quintiles 2–5.
    h3 = df.dropna(subset=['total_spent']).copy()
    h3['spend_quintile'] = pd.qcut(
        h3['total_spent'], 5, labels=False, duplicates='drop'
    )
    h3['grupo_riesgo'] = h3['spend_quintile'] == 0
    results.append(evaluate_binary_hypothesis(
        h3, 'grupo_riesgo',
        'H3: Q1 total_spent > Q2–Q5'
    ))

    out = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}')
    print(out)

    # Validaciones automáticas del contraste esperado.
    assert (out['p_value_unilateral'] < 0.05).all()
    assert (out['risk_ratio'] > 1).all()
    assert (out['cramers_v'] > 0).all()

    print('\nHIPÓTESIS VALIDADAS ESTADÍSTICAMENTE')
    print('Nota: asociación estadística no implica causalidad.')


if __name__ == '__main__':
    main()
