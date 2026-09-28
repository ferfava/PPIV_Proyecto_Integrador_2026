from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path('data/raw/sales_marketing_customer_raw.csv')
FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare(path=DATA_PATH):
    df = pd.read_csv(path)
    eda = df.copy()

    eda['signup_date'] = pd.to_datetime(eda['signup_date'], errors='coerce')
    eda['last_purchase_date'] = pd.to_datetime(eda['last_purchase_date'], errors='coerce')
    eda.loc[eda['age'] < 18, 'age'] = np.nan

    eda['temporal_inconsistency'] = (
        eda['last_purchase_date'] < eda['signup_date']
    ).astype(int)
    eda['coupon_used_flag'] = eda['coupon_code'].notna().astype(int)
    eda['gender_clean'] = eda['gender'].fillna('Unknown')

    reference_date = eda['last_purchase_date'].max() + pd.Timedelta(days=1)
    eda['customer_tenure_days'] = (reference_date - eda['signup_date']).dt.days
    eda['recency_days'] = (reference_date - eda['last_purchase_date']).dt.days

    return df, eda, reference_date


def save_bar(series, title, ylabel, xlabel, filename, figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    series.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=160)
    plt.close(fig)


def main():
    df, eda, reference_date = load_and_prepare()

    print(f'Dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas')
    print('Fecha de referencia:', reference_date.date())
    print(f'Tasa global de churn: {eda.churn.mean():.2%}')

    # 1. Target
    target = eda['churn'].value_counts().sort_index()
    save_bar(
        target.rename(index={0: 'No churn', 1: 'Churn'}),
        'Distribución de churn', 'Clientes', '', '01_churn_distribution.png', (6, 4)
    )

    # 2. Satisfacción
    sat = (
        eda.assign(
            satisfaction_group=eda['satisfaction_score'].fillna(-1).map(
                {-1: 'Sin dato', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
            )
        )
        .groupby('satisfaction_group', observed=True)['churn']
        .agg(['mean', 'count'])
    )
    sat['churn_pct'] = sat['mean'] * 100
    sat = sat.reindex(['1', '2', '3', '4', '5', 'Sin dato'])
    save_bar(
        sat['churn_pct'], 'Tasa de churn por satisfacción', 'Churn (%)',
        'Satisfaction score', '02_churn_by_satisfaction.png'
    )

    # 3. Tickets de soporte
    eda['support_group'] = pd.cut(
        eda['support_tickets'],
        bins=[-1, 0, 2, 4, np.inf],
        labels=['0', '1–2', '3–4', '5+']
    )
    support = eda.groupby('support_group', observed=True)['churn'].agg(['mean', 'count'])
    support['churn_pct'] = support['mean'] * 100
    save_bar(
        support['churn_pct'], 'Tasa de churn según tickets de soporte', 'Churn (%)',
        'Tickets de soporte', '03_churn_by_support.png'
    )

    # 4. Quintiles de gasto
    spent = eda.dropna(subset=['total_spent']).copy()
    spent['spend_quintile'] = pd.qcut(
        spent['total_spent'], 5,
        labels=['Q1 menor', 'Q2', 'Q3', 'Q4', 'Q5 mayor'],
        duplicates='drop'
    )
    spend = spent.groupby('spend_quintile', observed=True)['churn'].agg(['mean', 'count'])
    spend['churn_pct'] = spend['mean'] * 100
    save_bar(
        spend['churn_pct'], 'Tasa de churn por quintil de gasto', 'Churn (%)',
        'Quintil de total_spent', '04_churn_by_spend_quintile.png', (8, 4)
    )

    # 5. Correlaciones
    corr = (
        eda.select_dtypes(include=np.number)
        .corr(numeric_only=True)['churn']
        .drop('churn')
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    corr.head(12).sort_values().plot(kind='barh', ax=ax)
    ax.set_title('Variables numéricas con mayor correlación absoluta con churn')
    ax.set_xlabel('Correlación de Pearson')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '05_correlations_churn.png', dpi=160)
    plt.close(fig)

    # 6. Interacción satisfacción + soporte
    eda['satisfaction_band'] = np.select(
        [
            eda['satisfaction_score'].isin([1, 2]),
            eda['satisfaction_score'].isin([3, 4, 5])
        ],
        ['Baja (1–2)', 'Media/alta (3–5)'],
        default='Sin dato'
    )
    eda['support_band'] = np.where(eda['support_tickets'] >= 5, '5+ tickets', '0–4 tickets')
    interaction = (
        eda.groupby(['satisfaction_band', 'support_band'], observed=True)['churn']
        .agg(['mean', 'count'])
    )
    interaction['churn_pct'] = interaction['mean'] * 100

    print('\nSatisfacción:')
    print(sat[['count', 'churn_pct']].round(2))
    print('\nSoporte:')
    print(support[['count', 'churn_pct']].round(2))
    print('\nGasto:')
    print(spend[['count', 'churn_pct']].round(2))
    print('\nInteracción satisfacción + soporte:')
    print(interaction[['count', 'churn_pct']].round(2))

    # Controles de integridad
    assert len(eda) == len(df)
    assert eda['customer_id'].is_unique
    assert set(eda['churn'].dropna().unique()).issubset({0, 1})
    assert eda['customer_tenure_days'].notna().all()
    assert eda['recency_days'].notna().all()

    print('\nEDA VALIDADO')
    print(f'Filas analizadas: {len(eda):,}')
    print(f'Figuras generadas en: {FIG_DIR}')


if __name__ == '__main__':
    main()
