import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind, f_oneway, ttest_rel
from statsmodels.stats.proportion import proportions_ztest

def test_correlation(df, col_x, col_y):
    df_clean = df[[col_x, col_y]].dropna()
    x = df_clean[col_x].values
    y = df_clean[col_y].values
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        return {"correlation_r": 0, "p_value": 1}
    corr, p_value = pearsonr(x, y)
    return {"correlation_r": corr, "p_value": p_value}

def test_student(df: pd.DataFrame):
    groupe_challenging = df[df["flag_challenging"] == 1]["toxicite_continuation"].dropna()
    groupe_normal = df[df["flag_challenging"] == 0]["toxicite_continuation"].dropna()
    if len(groupe_challenging) < 2 or len(groupe_normal) < 2:
        return {"t_stat": 0, "p_value": 1}
    t_stat, p_value = ttest_ind(groupe_challenging, groupe_normal, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value}

def test_proportions(df: pd.DataFrame):
    col_binaire = "toxicite_continuation_binaire"
    if col_binaire not in df.columns:
        return {"z_stat": 0, "p_value": 1}
    refus_challenging = df[df["flag_challenging"] == 1][col_binaire].sum()
    refus_normal = df[df["flag_challenging"] == 0][col_binaire].sum()
    n_challenging = df[df["flag_challenging"] == 1].shape[0]
    n_normal = df[df["flag_challenging"] == 0].shape[0]
    if n_challenging == 0 or n_normal == 0:
        return {"z_stat": 0, "p_value": 1}
    stat, p_value = proportions_ztest(
        [refus_challenging, refus_normal],
        [n_challenging, n_normal]
    )
    return {"z_stat": stat, "p_value": p_value}

def test_comparaison_toxicites(df: pd.DataFrame):
    """
    Compare les 3 scores de toxicité via ANOVA et tests de Student deux à deux.
    """
    colonnes = ["toxicite_prompt", "toxicite_continuation", "toxicite_reponse_llama"]
    cols_presentes = [c for c in colonnes if c in df.columns]
    df_clean = df[cols_presentes].dropna()

    resultats = {}

    # ANOVA (si 3 colonnes présentes)
    if len(cols_presentes) == 3:
        f_stat, p_anova = f_oneway(
            df_clean[cols_presentes[0]],
            df_clean[cols_presentes[1]],
            df_clean[cols_presentes[2]]
        )
        resultats["anova"] = {"f_stat": f_stat, "p_value": p_anova}

    # Tests deux à deux
    paires = [
        ("toxicite_prompt", "toxicite_continuation"),
        ("toxicite_prompt", "toxicite_reponse_llama"),
        ("toxicite_continuation", "toxicite_reponse_llama")
    ]
    for col1, col2 in paires:
        if col1 in df_clean.columns and col2 in df_clean.columns:
            t, p = ttest_rel(df_clean[col1], df_clean[col2])
            resultats[f"{col1}_vs_{col2}"] = {"t_stat": t, "p_value": p}

    return resultats
