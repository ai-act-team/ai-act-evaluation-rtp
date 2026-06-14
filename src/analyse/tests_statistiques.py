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

def test_student(df: pd.DataFrame, col_toxicite="toxicite_continuation"):
    groupe_challenging = df[df["flag_challenging"] == 1][col_toxicite].dropna()
    groupe_normal = df[df["flag_challenging"] == 0][col_toxicite].dropna()
    
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
    Compare les scores de toxicité toxic-bert entre prompt et réponse Llama.
    Comparaison principale : toxicite_prompt_bert vs toxicite_reponse_llama (même référentiel).
    Comparaison secondaire optionnelle : toxicite_prompt (dataset) vs toxicite_reponse_llama.
    """
    colonnes_bert = ["toxicite_prompt_bert", "toxicite_reponse_llama"]
    cols_bert_presentes = [c for c in colonnes_bert if c in df.columns]
    df_clean = df[cols_bert_presentes].dropna()
    
    resultats = {}

    # ANOVA ( si on a plusieurs colonnes)
    if len(cols_bert_presentes) >= 2:
        f_stat, p_anova = f_oneway(*[df_clean[c] for c in cols_bert_presentes])
        resultats["anova"] = {"f_stat": f_stat, "p_value": p_anova}
 
    # Test principal : prompt-bert vs réponse-bert
    if "toxicite_prompt_bert" in df_clean.columns and "toxicite_reponse_llama" in df_clean.columns:
        t, p = ttest_rel(df_clean["toxicite_prompt_bert"], df_clean["toxicite_reponse_llama"])
        resultats["toxicite_prompt_bert_vs_toxicite_reponse_llama"] = {"t_stat": t, "p_value": p}
 
    # Test secondaire (pour garder la traçabilité avec l'ancien score dataset)
    if "toxicite_prompt" in df.columns and "toxicite_reponse_llama" in df.columns:
        df_sec = df[["toxicite_prompt", "toxicite_reponse_llama"]].dropna()
        t2, p2 = ttest_rel(df_sec["toxicite_prompt"], df_sec["toxicite_reponse_llama"])
        resultats["toxicite_prompt_vs_toxicite_reponse_llama"] = {"t_stat": t2, "p_value": p2}

    return resultats
