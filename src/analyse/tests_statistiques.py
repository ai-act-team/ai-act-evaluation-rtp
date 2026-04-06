import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.proportion import proportions_ztest


def test_correlation(df: pd.DataFrame):
    """
    Test de corrélation entre la toxicité du prompt et la toxicité de la réponse.
    Retourne le coefficient de corrélation r et la p-valeur.
    """
    r, p = pearsonr(df["toxicite_prompt"], df["toxicite_reponse"])
    return {"correlation_r": r, "p_value": p}


def test_student(df: pd.DataFrame):
    """
    Test de Student pour comparer la toxicité des réponses
    entre prompts challenging et non-challenging.
    """
    groupe_challenging = df[df["flag_challenging"] == 1]["toxicite_reponse"]
    groupe_normal = df[df["flag_challenging"] == 0]["toxicite_reponse"]

    t_stat, p_value = ttest_ind(groupe_challenging, groupe_normal, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value}


def test_proportions(df: pd.DataFrame):
    """
    Test de comparaison de proportions :
    compare le taux de refus entre prompts challenging et non-challenging.
    """
    refus_challenging = df[df["flag_challenging"] == 1]["refus"].sum()
    refus_normal = df[df["flag_challenging"] == 0]["refus"].sum()

    n_challenging = df[df["flag_challenging"] == 1].shape[0]
    n_normal = df[df["flag_challenging"] == 0].shape[0]

    stat, p_value = proportions_ztest([refus_challenging, refus_normal],
                                      [n_challenging, n_normal])

    return {"z_stat": stat, "p_value": p_value}
