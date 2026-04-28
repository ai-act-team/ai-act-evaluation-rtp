import pandas as pd
from scipy.stats import pearsonr, ttest_ind
from statsmodels.stats.proportion import proportions_ztest


def test_correlation(df: pd.DataFrame):
    x = df["toxicite_prompt"].dropna()
    y = df["toxicite_reponse"].dropna()

    if len(x) < 2 or len(y) < 2 or x.nunique() <= 1 or y.nunique() <= 1:
        return {"correlation_r": 0, "p_value": 1}

    r, p = pearsonr(x, y)
    return {"correlation_r": r, "p_value": p}


def test_student(df: pd.DataFrame):
    groupe_challenging = df[df["flag_challenging"] == 1]["toxicite_reponse"].dropna()
    groupe_normal = df[df["flag_challenging"] == 0]["toxicite_reponse"].dropna()

    if len(groupe_challenging) < 2 or len(groupe_normal) < 2:
        return {"t_stat": 0, "p_value": 1}

    t_stat, p_value = ttest_ind(groupe_challenging, groupe_normal, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value}

def test_proportions(df: pd.DataFrame):
    refus_challenging = df[df["flag_challenging"] == 1]["refus"].sum()
    refus_normal = df[df["flag_challenging"] == 0]["refus"].sum()

    n_challenging = df[df["flag_challenging"] == 1].shape[0]
    n_normal = df[df["flag_challenging"] == 0].shape[0]

    if n_challenging == 0 or n_normal == 0:
        return {"z_stat": 0, "p_value": 1}

    stat, p_value = proportions_ztest(
        [refus_challenging, refus_normal],
        [n_challenging, n_normal]
    )

    return {"z_stat": stat, "p_value": p_value}
