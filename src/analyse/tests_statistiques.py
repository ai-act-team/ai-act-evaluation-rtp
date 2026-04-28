# src/analyse/tests_statistiques.py
"""
Tests statistiques pour l'analyse de toxicité
Correlations, t-tests, tests de proportions
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def test_correlation(df, col_x, col_y):
    """
    Teste la corrélation entre deux colonnes
    
    Args:
        df: DataFrame pandas
        col_x: Nom de la colonne X
        col_y: Nom de la colonne Y
        
    Returns:
        dict avec résultats de corrélation
    """
    
    # Nettoyer les données
    df_clean = df[[col_x, col_y]].dropna()
    x = df_clean[col_x].values  # ✅ CORRECTION: Convertir en numpy array
    y = df_clean[col_y].values  # ✅ CORRECTION: Convertir en numpy array
    
    # Vérifier les valeurs uniques
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
        raise ValueError("Not enough unique values for correlation test")
    
    # Pearson correlation
    corr_pearson, p_pearson = stats.pearsonr(x, y)
    
    # Spearman correlation
    corr_spearman, p_spearman = stats.spearmanr(x, y)
    
    return {
        'pearson_r': corr_pearson,
        'pearson_p': p_pearson,
        'spearman_rho': corr_spearman,
        'spearman_p': p_spearman,
        'n_samples': len(df_clean),
        'pearson_significant': p_pearson < 0.05,
        'spearman_significant': p_spearman < 0.05
    }


def test_student(df, col_groupby, col_value):
    """
    T-test de Student entre deux groupes
    
    Args:
        df: DataFrame pandas
        col_groupby: Colonne contenant les groupes
        col_value: Colonne contenant les valeurs à comparer
        
    Returns:
        dict avec résultats du t-test
    """
    
    groups = df[col_groupby].unique()
    
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(groups)}")
    
    # ✅ CORRECTION: Utiliser .values pour éviter les Series
    group1_data = df[df[col_groupby] == groups[0]][col_value].values
    group2_data = df[df[col_groupby] == groups[1]][col_value].values
    
    # T-test
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'group1_mean': np.mean(group1_data),
        'group2_mean': np.mean(group2_data),
        'group1_std': np.std(group1_data),
        'group2_std': np.std(group2_data),
        'group1_n': len(group1_data),
        'group2_n': len(group2_data),
        'significant': p_value < 0.05
    }


def test_proportions(df, col_groupby, col_binary, success_value=1):
    """
    Test de proportions (chi-square)
    
    Args:
        df: DataFrame pandas
        col_groupby: Colonne des groupes
        col_binary: Colonne binaire (0/1)
        success_value: Valeur à considérer comme succès
        
    Returns:
        dict avec résultats du chi-square
    """
    
    # ✅ CORRECTION: Utiliser .values pour éviter les Series ambiguës
    binary_col = (df[col_binary] == success_value).values
    
    # Créer une table de contingence
    contingency_table = pd.crosstab(
        df[col_groupby],
        binary_col
    )
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'contingency_table': contingency_table.to_dict(),
        'n_samples': len(df),
        'significant': p_value < 0.05
    }


def compare_distributions(values1, values2, test_type='ks'):
    """
    Compare deux distributions
    
    Args:
        values1: Premier ensemble de valeurs (array-like)
        values2: Deuxième ensemble de valeurs (array-like)
        test_type: Type de test ('ks', 'mann_whitney')
        
    Returns:
        dict avec résultats du test
    """
    
    # Convertir en arrays numpy si nécessaire
    values1 = np.asarray(values1)
    values2 = np.asarray(values2)
    
    if test_type == 'ks':
        # Kolmogorov-Smirnov test
        stat, p_value = stats.ks_2samp(values1, values2)
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    elif test_type == 'mann_whitney':
        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(values1, values2)
        return {
            'test': 'Mann-Whitney U',
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def compare_toxicity_distributions(df_prompts, df_responses, col_toxicity='toxicity'):
    """
    Compare les distributions de toxicité entre prompts et réponses
    
    Args:
        df_prompts: DataFrame des prompts
        df_responses: DataFrame des réponses
        col_toxicity: Nom de la colonne de toxicité
        
    Returns:
        dict avec résultats des comparaisons
    """
    
    try:
        # ✅ CORRECTION: Convertir en arrays numpy
        prompt_tox = df_prompts[col_toxicity].dropna().values
        response_tox = df_responses[col_toxicity].dropna().values
        
        if len(prompt_tox) == 0 or len(response_tox) == 0:
            raise ValueError("Empty toxicity columns")
        
        # KS test
        ks_stat, ks_p = stats.ks_2samp(prompt_tox, response_tox)
        
        # Mann-Whitney test
        mw_stat, mw_p = stats.mannwhitneyu(prompt_tox, response_tox)
        
        return {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'mann_whitney_u': mw_stat,
            'mann_whitney_p': mw_p,
            'prompt_mean': np.mean(prompt_tox),
            'response_mean': np.mean(response_tox),
            'prompt_median': np.median(prompt_tox),
            'response_median': np.median(response_tox),
            'prompt_std': np.std(prompt_tox),
            'response_std': np.std(response_tox),
            'ks_significant': ks_p < 0.05,
            'mann_whitney_significant': mw_p < 0.05
        }
    
    except Exception as e:
        print(f"Error in compare_toxicity_distributions: {e}")
        raise


def print_results(results, test_name):
    """Affiche les résultats de façon lisible"""
    print(f"\n{'='*60}")
    print(f"Résultats - {test_name}")
    print(f"{'='*60}")
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        elif isinstance(value, bool):
            print(f"{key}: {'Oui' if value else 'Non'}")
        else:
            print(f"{key}: {value}")


# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    print("="*60)
    print("TESTS STATISTIQUES")
    print("="*60)
    
    # Créer données de test
    np.random.seed(42)
    df_test = pd.DataFrame({
        'toxicity_prompt': np.random.uniform(0, 1, 100),
        'toxicity_response': np.random.uniform(0, 0.8, 100),
        'group': np.random.choice(['A', 'B'], 100),
        'is_refusal': np.random.choice([0, 1], 100)
    })
    
    # Test corrélation
    print("\n[1] Test de Corrélation")
    try:
        corr_results = test_correlation(
            df_test, 'toxicity_prompt', 'toxicity_response'
        )
        print_results(corr_results, "Corrélation Toxicité")
    except Exception as e:
        print(f"Erreur: {e}")
    
    # Test proportions
    print("\n[2] Test des Proportions")
    try:
        prop_results = test_proportions(
            df_test, 'group', 'is_refusal'
        )
        print_results(prop_results, "Test des Proportions")
    except Exception as e:
        print(f"Erreur: {e}")
    
    # Test comparaison distributions
    print("\n[3] Test Comparaison des Distributions")
    try:
        dist_results = compare_toxicity_distributions(
            df_test, df_test
        )
        print_results(dist_results, "Comparaison Distributions")
    except Exception as e:
        print(f"Erreur: {e}")
    
    print("\n" + "="*60)
