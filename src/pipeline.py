from pretraitement.nettoyage_donnees import pretraiter
from analyse.graphiques import (
    histogrammes,
    boxplots,
    scatterplot,
    matrice_correlation
)
from analyse.tests_statistiques import (
    test_correlation,
    test_student,
    test_proportions
)
from evaluation.evaluation_llm import evaluer_modele


def main():
    chemin_entree = r"C:\Users\USER\ai-act-evaluation-rtp\data\raw\prompts.jsonl"
    chemin_sortie = r"C:\Users\USER\ai-act-evaluation-rtp\data\processed\rtp_propre.csv"

    # 1. Nettoyage
    df = pretraiter(chemin_entree)
    df.to_csv(chemin_sortie, index=False)

    print("Données nettoyées ✔")

    # 2. Évaluation LLM (🔥 AJOUT)
    df_eval = evaluer_modele(df, n_echantillon=5)

    # 3. Graphiques
    histogrammes(df_eval, colonnes=["toxicite_prompt", "toxicite_continuation", "toxicite_reponse_llama"])    
    boxplots(df_eval, colonnes=["toxicite_continuation", "toxicite_reponse_llama"])    
    scatterplot(df_eval, x="toxicite_prompt", y="toxicite_continuation")    
    scatterplot(df_eval, x="toxicite_prompt", y="toxicite_reponse_llama")
    matrice_correlation(df_eval)
    
    # 4. Tests statistiques (⚠ utiliser réponses LLM)
    print("\n=== Tests statistiques ===")
    
    corr = test_correlation(df_eval, "toxicite_prompt", "toxicite_reponse_llama")    
    res_student = test_student(df_eval.rename(columns={"toxicite_reponse_llama": "toxicite_reponse", "flag_challenging": "flag_challenging"}))
    res_prop = test_proportions(df_eval.rename(columns={"flag_challenging": "flag_challenging", "refus_llama": "refus"}))

    print(f"1. Corrélation (Prompt/Llama)  : r={corr['correlation_r']:.3f}, p={corr['p_value']:.3e}")
    print(f"2. Test Student (T-test)       : t={res_student['t_stat']:.3f}, p={res_student['p_value']:.3e}")
    print(f"3. Test Proportions (Z-test)   : z={res_prop['z_stat']:.3f}, p={res_prop['p_value']:.3e}")

    return df_eval


if __name__ == "__main__":
    main()
