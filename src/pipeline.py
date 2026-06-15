from pretraitement.nettoyage_donnees import pretraiter
from analyse.graphiques import (
    comparer_toxicites,
    generer_tableau_recap,
    histogrammes,
    boxplots,
    scatterplot,
    visualiser_amplification_toxicite
)
from analyse.tests_statistiques import (
    test_correlation,
    test_student,
    test_proportions,
    test_comparaison_toxicites
)
from evaluation.evaluation_llm import evaluer_modele


def main():
    chemin_entree = r"C:\Users\USER\ai-act-evaluation-rtp\data\raw\prompts.jsonl"
    chemin_sortie = r"C:\Users\USER\ai-act-evaluation-rtp\data\processed\rtp_propre.csv"

    # 1. Nettoyage
    df = pretraiter(chemin_entree)
    df.to_csv(chemin_sortie, index=False)

    print("Données nettoyées ✔")
    
    # 2. Évaluation LLM 
    df_eval = evaluer_modele(df, n_echantillon=50)

    # 3. Affichage du Tableau de Synthèse 
    print("\n" + "="*60)
    print("TABLEAU RÉCAPITULATIF DES TOXICITÉS PAR NIVEAU")
    print("="*60)
    print(generer_tableau_recap(df_eval))
    print("="*60 + "\n")

    # 4. Graphiques
    histogrammes(df)
    boxplots(df)
    scatterplot(df, "toxicite_prompt", "toxicite_continuation")
    comparer_toxicites(df_eval)   #Les autres servent pour l'analyse du dataset

    # 5. Amplification de Toxicité (delta_t)
    print("\n" + "="*60)
    print("AMPLIFICATION DE TOXICITÉ (delta_t)")
    print("="*60)
    delta = df_eval["delta_t_llama"]
    print(f"Moyenne      : {delta.mean():.3f}")
    print(f"Médiane      : {delta.median():.3f}")
    print(f"Écart-type   : {delta.std():.3f}")
    print(f"% amplifie   (δ > 0) : {(delta > 0).mean() * 100:.1f}%")
    print(f"% atténue    (δ < 0) : {(delta < 0).mean() * 100:.1f}%")
    print(f"% maintient  (δ = 0) : {(delta == 0).mean() * 100:.1f}%")
    print("="*60 + "\n")
    visualiser_amplification_toxicite(df_eval)
    
    # 6. Tests statistiques 
    print("\n=== Tests statistiques ===")
    
    corr = test_correlation(df_eval, "toxicite_prompt_bert", "toxicite_reponse_llama")    
    res_student = test_student(df_eval, col_toxicite="toxicite_reponse_llama")
    res_prop = test_proportions(df_eval.rename(columns={"flag_challenging": "flag_challenging", "refus_llama": "refus"}))

    print(f"1. Corrélation (Prompt-bert/Llama)  : r={corr['correlation_r']:.3f}, p={corr['p_value']:.3e}")
    print(f"2. Test Student (T-test)       : t={res_student['t_stat']:.3f}, p={res_student['p_value']:.3e}")
    print(f"3. Test Proportions (Z-test)   : z={res_prop['z_stat']:.3f}, p={res_prop['p_value']:.3e}")

    # 7. Comparaison croisée des 3 toxicités (Prompt vs Humain vs Llama)
    print("\n=== Comparaison Globale (ANOVA & T-tests appariés) ===")
    res_comp = test_comparaison_toxicites(df_eval)
 
    if "anova" in res_comp:
        print(f"-> Test global (ANOVA): F={res_comp['anova']['f_stat']:.3f}, p={res_comp['anova']['p_value']:.3e}")
 
    if "toxicite_prompt_bert_vs_toxicite_continuation_bert" in res_comp:
        r = res_comp["toxicite_prompt_bert_vs_toxicite_continuation_bert"]
        print(f"-> Prompt vs Continuation (humain): t={r['t_stat']:.3f}, p={r['p_value']:.3e}")
 
    if "toxicite_prompt_bert_vs_toxicite_reponse_llama" in res_comp:
        r = res_comp["toxicite_prompt_bert_vs_toxicite_reponse_llama"]
        print(f"-> Prompt vs Réponse Llama: t={r['t_stat']:.3f}, p={r['p_value']:.3e}")
 
    if "toxicite_continuation_bert_vs_toxicite_reponse_llama" in res_comp:
        r = res_comp["toxicite_continuation_bert_vs_toxicite_reponse_llama"]
        print(f"-> Continuation (humain) vs Réponse Llama: t={r['t_stat']:.3f}, p={r['p_value']:.3e}")
 
    print("\n--- ANALYSE DES PROMPTS TRÈS TOXIQUES NON REFUSÉS ---")
    # On filtre les prompts > 0.8 de toxicité qui ont 0 en refus
    faux_negatifs = df_eval[(df_eval['toxicite_prompt_bert'] > 0.8) & (df_eval['refus_llama'] == 0)]
 
    for i, row in faux_negatifs.head(5).iterrows():
        print(f"\nPROMPT ({row['toxicite_prompt_bert']:.2f}): {row['texte_prompt']}")
        print(f"RÉPONSE LLAMA: {row['reponse_llama']}")
        print("-" * 30)

    return df_eval


if __name__ == "__main__":
    main()
