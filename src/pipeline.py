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
    """
    Pipeline complet :
    - nettoyage des données
    - visualisations
    - tests statistiques
    """
    # Fichiers d'entrée et de sortie
    chemin_entree = "data/raw/prompts.jsonl"
    chemin_sortie = "data/processed/rtp_propre.csv"

    # 1. Prétraitement des données
    df = pretraiter(chemin_entree)
    df.to_csv(chemin_sortie, index=False)
    print(f"Données nettoyées enregistrées dans {chemin_sortie}")

    # 2. Évaluation du Modèle (Llama 3.1 8B)
    print("=== ÉTAPE 2 : Évaluation LLM (AI Act) ===")
    df_evalue = evaluer_modele(df, n_echantillon=50)
    df_evalue.to_csv(chemin_sortie_evalue, index=False)
    print(f"Résultats de l'évaluation enregistrés dans {chemin_sortie_evalue}\n")

    # 3. Visualisations
    print("\nGénération des graphiques...")
    histogrammes(df)
    boxplots(df)
    scatterplot(df)
    matrice_correlation(df)

    # 4. Tests statistiques
    print("\n=== Résultats des tests statistiques ===")

    corr = test_correlation(df)
    print(f"Corrélation toxicité prompt / réponse : r = {corr['correlation_r']:.3f}, p = {corr['p_value']:.3e}")

    stud = test_student(df)
    print(f"Test de Student (challenging vs non-challenging) : t = {stud['t_stat']:.3f}, p = {stud['p_value']:.3e}")

    prop = test_proportions(df)
    print(f"Test de proportions (refus) : z = {prop['z_stat']:.3f}, p = {prop['p_value']:.3e}")

    # --- Analyse rapide des résultats Llama ---
    print("\n=== Analyse rapide Llama 3.1 8B ===")
    tox_moyenne = df_evalue["toxicite_reponse_llama"].mean()
    taux_refus = df_evalue["refus_llama"].mean() * 100
    print(f"Toxicité moyenne des réponses Llama : {tox_moyenne:.3f}")
    print(f"Taux de refus global : {taux_refus:.1f}%")


if __name__ == "__main__":
    main()
