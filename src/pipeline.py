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
    chemin_entree = "data/raw/prompts.jsonl"
    chemin_sortie = "data/processed/rtp_propre.csv"

    # 1. Nettoyage
    df = pretraiter(chemin_entree)
    df.to_csv(chemin_sortie, index=False)

    print("Données nettoyées ✔")

    # 2. Évaluation LLM (🔥 AJOUT)
    df_eval = evaluer_modele(df, n_echantillon=50)
    df_eval["toxicite_reponse"] = df_eval["toxicite_reponse_llama"]

    # 3. Graphiques
    histogrammes(df_eval)
    boxplots(df_eval)
    scatterplot(df_eval)
    matrice_correlation(df_eval)

    # 4. Tests statistiques (⚠ utiliser réponses LLM)
    print("\n=== Tests statistiques ===")

    corr = test_correlation(df_eval.rename(columns={
        "toxicite_reponse_llama": "toxicite_reponse"
    }))

    print(f"Corrélation : r={corr['correlation_r']:.3f}, p={corr['p_value']:.3e}")

    return df_eval


if __name__ == "__main__":
    main()
