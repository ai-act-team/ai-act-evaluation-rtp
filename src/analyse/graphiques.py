import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style graphique lisible et propre
sns.set(style="whitegrid", palette="muted")

# Graphes pour l'analyse des variables du dataset et des résultats du test
def histogrammes(df: pd.DataFrame, colonnes=None):
    if colonnes is None:
        colonnes = ["toxicite_prompt", "toxicite_continuation", "delta_t"]

    for col in colonnes:
        if col in df.columns: # Sécurité : vérifie si la colonne existe
            plt.figure(figsize=(6, 4))
            plt.hist(df[col], bins=30, color="steelblue", edgecolor="black")
            plt.title(f"Distribution de {col}")
            plt.xlabel(col)
            plt.ylabel("Fréquence")
            plt.tight_layout()
            plt.show()


def boxplots(df: pd.DataFrame, colonnes=None):
    if colonnes is None:
        colonnes = ["toxicite_prompt", "toxicite_continuation"]
    
    plt.figure(figsize=(7, 5))
    # On filtre seulement sur les colonnes demandées présentes dans le df
    cols_presentes = [c for c in colonnes if c in df.columns]
    df_melt = df[cols_presentes].melt(var_name="type", value_name="toxicite")
    
    sns.boxplot(x="type", y="toxicite", data=df_melt)
    plt.title("Comparaison des toxicités")
    plt.tight_layout()
    plt.show()

# Graphes pour l'analyse des variables du dataset seulement
def scatterplot(df: pd.DataFrame):
    """
    Scatterplot : toxicité du prompt vs toxicité de la réponse.
    """
    plt.figure(figsize=(6, 5))

    # D'abord les bleus
    sns.scatterplot(
        data=df[df["flag_challenging"] == 0],
        x="toxicite_prompt",
        y="toxicite_continuation",
        color="C0",
        alpha=0.6,
        label="0"
    )

    # Ensuite les oranges
    sns.scatterplot(
        data=df[df["flag_challenging"] == 1],
        x="toxicite_prompt",
        y="toxicite_continuation",
        color="C1",
        alpha=0.6,
        label="1"
    )
    plt.title("Toxicité prompt vs toxicité continuation")
    plt.legend(title="flag_challenging")
    plt.tight_layout()
    plt.show()


# Graphes pour l'analyse des résultats seulement
def comparer_toxicites(df: pd.DataFrame):
    """
    Compare les 2 scores de toxicité : prompt_bert et réponse Llama.
    """
    colonnes = ["toxicite_prompt", "toxicite_continuation", "toxicite_reponse_llama"]
    cols_presentes = [c for c in colonnes if c in df.columns]

    labels_affichage = {
        "toxicite_prompt":       "Prompt",
        "toxicite_continuation": "Continuation humaine",
        "toxicite_reponse_llama":     "Réponse Llama",
    }
    
    # Boxplot comparatif
    plt.figure(figsize=(8, 5))
    df_melt = df[cols_presentes].rename(columns=labels_affichage).melt(var_name="type", value_name="toxicite")
    sns.boxplot(x="type", y="toxicite", data=df_melt, palette="Set2")
    plt.title("Comparaison des 3 scores de toxicité")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()

    # Histogrammes superposés
    plt.figure(figsize=(8, 5))
    for col in cols_presentes:
        plt.hist(df[col], bins=30, alpha=0.5, label=labels_affichage.get(col, col))
    plt.title("Distribution des 3 scores de toxicité")
    plt.xlabel("Score de toxicité")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualiser_amplification_toxicite(df: pd.DataFrame):
    """
    Visualise le score delta_t_llama (amplification de toxicité) :
    - Distribution du delta_t (histogramme centré sur 0)
    - Boxplot du delta_t par type de prompt (challenging vs normal)
    """
    if "delta_t_llama" not in df.columns:
        print("Colonne delta_t_llama absente, visualisation impossible.")
        return
 
    delta = df["delta_t_llama"]
 
    # 1. Distribution du delta_t
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=30, color="steelblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="δ = 0 (pas d'amplification)")
    plt.axvline(delta.mean(), color="orange", linestyle="-", linewidth=1.5, label=f"Moyenne = {delta.mean():.3f}")
    plt.title("Distribution du score d'amplification de toxicité (delta_t)")
    plt.xlabel("delta_t  (toxicité réponse − toxicité prompt)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.tight_layout()
    plt.show()


def generer_tableau_recap(df_eval):

    #Crée un tableau pour comparer les toxicités des prompts et réponses.

    # 1. Créer des catégories de toxicité pour le prompt
    bins = [0, 0.33, 0.66, 1.0]
    labels = ['Faible (0-33%)', 'Moyen (33-66%)', 'Élevé (66-100%)']

    # Copie pour ne pas modifier le dataframe original
    df_temp = df_eval.copy()
    df_temp['Niveau Prompt'] = pd.cut(df_temp['toxicite_prompt'], bins=bins, labels=labels)

    # 2. Grouper et calculer les moyennes
    tableau = df_temp.groupby('Niveau Prompt', observed=True).agg({
        'toxicite_prompt': 'mean',
        'toxicite_continuation': 'mean',
        'toxicite_reponse_llama': 'mean',
        'refus_llama': 'sum' # On compte le nombre total de refus par niveau
    }).rename(columns={
        'toxicite_prompt': 'Moy. Prompt',
        'toxicite_continuation': 'Moy. Humain',
        'toxicite_reponse_llama': 'Moy. Llama',
        'refus_llama': 'Nb Refus'
    })

    return tableau
