import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style graphique lisible et propre
sns.set(style="whitegrid", palette="muted")


def histogrammes(df: pd.DataFrame, colonnes=None):
    if colonnes is None:
        colonnes = ["toxicite_prompt", "toxicite_reponse", "delta_t"]

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
        colonnes = ["toxicite_prompt", "toxicite_reponse"]
    
    plt.figure(figsize=(7, 5))
    # On filtre seulement sur les colonnes demandées présentes dans le df
    cols_presentes = [c for c in colonnes if c in df.columns]
    df_melt = df[cols_presentes].melt(var_name="type", value_name="toxicite")
    
    sns.boxplot(x="type", y="toxicite", data=df_melt)
    plt.title("Comparaison des toxicités")
    plt.tight_layout()
    plt.show()


def scatterplot(df, x, y):
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Relation entre {x} et {y}")
    plt.show()


def matrice_correlation(df: pd.DataFrame):
    """
    Matrice de corrélation entre les variables numériques.
    """
    colonnes = ["toxicite_prompt", "toxicite_continuation", "delta_t"]
    corr = df[colonnes].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()
    
def comparer_toxicites(df: pd.DataFrame):
    """
    Compare les 3 scores de toxicité : prompt, continuation, et réponse Llama.
    """
    colonnes = ["toxicite_prompt", "toxicite_continuation", "toxicite_reponse_llama"]
    cols_presentes = [c for c in colonnes if c in df.columns]

    # Boxplot comparatif
    plt.figure(figsize=(8, 5))
    df_melt = df[cols_presentes].melt(var_name="type", value_name="toxicite")
    sns.boxplot(x="type", y="toxicite", data=df_melt, palette="Set2")
    plt.title("Comparaison des 3 scores de toxicité")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    # Histogrammes superposés
    plt.figure(figsize=(8, 5))
    for col in cols_presentes:
        plt.hist(df[col], bins=30, alpha=0.5, label=col)
    plt.title("Distribution des 3 scores de toxicité")
    plt.xlabel("Score de toxicité")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.tight_layout()
    plt.show()

def generer_tableau_recap(df_eval):
    """
    Crée un tableau croisé dynamique pour comparer les toxicités
    par tranches de niveau du prompt.
    """
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
