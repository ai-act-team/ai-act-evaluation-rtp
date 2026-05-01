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
