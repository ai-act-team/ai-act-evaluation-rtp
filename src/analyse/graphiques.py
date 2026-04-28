import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style graphique lisible et propre
sns.set(style="whitegrid", palette="muted")


def histogrammes(df: pd.DataFrame):
    """
    Affiche les histogrammes des toxicités et de delta_t.
    """
    colonnes = ["toxicite_prompt", "toxicite_reponse", "delta_t"]

    for col in colonnes:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=30, color="steelblue", edgecolor="black")
        plt.title(f"Distribution de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.show()


def boxplots(df: pd.DataFrame):
    """
    Compare les distributions de toxicité via boxplots.
    """
    plt.figure(figsize=(7, 5))
    df_melt = df[["toxicite_prompt", "toxicite_reponse"]].melt(
        var_name="type", value_name="toxicite"
    )
    sns.boxplot(x="type", y="toxicite", data=df_melt)
    plt.title("Comparaison des toxicités (prompt vs réponse)")
    plt.tight_layout()
    plt.show()


def scatterplot(df):
    plt.figure()
    plt.scatter(df["toxicite_prompt"], df["toxicite_reponse"])
    plt.xlabel("toxicite_prompt")
    plt.ylabel("toxicite_reponse")
    plt.show()


def matrice_correlation(df: pd.DataFrame):
    """
    Matrice de corrélation entre les variables numériques.
    """
    colonnes = ["toxicite_prompt", "toxicite_reponse", "delta_t"]
    corr = df[colonnes].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()
