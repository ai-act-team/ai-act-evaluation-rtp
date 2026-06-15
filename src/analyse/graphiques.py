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
    colonnes = ["toxicite_prompt", "toxicite_reponse", "delta_t"]
    corr = df[colonnes].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()

# Graphes pour l'analyse des résultats seulement
def comparer_toxicites(df: pd.DataFrame):
    """
    Compare les 2 scores de toxicité : prompt_bert et réponse Llama.
    """
    colonnes = ["toxicite_prompt_bert", "toxicite_continuation_bert", "toxicite_reponse_llama"]
    cols_presentes = [c for c in colonnes if c in df.columns]

    labels_affichage = {
        "toxicite_prompt_bert":       "Prompt (toxic-bert)",
        "toxicite_continuation_bert": "Continuation humaine (toxic-bert)",
        "toxicite_reponse_llama":     "Réponse Llama (toxic-bert)",
    }
    
    # Boxplot comparatif
    plt.figure(figsize=(8, 5))
    df_melt = df[cols_presentes].rename(columns=labels_affichage).melt(var_name="type", value_name="toxicite")
    sns.boxplot(x="type", y="toxicite", data=df_melt, palette="Set2")
    plt.title("Comparaison des 3 scores de toxicité (toxic-bert)")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()

    # Histogrammes superposés
    plt.figure(figsize=(8, 5))
    for col in cols_presentes:
        plt.hist(df[col], bins=30, alpha=0.5, label=labels_affichage.get(col, col))
    plt.title("Distribution des 3 scores de toxicité (toxic-bert)")
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
    plt.xlabel("delta_t  (toxicité réponse − toxicité prompt, toxic-bert)")
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
    
    # On catégorise sur toxicite_prompt_bert si disponible, sinon on replie sur toxicite_prompt
    col_prompt = "toxicite_prompt_bert" if "toxicite_prompt_bert" in df_temp.columns else "toxicite_prompt"
    df_temp['Niveau Prompt'] = pd.cut(df_temp[col_prompt], bins=bins, labels=labels)
 
    agg_dict  = {col_prompt: 'mean', 'refus_llama': 'sum'}
    rename_dict = {col_prompt: 'Moy. Prompt (bert)', 'refus_llama': 'Nb Refus'}
 
    if 'toxicite_continuation_bert' in df_temp.columns:
        agg_dict['toxicite_continuation_bert'] = 'mean'
        rename_dict['toxicite_continuation_bert'] = 'Moy. Continuation (bert)'
 
    if 'toxicite_reponse_llama' in df_temp.columns:
        agg_dict['toxicite_reponse_llama'] = 'mean'
        rename_dict['toxicite_reponse_llama'] = 'Moy. Llama (bert)'
 
    if 'delta_t_llama' in df_temp.columns:
        agg_dict['delta_t_llama'] = 'mean'
        rename_dict['delta_t_llama'] = 'Moy. delta_t'
 
    tableau = df_temp.groupby('Niveau Prompt', observed=True).agg(agg_dict).rename(columns=rename_dict)
    return tableau
