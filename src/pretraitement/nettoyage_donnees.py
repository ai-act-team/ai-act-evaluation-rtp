import pandas as pd
import json

def charger_donnees(chemin):
    """
    Charge les données depuis un fichier JSONL ou CSV de manière plus robuste.
    """
    if chemin.endswith(".jsonl"):
        data = []
        with open(chemin, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Erreur de décodage JSON à la ligne {line_num+1}: {line.strip()} - {e}")
                    # Vous pouvez choisir de gérer les lignes problématiques différemment, par ex. les ignorer ou les logger.
        df = pd.DataFrame(data)
    elif chemin.endswith(".csv"):
        df = pd.read_csv(chemin)
    return df

# Définir les fonctions restantes nécessaires pour `pipeline.py` si elles ne sont pas déjà définies
# Ceci est une version simplifiée, vous devrez intégrer cela dans le fichier existant.

def renommer_colonnes(df):
    # Votre logique pour renommer les colonnes ici
    # Exemple : Renommer 'prompt.toxicity' en 'toxicite_prompt' si la colonne existe après l'aplatissement
    if 'prompt' in df.columns and 'toxicity' in df['prompt'][0]:
        df['toxicite_prompt'] = df['prompt'].apply(lambda x: x.get('toxicity'))
    if 'continuation' in df.columns and 'toxicity' in df['continuation'][0]:
        df['toxicite_continuation'] = df['continuation'].apply(lambda x: x.get('toxicity'))
    return df

def nettoyer_texte(df):
    # Votre logique de nettoyage de texte ici
    return df

def pretraiter(chemin_entree):
    df = charger_donnees(chemin_entree)
    df = renommer_colonnes(df)
    df = nettoyer_texte(df)
    return df



def nettoyer_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les toxicités en valeurs numériques bornées entre 0 et 1.
    """
    for col in ["toxicite_prompt", "toxicite_reponse"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 1)
    return df


def nettoyer_binaires(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes binaires en 0/1.
    """
    for col in ["refus", "flag_challenging"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)
    return df


def supprimer_manquants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes contenant des valeurs critiques manquantes.
    """
    return df.dropna(subset=["texte_prompt", "toxicite_prompt", "toxicite_reponse"])


def creer_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable delta_t = toxicité réponse – toxicité prompt.
    """
    df["delta_t"] = df["toxicite_reponse"] - df["toxicite_prompt"]
    return df


def pretraiter(chemin: str) -> pd.DataFrame:
    """
    Pipeline complet de nettoyage.
    """
    df = charger_donnees(chemin)
    df = renommer_colonnes(df)
    df = nettoyer_texte(df)
    df = nettoyer_numeriques(df)
    df = nettoyer_binaires(df)
    df = supprimer_manquants(df)
    df = creer_variables(df)
    return df


if __name__ == "__main__":
    # Chemin du fichier brut (relatif au projet)
    entree = "data/raw/prompts.jsonl"

    # Chemin du fichier nettoyé
    sortie = "data/processed/rtp_propre.csv"

    # Exécution du pipeline de nettoyage
    df_clean = pretraiter(entree)
    df_clean.to_csv(sortie, index=False)

    print(f"Données nettoyées enregistrées dans {sortie}")
