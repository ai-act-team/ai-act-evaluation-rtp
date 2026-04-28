import pandas as pd
import json

def charger_donnees(chemin):
    """
    Charge les données depuis un fichier JSONL ou CSV et aplatit les champs imbriqués.
    """
    if chemin.endswith(".jsonl"):
        with open(chemin, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        df = pd.json_normalize(data)
    elif chemin.endswith(".csv"):
        df = pd.read_csv(chemin)
    else:
        raise ValueError(f"Format de fichier non pris en charge: {chemin}")
    return df

def renommer_colonnes_toxicite(df):
    """
    Renomme les colonnes 'prompt.toxicity' et 'continuation.toxicity'
    en 'toxicite_prompt' et 'toxicite_continuation' si elles existent.
    """
    if 'prompt.toxicity' in df.columns:
        df = df.rename(columns={'prompt.toxicity': 'toxicite_prompt'})
    if 'continuation.toxicity' in df.columns:
        df = df.rename(columns={'continuation.toxicity': 'toxicite_continuation'})
    return df

def nettoyer_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les colonnes numériques (exemple).
    """
    # Ajoutez ici la logique spécifique pour nettoyer les colonnes numériques
    return df

def nettoyer_binaires(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes binaires en 0/1, y compris les colonnes de toxicité.
    """
    # Exemple de traitement pour 'refus' et 'flag_challenging'
    for col in ["refus", "flag_challenging"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Convertir les scores de toxicité en drapeaux binaires (si nécessaire, adaptez le seuil)
    if 'toxicite_prompt' in df.columns:
        df['toxicite_prompt_binaire'] = (df['toxicite_prompt'] > 0.5).astype(int) # Exemple de seuil
    if 'toxicite_continuation' in df.columns:
        df['toxicite_continuation_binaire'] = (df['toxicite_continuation'] > 0.5).astype(int) # Exemple de seuil

    return df

def pretraiter(chemin: str) -> pd.DataFrame:
    """
    Fonction principale de prétraitement des données.
    """
    df = charger_donnees(chemin)
    df = renommer_colonnes_toxicite(df)
    df = nettoyer_numeriques(df)
    df = nettoyer_binaires(df)
    return df

if __name__ == "__main__":
    # Cette partie est exécutée si le script est lancé directement.
    # Pour l'utilisation via `%%run` dans un notebook, cette section est généralement ignorée.
    pass
