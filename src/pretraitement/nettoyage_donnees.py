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
    
    # DEBUG: Afficher les colonnes directement après le chargement
    print(f"✅ Fichier chargé. Colonnes brutes trouvées : {df.columns.tolist()[:10]}...") 
    return df

def renommer_colonnes_toxicite(df):
    """
    Renomme les colonnes de toxicité selon plusieurs formats possibles 
    (jsonl aplati, csv pré-traité, etc.) en 'toxicite_prompt' et 'toxicite_continuation'.
    """
    mapping = {}
    
    # Gestion de la colonne prompt
    if 'prompt.toxicity' in df.columns:
        mapping['prompt.toxicity'] = 'toxicite_prompt'
    elif 'prompt_toxicity' in df.columns:
        mapping['prompt_toxicity'] = 'toxicite_prompt'
    elif 'toxicity' in df.columns:
        mapping['toxicity'] = 'toxicite_prompt'

    # Gestion de la colonne continuation
    if 'continuation.toxicity' in df.columns:
        mapping['continuation.toxicity'] = 'toxicite_continuation'

    # Appliquer le renommage
    df = df.rename(columns=mapping)
    
    # DEBUG: Alerter si le renommage a échoué
    if 'toxicite_prompt' not in df.columns:
        print("⚠️ ATTENTION : La colonne 'toxicite_prompt' n'a pas pu être créée. Vérifiez le nom exact de vos colonnes de base.")
        print(f"Colonnes actuelles : {df.columns.tolist()}")
        
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
    for col in ["refus", "flag_challenging"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Convertir les scores de toxicité en drapeaux binaires
    if 'toxicite_prompt' in df.columns:
        # On s'assure d'abord qu'il s'agit bien de nombres avant d'appliquer la condition > 0.5
        df['toxicite_prompt'] = pd.to_numeric(df['toxicite_prompt'], errors='coerce')
        df['toxicite_prompt_binaire'] = (df['toxicite_prompt'] > 0.5).astype(int)
        
    if 'toxicite_continuation' in df.columns:
        df['toxicite_continuation'] = pd.to_numeric(df['toxicite_continuation'], errors='coerce')
        df['toxicite_continuation_binaire'] = (df['toxicite_continuation'] > 0.5).astype(int)

    return df

def pretraiter(chemin: str) -> pd.DataFrame:
    """
    Fonction principale de prétraitement des données.
    """
    print(f"--- Démarrage du prétraitement pour : {chemin} ---")
    df = charger_donnees(chemin)
    df = renommer_colonnes_toxicite(df)
    df = nettoyer_numeriques(df)
    df = nettoyer_binaires(df)
    print("--- Prétraitement terminé avec succès ---")
    return df

if __name__ == "__main__":
    pass
