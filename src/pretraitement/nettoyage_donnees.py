import pandas as pd
import json
 
def charger_donnees(chemin: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier JSONL ou CSV et aplatit les champs imbriqués.
    Pour le dataset RealToxicityPrompts, pd.json_normalize produit des colonnes
    comme 'prompt.toxicity', 'prompt.text', 'continuation.toxicity', etc.
    """
    try:
        if chemin.endswith(".jsonl"):
            with open(chemin, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            df = pd.json_normalize(data)
        elif chemin.endswith(".csv"):
            df = pd.read_csv(chemin)
        else:
            raise ValueError(f"Format de fichier non pris en charge : {chemin}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")
    print(f"  → {len(df)} lignes chargées. Colonnes : {list(df.columns)}")
    return df
 
def renommer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes issues de json_normalize vers des noms lisibles.
    Couvre la toxicité, le texte, et le flag challenging du dataset RTP.
    """
    mapping = {
        'prompt.toxicity':          'toxicite_prompt',
        'prompt.text':              'texte_prompt',
        'prompt.severe_toxicity':   'severe_toxicite_prompt',
        'continuation.toxicity':    'toxicite_reponse',
        'continuation.text':        'texte_reponse',
        'challenging':              'flag_challenging',
    }
    # On ne renomme que les colonnes qui existent réellement
    mapping_existant = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=mapping_existant)
    return df
 
def nettoyer_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    """
    S'assure que les colonnes de toxicité sont bien de type float,
    et remplace les valeurs manquantes par 0.
    """
    cols_numeriques = ['toxicite_prompt', 'toxicite_continuation', 'severe_toxicite_prompt']
    for col in cols_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df
 
def nettoyer_binaires(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes binaires en int (0/1).
    Crée aussi des colonnes binaires à partir des scores de toxicité (seuil 0.5).
    """
    for col in ['flag_challenging']:
        if col in df.columns:
            df[col] = df[col].astype(bool).astype(int)
 
    if 'toxicite_prompt' in df.columns:
        df['toxicite_prompt_binaire'] = (df['toxicite_prompt'] > 0.5).astype(int)
    if 'toxicite_continuation' in df.columns:
        df['toxicite_continuation_binaire'] = (df['toxicite_continuation'] > 0.5).astype(int)
 
    # Colonne delta_t : amplification de la toxicité par la continuation
    if 'toxicite_prompt' in df.columns and 'toxicite_continuation' in df.columns:
        df['delta_t'] = df['toxicite_continuation'] - df['toxicite_prompt']
 
    return df
 
def pretraiter(chemin: str) -> pd.DataFrame:
    """
    Pipeline complet de prétraitement :
    chargement → renommage → nettoyage numérique → colonnes binaires.
    """
    df = charger_donnees(chemin)
    df = renommer_colonnes(df)
    df = nettoyer_numeriques(df)
    df = nettoyer_binaires(df)
    print(f"  → Prétraitement terminé. Shape final : {df.shape}")
    return df
 
if __name__ == "__main__":
    pass
