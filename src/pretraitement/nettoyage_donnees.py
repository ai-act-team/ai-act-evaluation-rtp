import pandas as pd
import json

def charger_donnees(chemin: str) -> pd.DataFrame:
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
    mapping = {
        'prompt.toxicity':          'toxicite_prompt',
        'prompt.text':              'texte_prompt',
        'prompt.severe_toxicity':   'severe_toxicite_prompt',
        'continuation.toxicity':    'toxicite_continuation',
        'continuation.text':        'texte_continuation',
        'challenging':              'flag_challenging',
    }
    mapping_existant = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=mapping_existant)
    return df

def nettoyer_numeriques(df: pd.DataFrame) -> pd.DataFrame:
    cols_numeriques = ['toxicite_prompt', 'toxicite_continuation', 'severe_toxicite_prompt']
    for col in cols_numeriques:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df

def nettoyer_binaires(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['flag_challenging']:
        if col in df.columns:
            df[col] = df[col].astype(bool).astype(int)
    if 'toxicite_prompt' in df.columns:
        df['toxicite_prompt_binaire'] = (df['toxicite_prompt'] > 0.5).astype(int)
    if 'toxicite_continuation' in df.columns:
        df['toxicite_continuation_binaire'] = (df['toxicite_continuation'] > 0.5).astype(int)
    if 'toxicite_prompt' in df.columns and 'toxicite_continuation' in df.columns:
        df['delta_t'] = df['toxicite_continuation'] - df['toxicite_prompt']
    return df

def pretraiter(chemin: str) -> pd.DataFrame:
    df = charger_donnees(chemin)
    df = renommer_colonnes(df)
    df = nettoyer_numeriques(df)
    df = nettoyer_binaires(df)
    print(f"  → Prétraitement terminé. Shape final : {df.shape}")
    return df

if __name__ == "__main__":
    pass
