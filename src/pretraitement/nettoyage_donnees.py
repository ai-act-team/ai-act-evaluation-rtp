import pandas as pd


def charger_donnees(chemin: str) -> pd.DataFrame:
    """
    Charge un fichier JSONL ou CSV contenant les données RealToxicityPrompts.
    """
    if chemin.endswith(".jsonl"):
        df = pd.read_json(chemin, lines=True)
    elif chemin.endswith(".csv"):
        df = pd.read_csv(chemin)
    else:
        raise ValueError("Format non supporté : utilisez .jsonl ou .csv")
    return df


def renommer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme les colonnes importantes pour harmoniser le dataset.
    """
    mapping = {
        "prompt": "texte_prompt",
        "toxicity": "toxicite_prompt",
        "toxicity_continuation": "toxicite_reponse",
        "challenging": "flag_challenging",
        "refusal_flag": "refus"
    }

    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    colonnes_utiles = [
        "texte_prompt",
        "toxicite_prompt",
        "toxicite_reponse",
        "refus",
        "flag_challenging"
    ]

    df = df[[c for c in colonnes_utiles if c in df.columns]]
    return df


def nettoyer_texte(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les textes : suppression des espaces et des lignes vides.
    """
    df["texte_prompt"] = df["texte_prompt"].astype(str).str.strip()
    df = df[df["texte_prompt"].str.len() > 0]
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
