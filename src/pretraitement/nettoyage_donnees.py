i# src/pretraitement/nettoyage_donnees.py
"""
Nettoyage et prétraitement des données RealToxicityPrompts
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime


def load_prompts_jsonl(file_path):
    """
    Charge les prompts depuis un fichier JSONL
    
    Args:
        file_path: Chemin du fichier JSONL
        
    Returns:
        DataFrame pandas avec les prompts
    """
    
    print(f"[{datetime.now()}] Chargement des prompts depuis {file_path}...")
    
    try:
        prompts = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    prompt_data = json.loads(line)
                    prompts.append(prompt_data)
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Erreur ligne {line_num}: {e}")
                    continue
        
        df = pd.DataFrame(prompts)
        print(f"✓ {len(df)} prompts chargés")
        
        return df
    
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        raise


def clean_text(text):
    """
    Nettoie un texte
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    
    if not isinstance(text, str):
        return ""
    
    # Supprimer les espaces inutiles
    text = text.strip()
    
    # Normaliser les espaces
    text = ' '.join(text.split())
    
    # Supprimer les caractères de contrôle
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text


def preprocess_dataframe(df, text_column='text'):
    """
    Prétraite un DataFrame de prompts
    
    Args:
        df: DataFrame pandas
        text_column: Nom de la colonne contenant le texte
        
    Returns:
        DataFrame nettoyé
    """
    
    print(f"\n[{datetime.now()}] Prétraitement des données...")
    
    df = df.copy()
    
    # 1. Supprimer les lignes avec texte manquant
    print(f"  → Suppression des textes manquants...")
    initial_len = len(df)
    df = df[df[text_column].notna()]
    print(f"    Lignes supprimées: {initial_len - len(df)}")
    
    # 2. Nettoyer les textes
    print(f"  → Nettoyage des textes...")
    df[text_column] = df[text_column].apply(clean_text)
    
    # 3. Supprimer les textes vides après nettoyage
    initial_len = len(df)
    df = df[df[text_column].str.len() > 0]
    print(f"    Textes vides supprimés: {initial_len - len(df)}")
    
    # 4. Supprimer les doublons
    print(f"  → Suppression des doublons...")
    initial_len = len(df)
    df = df.drop_duplicates(subset=[text_column])
    print(f"    Doublons supprimés: {initial_len - len(df)}")
    
    # 5. Réinitialiser l'index
    df = df.reset_index(drop=True)
    
    print(f"\n✓ Prétraitement complété")
    print(f"  Lignes finales: {len(df)}")
    
    return df


def compute_text_statistics(df, text_column='text'):
    """
    Calcule des statistiques sur les textes
    
    Args:
        df: DataFrame pandas
        text_column: Nom de la colonne contenant le texte
        
    Returns:
        dict avec les statistiques
    """
    
    print(f"\n[{datetime.now()}] Calcul des statistiques...")
    
    lengths = df[text_column].str.len()
    word_counts = df[text_column].str.split().str.len()
    
    stats = {
        'n_samples': len(df),
        'text_length_mean': lengths.mean(),
        'text_length_std': lengths.std(),
        'text_length_min': lengths.min(),
        'text_length_max': lengths.max(),
        'word_count_mean': word_counts.mean(),
        'word_count_std': word_counts.std(),
        'word_count_min': word_counts.min(),
        'word_count_max': word_counts.max()
    }
    
    return stats


def print_statistics(stats):
    """Affiche les statistiques de façon lisible"""
    
    print(f"\n{'='*60}")
    print("STATISTIQUES DES TEXTES")
    print(f"{'='*60}")
    
    print(f"\nNombre d'échantillons: {stats['n_samples']}")
    
    print(f"\nLongueur des textes:")
    print(f"  Moyenne: {stats['text_length_mean']:.2f} caractères")
    print(f"  Écart-type: {stats['text_length_std']:.2f}")
    print(f"  Min: {stats['text_length_min']:.0f}")
    print(f"  Max: {stats['text_length_max']:.0f}")
    
    print(f"\nNombre de mots:")
    print(f"  Moyenne: {stats['word_count_mean']:.2f} mots")
    print(f"  Écart-type: {stats['word_count_std']:.2f}")
    print(f"  Min: {stats['word_count_min']:.0f}")
    print(f"  Max: {stats['word_count_max']:.0f}")
    
    print(f"\n{'='*60}")


# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    print("="*60)
    print("NETTOYAGE DES DONNÉES")
    print("="*60)
    
    # Créer données de test
    print("\n[1] Création de données de test...")
    df_test = pd.DataFrame({
        'text': [
            'This is a test prompt.',
            'Another test prompt...',
            '  Extra spaces here  ',
            None,
            'This is a test prompt.',  # Doublon
            'Valid prompt'
        ]
    })
    
    print(f"Données initiales: {len(df_test)} lignes")
    print(df_test)
    
    # Prétraitement
    print("\n[2] Prétraitement...")
    df_clean = preprocess_dataframe(df_test)
    print(f"\nDonnées nettoyées: {len(df_clean)} lignes")
    print(df_clean)
    
    # Statistiques
    print("\n[3] Statistiques...")
    stats = compute_text_statistics(df_clean)
    print_statistics(stats)
    
    # Sauvegarde (optionnel)
    print("\n[4] Sauvegarde (optionnel)")
    output_path = 'data_clean.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"✓ Données sauvegardées dans {output_path}")
