import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel
from transformers import pipeline as hf_pipeline
 
 
sns.set(style="whitegrid", palette="muted")
  
 
# ===============================
# SCORING TOXIC-BERT SUR df
# ===============================
 
# Chargement unique du modèle (évite de le recharger à chaque appel)
_evaluateur_toxicite = None
 
def _get_evaluateur():
    global _evaluateur_toxicite
    if _evaluateur_toxicite is None:
        _evaluateur_toxicite = hf_pipeline(
            "text-classification",
            model="unitary/toxic-bert"
        )
    return _evaluateur_toxicite
 
 
def _scorer_texte(texte: str) -> float:
    """Score toxic-bert pour un texte. Retourne 0.0 si vide."""
    if not texte or str(texte).strip() == "":
        return 0.0
    result = _get_evaluateur()(str(texte), truncation=True, max_length=512)[0]
    return float(result["score"])
 
 
def scorer_dataset_bert(df: pd.DataFrame, n_echantillon: int = None) -> pd.DataFrame:
    """
    Calcule les scores toxic-bert sur les colonnes texte_prompt et texte_continuation
    du dataset nettoyé df, et les ajoute comme nouvelles colonnes.
 
    Paramètres
    ----------
    df           : DataFrame issu de pretraiter(), doit contenir texte_prompt
                   et texte_continuation.
    n_echantillon: si fourni, on travaille sur un sous-échantillon aléatoire
                   pour limiter le temps de calcul (optionnel).
 
    Retourne le DataFrame avec deux colonnes supplémentaires :
    - toxicite_prompt_bert
    - toxicite_continuation_bert
    """
    if n_echantillon and n_echantillon < len(df):
        df = df.sample(n_echantillon, random_state=42).copy()
    else:
        df = df.copy()
 
    print(f"  Scoring toxic-bert sur {len(df)} lignes (prompt + continuation)...")
 
    df["toxicite_prompt_bert"] = df["texte_prompt"].apply(_scorer_texte)
 
    if "texte_continuation" in df.columns:
        df["toxicite_continuation_bert"] = df["texte_continuation"].apply(_scorer_texte)
    else:
        print("  [AVERTISSEMENT] Colonne texte_continuation absente, scoring continuation ignoré.")
 
    print("  Scoring terminé ✔")
    return df
 
 
# ===============================
# CALCUL DU BIAIS
# ===============================
 
def calculer_biais(df: pd.DataFrame) -> dict:
    """
    Compare les scores de toxicité du dataset RTP (Perspective API)
    avec les scores recalculés par toxic-bert.
 
    Retourne un dictionnaire avec :
    - biais_moyen        : différence moyenne (bert - rtp)
    - ecart_type_biais   : dispersion de cette différence
    - correlation_r      : corrélation de Pearson entre les deux méthodes
    - p_value_correlation: significativité de la corrélation
    - t_stat, p_value_t  : test de Student apparié (H0 : biais = 0)
    - comparables        : bool — décision finale
    - raison             : explication textuelle de la décision
    """
    paires = [
        ("toxicite_prompt",       "toxicite_prompt_bert",       "prompt"),
        ("toxicite_continuation", "toxicite_continuation_bert", "continuation"),
    ]
 
    resultats = {}
 
    for col_rtp, col_bert, label in paires:
        if col_rtp not in df.columns or col_bert not in df.columns:
            print(f"  [AVERTISSEMENT] Colonnes manquantes pour '{label}' : "
                  f"'{col_rtp}' ou '{col_bert}' absente du DataFrame.")
            continue
 
        serie = df[[col_rtp, col_bert]].dropna()
        if len(serie) < 10:
            print(f"  [AVERTISSEMENT] Trop peu d'observations pour '{label}' ({len(serie)} lignes).")
            continue
 
        rtp  = serie[col_rtp].values
        bert = serie[col_bert].values
 
        differences   = bert - rtp
        biais_moyen   = float(np.mean(differences))
        ecart_type    = float(np.std(differences, ddof=1))
        corr, p_corr  = pearsonr(rtp, bert)
        t_stat, p_t   = ttest_rel(bert, rtp)
 
        # --- Décision de comparabilité ---
        # Seuils choisis : corrélation forte (r > 0.7) ET biais absolu < 0.15
        seuil_corr  = 0.70
        seuil_biais = 0.15
 
        comparables = bool(abs(corr) >= seuil_corr and abs(biais_moyen) < seuil_biais)
 
        if abs(corr) < seuil_corr and abs(biais_moyen) >= seuil_biais:
            raison = (f"Corrélation insuffisante (r={corr:.3f} < {seuil_corr}) "
                      f"ET biais trop élevé (|biais|={abs(biais_moyen):.3f} ≥ {seuil_biais}).")
        elif abs(corr) < seuil_corr:
            raison = (f"Corrélation insuffisante (r={corr:.3f} < {seuil_corr}) — "
                      f"les deux méthodes ne mesurent pas le même signal.")
        elif abs(biais_moyen) >= seuil_biais:
            raison = (f"Biais systématique trop élevé (|biais|={abs(biais_moyen):.3f} ≥ {seuil_biais}) — "
                      f"toxic-bert sur-/sous-estime systématiquement Perspective API.")
        else:
            raison = (f"Corrélation suffisante (r={corr:.3f} ≥ {seuil_corr}) "
                      f"ET biais acceptable (|biais|={abs(biais_moyen):.3f} < {seuil_biais}).")
 
        resultats[label] = {
            "col_rtp":              col_rtp,
            "col_bert":             col_bert,
            "n":                    len(serie),
            "biais_moyen":          biais_moyen,
            "ecart_type_biais":     ecart_type,
            "correlation_r":        float(corr),
            "p_value_correlation":  float(p_corr),
            "t_stat":               float(t_stat),
            "p_value_t":            float(p_t),
            "comparables":          comparables,
            "raison":               raison,
        }
 
    return resultats
 
 
# ===============================
# AFFICHAGE CONSOLE
# ===============================
 
def afficher_biais(resultats: dict) -> None:
    """
    Affiche dans la console un résumé lisible des résultats de comparabilité.
    """
    print("\n" + "=" * 65)
    print("ANALYSE DE BIAIS — COMPARABILITÉ DES MÉTHODES DE SCORING")
    print("=" * 65)
    print("  Méthode A : Perspective API (scores du dataset RTP)")
    print("  Méthode B : toxic-bert (scores recalculés)")
    print("=" * 65)
 
    for label, r in resultats.items():
        verdict = "✅ COMPARABLES" if r["comparables"] else "❌ NON COMPARABLES"
        signe_biais = "+" if r["biais_moyen"] >= 0 else ""
        sig_corr = "✓ significative" if r["p_value_correlation"] < 0.05 else "✗ non significative"
        sig_t    = "✓ significatif"  if r["p_value_t"]           < 0.05 else "✗ non significatif"
 
        print(f"\n  [{label.upper()}]")
        print(f"  Nombre d'observations     : {r['n']}")
        print(f"  Corrélation de Pearson    : r = {r['correlation_r']:.3f}  "
              f"(p = {r['p_value_correlation']:.2e}, {sig_corr})")
        print(f"  Biais moyen (B − A)       : {signe_biais}{r['biais_moyen']:.4f}  "
              f"(σ = {r['ecart_type_biais']:.4f})")
        print(f"  Test Student apparié      : t = {r['t_stat']:.3f}  "
              f"(p = {r['p_value_t']:.2e}, {sig_t})")
        print(f"  → {r['raison']}")
        print(f"  → Verdict : {verdict}")
 
    print("\n" + "=" * 65 + "\n")
 
 
# ===============================
# GRAPHIQUES DE BIAIS
# ===============================
 
def visualiser_biais(df: pd.DataFrame, resultats: dict) -> None:
    """
    Produit deux graphiques par paire de méthodes comparées :
    1. Scatter (méthode A vs méthode B) avec droite de référence y=x
    2. Bland-Altman : différence (B−A) en fonction de la moyenne (A+B)/2
    """
    for label, r in resultats.items():
        col_rtp  = r["col_rtp"]
        col_bert = r["col_bert"]
 
        serie = df[[col_rtp, col_bert]].dropna()
        rtp   = serie[col_rtp].values
        bert  = serie[col_bert].values
 
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"Comparabilité des méthodes — {label}  "
            f"({'Comparables ✓' if r['comparables'] else 'Non comparables ✗'})",
            fontsize=13, fontweight="bold"
        )
 
        # --- Graphique 1 : Scatter ---
        ax = axes[0]
        ax.scatter(rtp, bert, alpha=0.35, s=12, color="steelblue")
        lim = [0, 1]
        ax.plot(lim, lim, "r--", linewidth=1.5, label="y = x (accord parfait)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Score Perspective API (dataset RTP)")
        ax.set_ylabel("Score toxic-bert (recalculé)")
        ax.set_title(f"Corrélation : r = {r['correlation_r']:.3f}")
        ax.legend(fontsize=8)
 
        # --- Graphique 2 : Bland-Altman ---
        ax2 = axes[1]
        moyenne   = (rtp + bert) / 2
        diff      = bert - rtp
        biais     = r["biais_moyen"]
        sd        = r["ecart_type_biais"]
        loa_sup   = biais + 1.96 * sd   # Limite d'accord supérieure
        loa_inf   = biais - 1.96 * sd   # Limite d'accord inférieure
 
        ax2.scatter(moyenne, diff, alpha=0.35, s=12, color="steelblue")
        ax2.axhline(biais,   color="red",    linewidth=1.5,
                    linestyle="-",  label=f"Biais = {biais:+.3f}")
        ax2.axhline(loa_sup, color="orange", linewidth=1.2,
                    linestyle="--", label=f"+1.96σ = {loa_sup:+.3f}")
        ax2.axhline(loa_inf, color="orange", linewidth=1.2,
                    linestyle="--", label=f"−1.96σ = {loa_inf:+.3f}")
        ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
 
        ax2.set_xlabel("Moyenne (Perspective API + toxic-bert) / 2")
        ax2.set_ylabel("Différence (toxic-bert − Perspective API)")
        ax2.set_title("Graphique de Bland-Altman")
        ax2.legend(fontsize=8)
 
        plt.tight_layout()
        plt.show()
