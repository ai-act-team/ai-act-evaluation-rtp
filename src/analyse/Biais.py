"""
calibration_biais.py
Mesure et correction du biais entre Perspective API et Toxic-BERT.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib, sqlite3
from transformers import pipeline

# ── 1. Charger l'ensemble pivot depuis la BDD ──────────────────
def charger_pivot(db_path="regalia.db", n=300):
    conn = sqlite3.connect(db_path)
    # On prend des textes de PROMPTS dont on connaît le score Perspective
    df = pd.read_sql_query("""
        SELECT p.texte, p.toxicity AS score_perspective
        FROM prompts p
        ORDER BY RANDOM() LIMIT ?
    """, conn, params=(n,))
    conn.close()
    # Stratifier : 100 faibles, 100 moyens, 100 forts
    faibles  = df[df.score_perspective < 0.3].sample(100, random_state=42)
    moyens   = df[(df.score_perspective >= 0.3) &
                  (df.score_perspective < 0.6)].sample(
                  min(100, len(df[(df.score_perspective >= 0.3) &
                  (df.score_perspective < 0.6)])), random_state=42)
    forts    = df[df.score_perspective >= 0.6].sample(
                  min(100, len(df[df.score_perspective >= 0.6])),
                  random_state=42)
    return pd.concat([faibles, moyens, forts]).reset_index(drop=True)

# ── 2. Scorer avec Toxic-BERT ──────────────────────────────────
toxic_pipe = pipeline(
    "text-classification",
    model="martin-ha/toxic-comment-model",
    truncation=True, max_length=512
)

def scorer_toxicbert(textes):
    results = toxic_pipe(textes, batch_size=16)
    scores = []
    for r in results:
        s = r["score"] if r["label"] == "toxic" else 1 - r["score"]
        scores.append(s)
    return scores

# ── 3. Calculer les métriques de biais ────────────────────────
def mesurer_biais(df_pivot):
    p = df_pivot["score_perspective"].values
    t = df_pivot["score_toxicbert"].values

    biais_moyen = np.mean(t - p)
    mae         = mean_absolute_error(p, t)
    r, pval     = stats.pearsonr(p, t)
    rho, _      = stats.spearmanr(p, t)

    # ICC(2,1) — accord inter-juges absolu
    n = len(p)
    grand_mean  = np.mean([p, t])
    SS_between  = n * np.var([np.mean(p), np.mean(t)])
    SS_error    = np.sum((p - t) ** 2) / 2
    MS_between  = SS_between
    MS_error    = SS_error / (n - 1)
    icc         = (MS_between - MS_error) / (MS_between + MS_error)

    print(f"Biais moyen (ToxicBERT − Perspective) : {biais_moyen:+.4f}")
    print(f"MAE                                   : {mae:.4f}")
    print(f"Corrélation de Pearson                : r={r:.4f}, p={pval:.2e}")
    print(f"Corrélation de Spearman               : ρ={rho:.4f}")
    print(f"ICC (accord inter-juges)              : {icc:.4f}")
    return {"biais": biais_moyen, "mae": mae,
            "r": r, "rho": rho, "icc": icc}

# ── 4. Calibration par régression isotonique ──────────────────
def calibrer(df_pivot):
    X = df_pivot["score_toxicbert"].values
    y = df_pivot["score_perspective"].values

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(X, y)
    joblib.dump(iso, "calibration_iso.pkl")
    print("✔ Modèle de calibration sauvegardé → calibration_iso.pkl")

    # MAE après calibration
    y_pred = iso.predict(X)
    mae_apres = mean_absolute_error(y, y_pred)
    print(f"MAE après calibration : {mae_apres:.4f}")
    return iso

# ── 5. Graphique de calibration ───────────────────────────────
def plot_calibration(df_pivot, iso):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    p = df_pivot["score_perspective"].values
    t = df_pivot["score_toxicbert"].values
    t_sort = np.sort(t)

    axes[0].scatter(p, t, alpha=0.4, s=15, label="Avant calibration")
    axes[0].plot([0,1],[0,1], "r--", label="y=x (accord parfait)")
    axes[0].set(title="Avant calibration",
               xlabel="Perspective", ylabel="Toxic-BERT")
    axes[0].legend()

    t_cal = iso.predict(t)
    axes[1].scatter(p, t_cal, alpha=0.4, s=15, color="#2ecc71",
                   label="Après calibration")
    axes[1].plot([0,1],[0,1], "r--", label="y=x (accord parfait)")
    axes[1].set(title="Après calibration isotonique",
               xlabel="Perspective", ylabel="Toxic-BERT calibré")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/calibration_biais.png", dpi=130)

# ── 6. Utilisation dans evaluation_llm.py ─────────────────────
def delta_t_calibre(toxicbert_score, perspective_prompt_score, iso):
    score_calibre = iso.predict([toxicbert_score])[0]
    return score_calibre - perspective_prompt_score

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    df_pivot = charger_pivot()
    df_pivot["score_toxicbert"] = scorer_toxicbert(
        df_pivot["texte"].tolist()
    )
    metriques = mesurer_biais(df_pivot)
    iso = calibrer(df_pivot)
    plot_calibration(df_pivot, iso)
