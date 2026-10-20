"""Métriques demandées par l'énoncé : F1_j par classe et F1 pondéré (poids = proportion de la classe dans le test)."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .prompts import labels


def f1_by_class(y_true, y_pred, classes=None) -> pd.DataFrame:
    """F1_j = 2 TP_j / (2 TP_j + FP_j + FN_j) ; weighted F1 = somme_j alpha_j * F1_j, alpha_j = n_j / n."""
    classes = classes or labels()
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        rows.append({"classe": c, "support": tp + fn, "TP": tp, "FP": fp, "FN": fn,
                     "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "rappel": tp / (tp + fn) if tp + fn else 0.0,
                     "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0})
    df = pd.DataFrame(rows).set_index("classe")
    w = df["support"] / df["support"].sum()
    df.attrs["weighted_f1"] = float((w * df["F1"]).sum())
    # contrôle croisé avec scikit-learn
    assert abs(df.attrs["weighted_f1"] - f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)) < 1e-9
    return df


def weighted_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=labels(), average="weighted", zero_division=0)


def bootstrap_ci(y_true, y_pred, n=2000, seed=0, level=0.95):
    """IC bootstrap du F1 pondéré (variabilité due à la petite taille du test)."""
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    vals = [weighted_f1(y_true[i], y_pred[i]) for i in (rng.randint(0, len(y_true), len(y_true)) for _ in range(n))]
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def confusion(y_true, y_pred) -> pd.DataFrame:
    L = labels()
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=L), index=L, columns=L)


def capture_png(reports: dict, path, title="F1-score par classe et F1 pondéré — jeu de test stratifié"):
    """Image de synthèse demandée par l'énoncé : F1_j de chaque classe + F1 pondéré, pour chaque version du LLM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(reports)
    first = reports[names[0]]
    cells = [[c, int(first.loc[c, "support"])] + [f"{reports[n].loc[c, 'F1']:.3f}" for n in names] for c in first.index]
    cells.append(["F1 pondéré (weighted)", int(first["support"].sum())] +
                 [f"{reports[n].attrs['weighted_f1']:.3f}" for n in names])
    fig, ax = plt.subplots(figsize=(7.2 + 2.0 * len(names), 0.46 * len(cells) + 1.5))
    ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    tab = ax.table(cellText=cells, colLabels=["Classe j", "Support"] + [f"F1_j — {n}" for n in names],
                   loc="center", cellLoc="center", colWidths=[0.40, 0.10] + [0.50 / len(names)] * len(names))
    tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 1.55)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f3a5f"); cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
        elif r == len(cells):
            cell.set_facecolor("#e8eef7"); cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
