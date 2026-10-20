import json

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import COLUMNS, SEED, SPLITS_DIR, TEST_SIZE, TICKETS_CSV, VAL_SIZE


def load_tickets(path=TICKETS_CSV) -> pd.DataFrame:
    """Charge uniquement les 5 colonnes autorisées (id, answer, type, priority, tags ne sont jamais lus)."""
    df = pd.read_csv(path, usecols=COLUMNS)[COLUMNS].fillna("")
    df.index.name = "ticket"
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """train / val / test, tous stratifiés sur `queue` (les proportions des classes sont conservées)."""
    rest, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["queue"], random_state=seed)
    train, val = train_test_split(rest, test_size=VAL_SIZE, stratify=rest["queue"], random_state=seed)
    return train.sort_index(), val.sort_index(), test.sort_index()


def save_splits(train, val, test):
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(SPLITS_DIR / f"{name}.csv")


def load_splits():
    return tuple(pd.read_csv(SPLITS_DIR / f"{n}.csv", index_col="ticket").fillna("") for n in ("train", "val", "test"))


def proportions(*frames_names) -> pd.DataFrame:
    """Tableau des proportions de classes par ensemble (preuve de la stratification)."""
    cols = {n: f["queue"].value_counts(normalize=True) for n, f in frames_names}
    return pd.DataFrame(cols).fillna(0.0).sort_index()


def to_sft_jsonl(df: pd.DataFrame, path, build_messages):
    """Ecrit les exemples d'entraînement au format chat : messages + réponse attendue (la queue)."""
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            ex = {"messages": build_messages(r) + [{"role": "assistant", "content": r["queue"]}]}
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
