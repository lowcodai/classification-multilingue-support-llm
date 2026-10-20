import json
from functools import lru_cache

from .config import PROMPTS_DIR


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Charge un template depuis prompts/ (aucun prompt n'est écrit en dur dans le code)."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def queue_definitions() -> dict:
    return json.loads(load("queues.json"))


def labels() -> list:
    return sorted(queue_definitions())


def system_prompt(compact: bool = False) -> str:
    """compact=False : prompt complet avec définitions des files (utilisé pour le modèle de base, le plus favorable).
    compact=True : liste des files seule (utilisé pour le fine-tuning : les définitions deviennent inutiles et
    triplent le coût de calcul de chaque exemple)."""
    if compact:
        return load("01b_system_classification_compact.md").format(queues="\n".join(f"- {q}" for q in queue_definitions()))
    liste = "\n".join(f"- {q} : {d}" for q, d in queue_definitions().items())
    return load("01_system_classification.md").format(queues=liste)


def user_prompt(row) -> str:
    return load("02_user_ticket.md").format(
        business_type=row["business_type"], language=row["language"],
        subject=row["subject"] or "(sans objet)", body=row["body"])


def messages(row, compact: bool = False) -> list:
    """Message utilisateur identique pour toutes les versions du LLM ; seul le prompt système change (compact ou complet)."""
    return [{"role": "system", "content": system_prompt(compact)},
            {"role": "user", "content": user_prompt(row)}]
