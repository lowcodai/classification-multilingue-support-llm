import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # sorties de notebooks lisibles
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TICKETS_CSV = DATA_DIR / "helpdesk_customer_tickets.csv"
SPLITS_DIR = DATA_DIR / "splits"
PROMPTS_DIR = BASE_DIR / "prompts"
RESULTS_DIR = BASE_DIR / "resultats"
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "modeles"))   # adaptateurs LoRA (non versionnés)

# Seules colonnes autorisées par l'énoncé : les autres ne sont jamais lues
COLUMNS = ["subject", "body", "queue", "language", "business_type"]

SEED = 42
TEST_SIZE = 0.20   # ensemble de test stratifié sur `queue`
VAL_SIZE = 0.20    # part de l'entraînement mise de côté pour choisir les hyperparamètres

# LLM open weights personnalisé (LoRA) : mêmes poids de base pour la version non personnalisée
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-8B")

# LLM de référence servi par vLLM (API compatible OpenAI), utilisé seulement comme point de comparaison
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # vide = premier modèle servi
