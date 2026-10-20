"""Inférence locale (Transformers) du LLM open weights, avec ou sans adaptateur LoRA.

Le modèle non personnalisé et le modèle personnalisé passent par exactement le même code :
- même prompt (prompts/), même gabarit de chat, mode « thinking » désactivé ;
- décision = file dont le premier jeton a la plus forte probabilité au prochain pas
  (équivalent à une génération gloutonne contrainte à la liste des 10 files, sans sortie hors liste).
"""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BASE_MODEL

transformers_logging = __import__('transformers').utils.logging
transformers_logging.disable_progress_bar()
from .prompts import labels, messages


def cap_gpu_memory(gb=None):
    """Plafonne la mémoire de ce processus : le GPU unifié du DGX Spark est partagé avec le serveur vLLM."""
    gb = gb or float(os.getenv("GPU_MEM_GB", 30))
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gb / total))


def load_tokenizer(base=BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    return tok


def load_model(base=BASE_MODEL, adapter=None):
    cap_gpu_memory()
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return model.eval()


def prompt_text(tok, row, compact=False) -> str:
    return tok.apply_chat_template(messages(row, compact), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def label_token_ids(tok) -> list:
    ids = [tok(l, add_special_tokens=False).input_ids[0] for l in labels()]
    assert len(set(ids)) == len(ids), "les files doivent commencer par des jetons distincts"
    return ids


@torch.no_grad()
def score(model, tok, df, batch_size=8, compact=False) -> np.ndarray:
    """Log-probabilités (n, 10) du premier jeton de chaque file, dans l'ordre de prompts.labels()."""
    ids = torch.tensor(label_token_ids(tok), device="cuda")
    texts = [prompt_text(tok, r, compact) for _, r in df.iterrows()]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        out.append(torch.log_softmax(logits, dim=-1)[:, ids].cpu())
    return torch.cat(out).numpy()


def predict(scores: np.ndarray) -> list:
    L = labels()
    return [L[i] for i in scores.argmax(1)]


@torch.no_grad()
def generate(model, tok, df, max_new_tokens=16, batch_size=8, compact=False) -> list:
    """Génération libre gloutonne (sans contrainte) : sert à vérifier que le modèle respecte le format."""
    texts = [prompt_text(tok, r, compact) for _, r in df.iterrows()]
    res = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        res += [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for g in gen]
    return res
