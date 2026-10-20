"""Personnalisation du LLM par fine-tuning supervisé LoRA (SFT) : prompt -> nom exact de la file.

Perte calculée uniquement sur la réponse (nom de la file + jeton de fin de tour), pas sur le prompt.
"""
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

from . import hf_model
from .config import BASE_MODEL


def encode(tok, row, compact=True):
    prompt = tok(hf_model.prompt_text(tok, row, compact), add_special_tokens=False).input_ids
    answer = tok(row["queue"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    return prompt + answer, [-100] * len(prompt) + answer


def collate(batch, pad_id):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)], lab[i, :len(b)], att[i, :len(a)] = torch.tensor(a), torch.tensor(b), 1
    return ids.cuda(), lab.cuda(), att.cuda()


def train(df_train, df_val=None, out_dir=None, epochs=3, lr=2e-4, rank=16, alpha=32, dropout=0.05,
          batch_size=4, grad_accum=2, seed=0, base=BASE_MODEL, compact=True, log=print):
    """Entraîne un adaptateur LoRA. Si df_val est fourni, évalue F1 pondéré / exactitude à chaque époque."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tok = hf_model.load_tokenizer(base)
    model = hf_model.load_model(base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    data = [encode(tok, r, compact) for _, r in df_train.iterrows()]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * -(-len(data) // (batch_size * grad_accum))
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    history, t0, step = [], time.time(), 0
    for ep in range(1, epochs + 1):
        model.train()
        order = np.random.RandomState(seed * 100 + ep).permutation(len(data))
        losses = []
        for k, i in enumerate(range(0, len(order), batch_size)):
            ids, lab, att = collate([data[j] for j in order[i:i + batch_size]], tok.pad_token_id)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / grad_accum).backward()
            losses.append(loss.item())
            if (k + 1) % grad_accum == 0 or i + batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)), "secondes": round(time.time() - t0)}
        if df_val is not None:
            model.eval()
            pred = hf_model.predict(hf_model.score(model, tok, df_val, compact=compact))
            rec["val_f1_pondere"] = float(f1_score(df_val["queue"], pred, average="weighted"))
            rec["val_exactitude"] = float(accuracy_score(df_val["queue"], pred))
        history.append(rec)
        log(json.dumps(rec))
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
    del model, opt
    gc.collect(); torch.cuda.empty_cache()
    return history
