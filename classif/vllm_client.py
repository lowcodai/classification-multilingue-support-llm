"""Point de comparaison : le LLM servi localement par vLLM (API compatible OpenAI), sans personnalisation.

La sortie est contrainte à la liste des files (décodage guidé) pour que la comparaison porte sur le choix
de la file et non sur le respect du format.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from .config import LLM_BASE_URL, LLM_MODEL
from .prompts import labels, messages


def model_name() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    with urllib.request.urlopen(LLM_BASE_URL + "/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def classify(df, guided=True, workers=4, max_tokens=24):
    client, model = OpenAI(base_url=LLM_BASE_URL, api_key="EMPTY", timeout=180), model_name()
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if guided:
        extra["structured_outputs"] = {"choice": labels()}

    def one(row):
        r = client.chat.completions.create(model=model, messages=messages(row), temperature=0,
                                           max_tokens=max_tokens, extra_body=extra)
        return (r.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, [r for _, r in df.iterrows()]))
