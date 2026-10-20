# Runbook — relancer JupyterLab (projet classification des demandes de support)

Contexte : JupyterLab est installé dans `~/projet_classification/.venv` sur le DGX Spark (`ssh dgxspark`). Il n'écoute que sur `127.0.0.1:8897` côté Spark (8899 = projet *assistant service client*, 8898 = projet *agent RAG*) ; on y accède depuis le Mac par un tunnel SSH. Le serveur vLLM (`localhost:8000`) n'est nécessaire que pour la ligne de comparaison « LLM 27B de référence » du notebook 04 ; le fine-tuning et l'inférence du Qwen3-8B se font dans le noyau Jupyter (GPU partagé : la mémoire du processus est plafonnée à 30 Go, variable `GPU_MEM_GB`).

## 1. Vérifier les prérequis (depuis le Mac)
```bash
ssh dgxspark 'curl -s localhost:8000/v1/models | head -c 120; echo; ls ~/projet_classification/.venv/bin/jupyter; ls ~/.cache/huggingface/hub | grep Qwen3-8B; free -g | head -2'
```
Il faut environ 25 Go de mémoire disponible pour l'entraînement (le serveur vLLM en occupe environ 75). Si le LLM ne répond pas, le démarrer relève du runbook vLLM du dépôt `itshaker-dgx-spark-V2`.

## 2. Démarrer JupyterLab sur le Spark (avec un nouveau token)
```bash
TOKEN=$(openssl rand -hex 16); echo "token=$TOKEN"
ssh dgxspark "cd ~/projet_classification && nohup .venv/bin/jupyter lab --no-browser --ip=127.0.0.1 --port=8897 --ServerApp.token=$TOKEN --notebook-dir=\$HOME/projet_classification > jupyter.log 2>&1 < /dev/null &"
```

## 3. Ouvrir le tunnel SSH (depuis le Mac)
```bash
ssh -f -N -o ExitOnForwardFailure=yes -L 8897:127.0.0.1:8897 dgxspark
```

## 4. Ouvrir dans le navigateur
`http://127.0.0.1:8897/lab/tree/notebooks/04_evaluation_comparaison.ipynb?token=<TOKEN>`
Noyau : **Python 3 (classif-support)**. Ordre : `01` → `02` → `03` (≈ 2 h : recherche d'hyperparamètres puis 3 entraînements finaux LoRA) → `04` (≈ 10 min). Les adaptateurs sont écrits dans `modeles/`, les résultats dans `resultats/`.

## 5. Exécution sans interface (équivalent)
```bash
ssh dgxspark 'cd ~/projet_classification/notebooks && for n in 01 02 03 04; do ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=36000 --ExecutePreprocessor.kernel_name=classif-support ${n}_*.ipynb; done'
```

## 6. Diagnostic
| Symptôme | Vérification / action |
|---|---|
| Page inaccessible | Tunnel absent : `lsof -iTCP:8897 -sTCP:LISTEN` ; relancer l'étape 3 |
| « Address already in use » côté Spark | `ssh dgxspark 'pkill -f "jupyter-lab.*8897"'` puis étape 2 |
| Récupérer le token d'un serveur en cours | `ssh dgxspark "pgrep -fa jupyter-lab \| grep -o 'token=[0-9a-f]*'"` |
| `OutOfMemoryError` au chargement ou à l'entraînement | Mémoire partagée avec vLLM : réduire `batch_size` dans `classif/finetune.py`, ou lever `GPU_MEM_GB` si de la mémoire est libre (`free -g`) |
| Noyau « classif-support » absent | `ssh dgxspark 'cd ~/projet_classification && .venv/bin/python -m ipykernel install --user --name classif-support --display-name "Python 3 (classif-support)"'` |
| Chargement du modèle lent (1-2 min) | Normal au premier chargement (lecture de 16 Go depuis le disque) |
| Notebook 04 : erreur de connexion à `localhost:8000` | Seule la ligne « LLM 27B de référence » en dépend ; démarrer vLLM ou supprimer cette ligne |

## 7. Arrêter proprement
```bash
ssh dgxspark 'pkill -f "jupyter-lab.*8897"'
pkill -f "ssh -f -N.*8897"
```
