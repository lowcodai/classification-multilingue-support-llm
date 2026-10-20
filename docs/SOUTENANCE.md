# Soutenance — Classification multilingue de demandes de support par un LLM

Formation LLM Engineer · Auteur : Jérémie Coste · Environnement : DGX Spark + JupyterLab

---

## 1. Le projet en 30 secondes
Un prestataire de support reçoit des milliers de messages en 5 langues et les classe à la main en 10 files. J'ai personnalisé un **LLM open weights** (`Qwen3-8B` + LoRA) pour les classer automatiquement, et mesuré le gain sur un test stratifié, le tout en local sur un DGX Spark.

**Preuve :** F1 pondéré **0,471 → 0,933** (seuil demandé 0,92) sur 120 demandes de test, avec les réserves du §7 : 2 entraînements sur 3 font moins de 0,92, et une partie du score vient de quasi-doublons entre entraînement et test.

## 2. Contexte et contraintes
- 5 colonnes seulement (`subject, body, queue, language, business_type`) ; comparer LLM non personnalisé / personnalisé ; **F1 pondéré ≥ 92 %** ; test **stratifié** ; capture d'écran des F1_j et du F1 pondéré pour les deux versions.
- Cloud Sandbox remplacé par mon **DGX Spark** : `Qwen3-8B` (bf16), fine-tuning LoRA avec PEFT, serveur vLLM existant pour le point de comparaison 27B.
- Livrable central : **templates de prompts** (`prompts/`) + code Python + notebooks exécutés + capture des F1.

## 3. Démarche
1. **Inspecter avant d'entraîner** : classes très déséquilibrées (35 % → 0,8 %), 5 langues, étiquettes bruitées (TF-IDF ≈ 0,67), et **quasi-doublons traduits** (le plus proche voisin porte la même file dans 97,7 % des cas).
2. **Construire un test fiable** : 20 % stratifié (120), validation 20 % du reste pour les hyperparamètres, test non utilisé avant le notebook 04.
3. **Mesurer la version non personnalisée** avec le prompt le plus favorable (définitions des files).
4. **Personnaliser** (LoRA), choisir sur la validation, réentraîner sur train + val avec 3 graines, désigner la graine de référence **avant** de regarder le test.
5. **Contrôler le score** : intervalles bootstrap, 3 graines, F1 par langue, effet des quasi-doublons, modèles de comparaison sans personnalisation.

## 4. Architecture
```
demande → prompt (système + secteur, langue, objet, message) → Qwen3-8B [+ LoRA]
        → probabilité du 1er jeton de chacune des 10 files → file prédite
```
| Fichier | Rôle |
|---|---|
| `prompts/` | tous les templates (voir `prompts/README.md`) |
| `classif/data.py` | 5 colonnes seulement, découpage stratifié, JSONL d'entraînement |
| `classif/hf_model.py` | inférence locale identique pour les deux versions |
| `classif/finetune.py` | SFT LoRA, perte sur la réponse |
| `classif/evaluation.py` | F1_j et F1 pondéré (formule de l'énoncé, contrôlée vs scikit-learn), bootstrap, capture |

## 5. Choix de conception et justification
- **Décision par probabilité du 1er jeton** : les 10 files commencent par des jetons distincts → équivalent à une génération contrainte, aucune sortie hors liste (vérifié : 100 % de sorties valides en génération libre).
- **Même entrée pour les deux versions** : seule change la valeur des poids → la comparaison mesure bien l'effet de la personnalisation.
- **Prompt complet pour le modèle de base, compact pour le fine-tuning** : on aide au maximum le modèle de base (0,471 contre 0,391 en compact) ; le fine-tuning n'a plus besoin des définitions et coûte 2× moins cher.
- **LoRA plutôt que fine-tuning complet** : 8 Md de paramètres sur un GPU partagé ; adaptateur ≈ 170 Mo.
- **Graine de référence désignée à l'avance** + 3 graines rapportées : pas de choix de la meilleure graine sur le test.
- **Mémoire GPU plafonnée à 30 Go** : le serveur vLLM partagé n'est pas perturbé.

## 6. Difficultés rencontrées et corrections (à citer)
| Constat | Correction |
|---|---|
| Première série : validation encore en hausse (0,626 → 0,756), 9 min par époque | Interrompue ; prompt compact (≈ 2× plus rapide), plus d'époques, lr plus élevé |
| lr 2e-4 × 6 époques : 0,829 | lr 4e-4 × 4 époques : 0,893 → retenu |
| Un plus proche voisin sans LLM faisait déjà 0,906 sur la validation | Mesure de l'effet des quasi-doublons et publication des deux lectures du score |
| Capture trop grossière (en-têtes qui se chevauchent) | Fonction de capture corrigée, notebook 04 réexécuté |

## 7. Résultats et lecture critique
- **F1 pondéré** : non personnalisé **0,471** → personnalisé **0,933** (IC95 % [0,884 ; 0,975]). Graines 0/1/2 : 0,933 / 0,908 / 0,916.
- **Mais** : (a) le seuil de 0,92 est atteint par le modèle désigné, moyenne 0,919 ; (b) 73 % des tickets de test ont un voisin très proche dans l'entraînement ; sur les 32 tickets sans quasi-doublon, 84 % de bonnes réponses (contre 44 % sans personnalisation) ; (c) un **plus proche voisin sans LLM** obtient 0,975 sur ce test ; (d) portugais plus faible (0,75, 15 tickets).
- Sans personnalisation, ni le Qwen3-8B (0,471) ni le Qwen3.8-27B (≈ 0,43) ne devinent les conventions du service : les files sont définies par les opérateurs.
- Détails : `resultats/`, notebook 04.

## 8. Démonstration suggérée (3–4 minutes)
1. Notebook `01` : GPU, 5 colonnes seulement, déséquilibre, étiquettes ambiguës, quasi-doublons.
2. Notebook `02` : stratification (tableau des proportions), prompts d'exemple.
3. Notebook `03` : référence non personnalisée sur la validation, grille d'hyperparamètres, entraînements finaux.
4. Notebook `04` : **capture des F1**, matrices de confusion, 3 graines, contrôle des quasi-doublons.

## 9. Limites et améliorations
Voir `docs/RAPPORT.md` : test de 120 tickets, un seul découpage, 4 essais d'hyperparamètres sur 96 tickets de validation, quasi-doublons, un seul modèle de base, pas de seuil de confiance ni d'API. Pistes : découpage par groupes de doublons, exemples proches dans le prompt, validation croisée, renvoi vers un opérateur sous un seuil de confiance, Llama/Mistral.

## 10. Questions probables
**Le seuil de 92 % est-il vraiment atteint ?** Pour le modèle de référence désigné à l'avance, oui (0,933). Sur trois entraînements, la moyenne est 0,919 : le seuil est atteint sans marge, et l'IC95 % descend à 0,884.

**Pourquoi ne pas simplement prendre le plus proche voisin, qui fait 0,975 ?** Sur ce jeu, c'est effectivement très fort, ce que je rapporte. Le LLM personnalisé répond toujours dans les 10 files, comprend le texte libre et dépasse 84 % sur les tickets sans copie ; mais l'énoncé demande un LLM, et je ne présente pas le LLM comme meilleur qu'un voisin.

**Le test a-t-il servi à régler quelque chose ?** Non : hyperparamètres choisis sur la validation, graine de référence désignée avant. Le notebook 04 a été exécuté plusieurs fois pour corriger un libellé et la capture, sans changer aucun choix.

**Pourquoi un prompt différent pour le fine-tuning ?** Coût de calcul : les définitions triplent la longueur de chaque exemple. Le modèle de base garde les définitions pour ne pas le désavantager.

**Qu'est-ce qui reste faible ?** Voir §9 ; en particulier le portugais et les classes à 1-3 tickets de test.

**Un autre modèle de base aurait-il été meilleur ?** Non testé ; le code est indépendant du modèle (variable `BASE_MODEL`).

## 11. Où trouver quoi
`README.md`, `prompts/`, `notebooks/`, `resultats/capture_f1_scores.png`, `docs/RAPPORT.md`, `docs/RUNBOOK_JUPYTERLAB.md`.
