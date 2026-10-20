# Rapport synthétique

## Architecture
`demande (secteur, langue, objet, message) → prompt (prompts/) → Qwen3-8B (bf16, local) [+ adaptateur LoRA] → probabilité du premier jeton de chacune des 10 files → file prédite`. Tout est local (DGX Spark) ; seuls les poids de `Qwen3-8B` et de `bge-m3` ont été téléchargés depuis HuggingFace, aucune donnée n'est envoyée.

- **Décision** : la file dont le premier jeton a la plus forte probabilité au prochain pas. Les 10 files commencent par des jetons distincts (*Billing, Customer, General, Human, IT, Product, Returns, Sales, Service, Technical*), donc c'est exactement la génération gloutonne contrainte à la liste. Contrôle : en génération libre non contrainte, **100 %** des sorties du modèle de base comme du modèle personnalisé sont un nom de file exact, et le modèle personnalisé produit le même nom que la décision par probabilité dans 100 % des cas.
- **Mêmes entrées pour les deux versions** : même code, même gabarit de chat, mode « thinking » désactivé. Seule la valeur des poids change (adaptateur LoRA).

## Étape 1 — Données, découpage, prompts
- **Colonnes** : `usecols` ne charge que `subject, body, queue, language, business_type` (1 objet vide sur 600, remplacé par « (sans objet) »).
- **Classes** : 10 files très déséquilibrées (*Technical Support* 35 %, *General Inquiry* 0,8 % = 5 tickets). Langues : en 163, de 152, es 133, pt 81, fr 71.
- **Découpage** (graine 42) : test 20 % stratifié (120), puis validation 20 % du reste (96), entraînement 384. Proportions de classes du test à ≤ 0,33 point du jeu initial. Aucun ticket commun entre ensembles.
- **Prompts** (`prompts/`) : prompt système **complet** (rôle, 10 files avec une définition générique rédigée à partir des seuls noms, format de sortie, demande traitée comme donnée) pour le modèle de base ; prompt **compact** (liste des files seule) pour le fine-tuning. Le message utilisateur (secteur, langue, objet, message) est identique.

## Étape 2 — Personnalisation (notebook 03)
- **Méthode** : SFT avec adaptateur **LoRA** (rang 16, α 32, dropout 0,05, projections q/k/v/o/gate/up/down), perte calculée sur la seule réponse (nom de la file + fin de tour), bf16, checkpointing des gradients, lot effectif 8, décroissance linéaire du taux d'apprentissage, mémoire du processus plafonnée à 30 Go (GPU partagé avec le serveur vLLM).
- **Choix des hyperparamètres sur la validation seule** (le test n'a servi à rien avant le notebook 04) :

| Essai (entraînement sur `train`) | Prompt | F1 pondéré validation | Perte finale |
|---|---|---|---|
| Modèle de base, sans adaptateur | complet / compact | 0,378 / — | — |
| lr 1e-4, 3 époques (série archivée) | complet | 0,626 | 0,19 |
| lr 2e-4, 3 époques (série archivée) | complet | 0,756 | 0,12 |
| lr 2e-4, 6 époques | compact | 0,829 | 0,0003 |
| **lr 4e-4, 4 époques (retenu)** | compact | **0,893** | 0,020 |

- **Modèle final** : réentraînement sur train + validation (480 tickets) avec la configuration retenue, **3 graines** (≈ 20 min chacune). La graine 0 a été désignée comme modèle de référence **avant** l'évaluation sur le test.

## Étape 3 — Évaluation sur le test (notebook 04)
Même test de 120 demandes pour toutes les lignes. F1 par classe `F1_j = 2TP_j/(2TP_j+FP_j+FN_j)` et F1 pondéré `Σ α_j F1_j` (α_j = part de la classe dans le test), vérifiés contre scikit-learn.

| Version | F1 pondéré | IC95 % bootstrap | Exactitude |
|---|---|---|---|
| **Qwen3-8B non personnalisé** (prompt complet) | **0,471** | [0,380 ; 0,572] | 0,533 |
| Qwen3-8B non personnalisé, prompt compact | 0,391 | [0,305 ; 0,483] | 0,417 |
| **Qwen3-8B + LoRA (graine 0, référence)** | **0,933** | [0,884 ; 0,975] | 0,933 |
| Qwen3-8B + LoRA, graine 1 / graine 2 | 0,908 / 0,916 | — | 0,908 / 0,917 |
| *Qwen3.8-27B vLLM, sans personnalisation (comparaison)* | ≈ 0,43 | ≈ [0,33 ; 0,53] | ≈ 0,51 |
| *TF-IDF caractères + régression logistique (sans LLM)* | 0,636 | [0,540 ; 0,714] | 0,650 |
| *1 plus proche voisin `bge-m3` (sans LLM)* | 0,975 | [0,943 ; 1,000] | 0,975 |

**F1 par classe** (capture : [`resultats/capture_f1_scores.png`](../resultats/capture_f1_scores.png) ; détail précision/rappel : `resultats/detail_par_classe.csv`) :

| Classe | Support | Non personnalisé | Personnalisé |
|---|---|---|---|
| Billing and Payments | 9 | 0,941 | 1,000 |
| Customer Service | 17 | 0,000 | 0,938 |
| General Inquiry | 1 | 0,000 | 1,000 |
| Human Resources | 3 | 0,000 | 1,000 |
| IT Support | 15 | 0,194 | 0,903 |
| Product Support | 19 | 0,432 | 0,895 |
| Returns and Exchanges | 8 | 0,889 | 0,941 |
| Sales and Pre-Sales | 3 | 0,000 | 1,000 |
| Service Outages and Maintenance | 3 | 0,857 | 0,800 |
| Technical Support | 42 | 0,648 | 0,941 |
| **F1 pondéré** | 120 | **0,471** | **0,933** |

Le modèle de base ne peut pas deviner les files qui dépendent des conventions des opérateurs : il n'attribue jamais *Customer Service* (F1 0) et sur-attribue *Technical Support* / *IT Support*. Le fine-tuning corrige cela. Le personnalisé fait 8 erreurs sur 120, toutes entre files voisines (Technical / Product / IT Support, Customer Service / Product Support ou Returns).

### Lecture critique du score
1. **Le seuil est atteint, sans marge confortable.** 0,933 pour la graine désignée, mais 0,908 et 0,916 pour les deux autres (moyenne 0,919, écart-type 0,010). Avec 120 tickets, l'IC95 % [0,884 ; 0,975] contient des valeurs sous 0,92. Dire « le modèle dépasse 92 % » est vrai pour le modèle de référence, pas pour tout entraînement.
2. **Quasi-doublons entre entraînement et test.** Pour chaque ticket du jeu complet, le plus proche voisin (embeddings multilingues `bge-m3`) porte la même file dans 97,7 % des cas, et 70 % de ces voisins sont dans une autre langue : le jeu est constitué de modèles de messages déclinés (traductions, reformulations). Un test tiré au hasard contient donc presque toujours une « copie » du ticket dans l'entraînement : 73 % des tickets de test ont un voisin à ≥ 0,90 de similarité, 28 % à ≥ 0,95.

| Similarité max avec l'entraînement | Tickets de test | Exactitude non personnalisé | **Exactitude personnalisé** | 1-plus proche voisin |
|---|---|---|---|---|
| < 0,85 | 7 | 0,571 | 0,857 | 1,000 |
| 0,85 – 0,90 | 25 | 0,400 | 0,840 | 0,920 |
| 0,90 – 0,95 | 54 | 0,648 | 0,944 | 0,981 |
| ≥ 0,95 (quasi-copie) | 34 | 0,441 | 1,000 | 1,000 |

Sur les 32 tickets sans quasi-doublon (< 0,90), le LLM personnalisé classe juste dans 84 % des cas (non personnalisé 44 %). C'est une estimation sur peu de tickets (IC très large), et inférieure au score global. Le LLM a donc bien appris au-delà de la simple mémorisation (+40 points sur les tickets inédits), mais le chiffre de 93 % décrit surtout ce que l'on obtient quand le test ressemble à l'entraînement, ce qui est le cas de ce jeu de données mais pas nécessairement de demandes réelles futures.
3. **Un voisin le plus proche sans LLM fait mieux** (0,975) : sur ce jeu, la tâche est en grande partie de la recherche de ticket similaire. Le LLM apporte ce qu'une base de voisins n'apporte pas (réponse bornée aux 10 files, traitement d'un ticket sans équivalent, texte libre), mais ce n'est pas la solution la plus performante sur ce test, et je ne prétends pas le contraire.
4. **Par langue** (graine 0) : de 0,971 ; es 1,000 ; fr 0,937 ; en 0,920 ; **pt 0,748** (15 tickets). Le non personnalisé varie de 0,40 à 0,62.
5. **Le prompt avec définitions aide le modèle de base** (0,471 contre 0,391 en compact) ; c'est donc ce prompt qui sert de référence « non personnalisé », pour ne pas favoriser artificiellement la comparaison.

## Décisions et difficultés
| Constat (par la mesure) | Décision |
|---|---|
| Qwen3-8B sans personnalisation : F1 0,38 en validation ; le 27B encore moins bien sur le test (≈ 0,43) | Personnaliser par fine-tuning ; une taille de modèle plus grande n'aide pas (les files sont des conventions de l'opérateur) |
| TF-IDF : ≈ 0,67 en validation croisée ; un ticket « Visual Studio Code » peut être *Human Resources* | Étiquettes bruitées, plafond < 100 % attendu : le mesurer plutôt que le supposer |
| 1ʳᵉ série d'entraînement : validation qui monte encore (0,626 → 0,756), perte 0,12, **≈ 9 min par époque** avec le prompt complet | Interrompre avant l'entraînement final ; prompt compact pour le fine-tuning (≈ 2× plus rapide), plus d'époques et un taux plus élevé |
| lr 2e-4 × 6 époques : 0,829 ; lr 4e-4 × 4 époques : 0,893 | Retenu : 4e-4 × 4 (choix sur 96 tickets de validation, à ± 4 points près) |
| Un plus proche voisin donnait déjà 0,906 sur la validation | Mesurer et publier l'effet des quasi-doublons au lieu de présenter 93 % comme une généralisation |
| Mémoire GPU partagée avec le serveur vLLM (≈ 75 Go) | Mémoire du processus plafonnée à 30 Go : le service existant n'a pas été perturbé |

## Limites
- **Taille du test** : 120 tickets ; IC95 % de ± 5 points sur le F1 pondéré ; les classes à 1-3 tickets (*General Inquiry*, *Human Resources*, *Sales and Pre-Sales*, *Service Outages*) ont un F1 non significatif.
- **Un seul découpage** (graine 42) : la variabilité due au découpage n'a pas été mesurée, seulement celle de l'entraînement (3 graines) et du bootstrap.
- **Hyperparamètres choisis sur 96 tickets** de validation, avec 4 essais seulement ; un autre découpage pourrait préférer une autre configuration.
- **Quasi-doublons** : le test reflète la structure du jeu fourni. Pas d'évaluation sur de vraies demandes futures ni sur des groupes de traductions retenus hors entraînement (un découpage par groupe de doublons serait le contrôle le plus rigoureux).
- **Un seul modèle de base** (Qwen3-8B) ; le texte de l'énoncé cite Llama et Mistral à titre d'exemple. Pas de comparaison entre modèles open weights fine-tunés.
- Le modèle de comparaison de 27 milliards de paramètres est servi par vLLM avec décodage guidé ; ses sorties varient légèrement d'une exécution à l'autre (0,426 à 0,436 sur trois exécutions du notebook 04) ; la valeur publiée est celle de la dernière exécution (0,426).
- Prompts rédigés en français avec noms de files en anglais ; les définitions des files ont été écrites à partir des noms, avant tout entraînement, et pas affinées ensuite.
- Pas de calibration de la confiance ni de seuil de renvoi vers un humain ; pas d'API.
- Adaptateurs non versionnés (≈ 170 Mo chacun).

## Pistes
Découpage par groupes de quasi-doublons pour une estimation honnête de la généralisation ; fusion de l'adaptateur et de la recherche de voisins (exemples proches dans le prompt) ; validation croisée pour stabiliser le choix d'hyperparamètres ; seuil de confiance pour renvoi vers un opérateur ; comparaison avec Llama et Mistral ; API de service.
