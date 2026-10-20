# Templates de prompts — classification multilingue des demandes de support

Ces fichiers sont chargés par le code ([`classif/prompts.py`](../classif/prompts.py)) ; aucun prompt n'est écrit en dur dans le code Python. **Les deux versions du LLM (non personnalisée et personnalisée) reçoivent exactement la même entrée** : seule change la valeur des poids (adaptateur LoRA).

| Fichier | Rôle | Variables | Étape |
|---|---|---|---|
| `01_system_classification.md` | Rôle de l'agent de pré-tri, liste des files, format de sortie imposé (nom exact de la file), demande traitée comme donnée | `{queues}` | 1 — Prompts d'entraînement / 3 — Évaluation |
| `02_user_ticket.md` | Message utilisateur : secteur d'activité, langue, objet, corps de la demande (les 4 seuls champs d'entrée autorisés) | `{business_type}`, `{language}`, `{subject}`, `{body}` | 1 |
| `queues.json` | Les 10 files avec une définition générique en une ligne, injectée dans `{queues}` | — | 1 |
| `reponse_invalide.txt` | Règle appliquée à une sortie hors liste (comptée comme erreur) | — | 3 |

Un exemple d'entraînement est la paire **(prompt système + prompt utilisateur) → nom exact de la file** ; les 5 colonnes autorisées (`subject`, `body`, `queue`, `language`, `business_type`) sont les seules lues du CSV (`usecols`).

Les définitions de `queues.json` ont été rédigées à partir du seul nom des files, avant tout entraînement, pour que le modèle de base ait une chance raisonnable ; elles ne contiennent pas d'indice tiré des tickets.
