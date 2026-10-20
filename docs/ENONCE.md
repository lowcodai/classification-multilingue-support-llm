# Énoncé du projet (résumé)

**Contexte.** Un prestataire de support client assure le service pour des dizaines d'entreprises, en français, anglais, allemand et portugais (le jeu de données contient aussi de l'espagnol). Il reçoit chaque jour des milliers de messages (remboursements, problèmes techniques, questions produit…) que des agents humains lisent et classent à la main avant de les orienter vers la bonne équipe. Le volume rend cette approche trop lente.

**Mission (LLM Engineer).** Construire une solution de **catégorisation par un LLM open weights** (Llama, Mistral, etc.) qui classe automatiquement les messages dans les files (« queues ») du service support.

**Contraintes.**
1. Seules 5 colonnes sont utilisables : `subject`, `body`, `queue` (cible), `language`, `business_type`. Toutes les autres (tags, priorité, type, réponse…) sont interdites car inconnues lors d'une classification réelle.
2. **Comparer** deux versions du LLM : non personnalisée et personnalisée avec le jeu d'entraînement, avec le **F1-score** par classe `F1_j = 2·TP_j / (2·TP_j + FP_j + FN_j)` et le **F1 pondéré** `Σ_j α_j · F1_j` (α_j = proportion de la classe j dans le test).
3. Le **F1 pondéré du LLM personnalisé doit être ≥ 92 %**.
4. **Ensemble de test stratifié** : mêmes proportions de classes que le jeu initial.

**Données.** 600 demandes de support multilingues avec leur `queue` (10 classes).

**Étapes.**
1. Construire les prompts d'entraînement et de test à partir des 5 colonnes autorisées.
2. Personnaliser un LLM open weights (méthode au choix).
3. Évaluer les deux versions sur le même test ; **capture d'écran** des F1_j et du F1 pondéré pour les deux versions.

**Environnement.** Prévu : Cloud Sandbox (JupyterLab). Utilisé ici : **DGX Spark + JupyterLab**, comme pour les projets précédents.
