# Guide du Modèle SMC Deep Learning 🧠📊

L'intelligence artificielle de ce Trading Bot intègre un réseau de neurones profond (Deep Learning) conçu pour apprendre les concepts institutionnels (SMC - Smart Money Concepts) et trader de manière autonome. Ce guide détaille son fonctionnement, sa structure, et comment déclencher son apprentissage.

## 1. Architecture du Système

Le système est divisé en plusieurs couches :

### A. Feature Engineering (L'extraction de données)
Le fichier `smc_dl_features.py` analyse les bougies brutes et en extrait **57 variables explicatives (features)** :
- **L'Action des prix (Price Action) :** Volatilité (ATR), taille des corps/mèches, ratios.
- **La Structure de Marché (Market Structure) :** Détection de BOS (Break of Structure), CHoCH (Change of Character), et des Swings High/Low.
- **Les Zones SMC :** Détection des Order Blocks (OB) et Fair Value Gaps (FVG), leur âge, leur taille, et la distance au prix actuel.
- **Contextes Session & Temps :** Variables cycliques (heure, jour) et sessions (Asia, London, NY).

### B. Le Réseau de Neurones (Multi-Layer Perceptron)
Défini dans `smc_dl_model.py`, c'est un réseau de type **Multi-Task Learning** (Apprentissage Multi-Tâches) développé avec PyTorch :
- **Input Layer :** 57 neurones (les features normalisées).
- **Hidden Layers :** 2 à 3 couches cachées (128 à 256 neurones) utilisant BatchNorm, ReLU, et Dropout.
- **Output Heads (Têtes de sortie) :**
  1. `tp_prob` (Probabilité de toucher le Take Profit avant le Stop Loss)
  2. `direction` (Probabilité Hausse vs Baisse, classification softmax)
  3. `risk_score` (Une note de risque globale pour le trade)

### C. Le Labeling Tri-Barrière (Triple-Barrier Method)
Pour s'entraîner, l'IA a besoin d'exemples. Le fichier `smc_dl_labels.py` regarde le futur pour chaque bougie de l'historique et génère une étiquette :
- Est-ce que le trade a touché `TP = 2.0 * ATR` avant `SL = 1.0 * ATR` ?
- Si oui, c'est une réussite (1), sinon un échec (0).

## 2. Déclencher l'Apprentissage (Training Loop)

Le script `run_smc_dl_loop.py` gère le cycle complet d'optimisation (Génération -> Apprentissage -> Backtest).

**Pour lancer l'entraînement, exécutez la commande suivante à la racine du projet :**

```bash
make train-dl
```

**Que fait cette commande ?**
1. Elle extrait les données de la base SQLite locale.
2. Elle calcule les 57 features et génère les labels temporels.
3. Elle entraîne le modèle PyTorch sur les mois historiques (ex: jusqu'en juin).
4. Elle teste le modèle via un backtest sur des mois vierges (ex: juin-août).
5. Si les résultats ne sont pas au rendez-vous (Win Rate < 50% ou Profit Factor < 1.3), le script modifie ses hyperparamètres (Taille du TP, Neurones, Learning Rate) et **recommence automatiquement**.

## 3. Le Dashboard Interactif

Une fois l'entraînement terminé, le réseau de neurones produit un rapport JSON officiel de ses prédictions.
Vous pouvez visualiser le "cerveau" de l'IA en temps réel en naviguant dans l'application Web :
1. Lancez l'interface locale (`make dev`).
2. Ouvrez le tiroir de navigation à gauche.
3. Cliquez sur **Model Dashboard**.

Vous y verrez :
- Le solde final et la courbe de croissance virtuelle de l'IA.
- Les jauges colorées des probabilités exactes de chaque trade.
- Une animation visuelle de la transmission synaptique (activation des couches) lorsque vous cliquez sur un trade.

## 4. Intégration en Production

Le fichier `smc_dl_m5_v1.py` enveloppe le modèle entraîné (fichiers `.pt` et `.npz`). Il est chargé dans le registre des stratégies du bot.
Dès que `make dev` tourne, si la stratégie `smc_dl_m5` est assignée à un bot actif, elle évaluera chaque nouvelle bougie en direct et enverra les signaux à MetaTrader 5 en respectant la gestion du spread du Broker.
