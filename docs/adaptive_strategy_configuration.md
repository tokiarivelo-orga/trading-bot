# Configuration de la Stratégie XAUUSD Adaptive M1

Cette stratégie (`xauusd_snd_qm_structure_adaptive_m1`) possède 73 paramètres qui contrôlent le moteur de Supply & Demand, l'apprentissage dynamique, la gestion du risque, et la détection des régimes.

Voici la documentation complète des aspects cruciaux de cette configuration.

---

## 1. Moteur d'Apprentissage Actif (Online Learner)

Le cœur de la stratégie repose sur un modèle logistique à 31 variables (features) et un système de "bucketing" (catégorisation de l'état du marché).

| Paramètre | Défaut | Explication |
|-----------|--------|-------------|
| `learner_enabled` | `True` | Active ou désactive totalement l'apprentissage online. Si `False`, la stratégie utilise uniquement les règles fixes. |
| `learner_secure_r` | `0.2` | Objectif de gain minimum (en Multiple de R) qu'un trade doit atteindre pour être considéré comme une "victoire" (y=1) par le modèle. |
| `learner_horizon_bars` | `120` | Nombre maximal de bougies M1 pour qu'un trade atteigne son but. Passé ce délai, le setup expire. |
| `learner_min_model_samples`| `150` | Nombre minimum total d'exemples résolus avant que la régression logistique ne commence à émettre des prédictions actives (mode "Cold Start" avant cela). |
| `learner_half_life_samples`| `750.0` | Vitesse d'oubli (en nombre d'exemples). Les événements récents pèsent plus lourd que les anciens pour s'adapter vite aux changements de marché. |
| `learner_edge_margin` | `0.15` | Marge supplémentaire exigée sur la probabilité estimée pour accepter un trade, agissant comme un filtre de confiance. |

---

## 2. Détection de Régime et Macro-Confluence

La stratégie filtre les mauvais trades en exigeant de la confluence structurelle et momentum (avant même de demander l'avis du Learner).

| Paramètre | Défaut | Explication |
|-----------|--------|-------------|
| `min_confluence_score` | `2` | **Paramètre le plus important pour le volume de trade.** Score minimal requis (sur la base de la tendance HTF, RSI, qualité de session, etc.) pour ouvrir une position. *Augmenter à 3 ou 4 réduit drastiquement l'overtrading.* |
| `rsi_period` / `rsi_overbought` / `rsi_oversold` | `14` / `70` / `30` | Configuration du RSI. Un signal d'achat sur une zone alors que le RSI est > 70 pénalise le score de confluence. |
| `bb_period` / `bb_std_mult` | `20` / `2.0` | Paramètres des Bandes de Bollinger. Utilisé pour calculer la compression du prix et identifier si le marché "range" ou explose. |
| `zone_confluence_atr_radius`| `1.0` | Définit la "zone de recherche" (en ATR) pour compter combien de zones Supply/Demand supportent la position actuelle. |

---

## 3. Paramétrage des Sessions Cycliques (Or - XAUUSD)

L'algorithme évalue indépendamment les performances de chaque signal en fonction de la session de trading, car l'or réagit différemment selon la liquidité des grandes places boursières.

| Paramètre (UTC) | Heure | Conséquence de la Session |
|-----------------|-------|---------------------------|
| `session_asian_start_utc` | `0` à `8` | Session Asiatique. Souvent associée à des "ranges" serrés. Le score de confluence y est pénalisé par défaut. |
| `session_london_start_utc`| `8` à `16` | Session de Londres. Volatilité croissante, cassures de structures fortes. |
| `session_ny_start_utc` | `13` à `21` | Session de New York. Fort momentum et nouvelles économiques majeures. |
| **Overlap (Chevauchement)** | `13` à `16` | **Bonus de Confluence.** La superposition Londres/NY offre la meilleure liquidité, idéale pour les cibles de Take Profit éloignées. |

---

## 4. Gestion Adaptative du Risque (SL & TP)

Les niveaux de Stop Loss et Take Profit ne sont pas fixes ; ils "respirent" entre des bornes (Bands) que tu définis ici, et le Learner choisit le point idéal selon le comportement récent du marché.

### Stop Loss (SL)
| Paramètre | Défaut | Explication |
|-----------|--------|-------------|
| `sl_zone_buffer_atr_mult` | `0.15` | Marge de sécurité (buffer) de base ajoutée sous une zone (en multiple de l'ATR). |
| `sl_buffer_band_lo` | `0.6` | Le Learner ne peut jamais réduire le buffer de SL en dessous de 60% du buffer de base (protection). |
| `sl_buffer_band_hi` | `2.2` | Le Learner peut élargir le buffer jusqu'à 2.2x en cas de fortes "chasses aux stops" récentes. |

### Take Profit (TP)
| Paramètre | Défaut | Explication |
|-----------|--------|-------------|
| `tp1_target_rr` | `1.8` | Ratio Risk/Reward de base du TP1 (Scalp). |
| `tp_band_lo` / `tp_band_hi` | `0.6` / `1.6` | Limites dynamiques d'ajustement du TP. Le système peut diviser par presque 2 ou multiplier par 1.6 les distances de Take Profit selon l'excursion favorable constatée sur les trades précédents. |

---

## 5. Comment optimiser avec `walk_forward_optimizer.py`

Le script ajouté précédemment (`backend/scripts/walk_forward_optimizer.py`) est conçu pour ré-aligner les paramètres globaux (macro) listés ci-dessus. 

Il fera varier par exemple `min_confluence_score` (1, 2, 3), `bb_std_mult` (1.8, 2.0, 2.2), testera toutes les combinaisons sur les N derniers jours, et écrira la meilleure configuration dans la section `param_overrides` du fichier `.yaml` de la stratégie. 

Le Learner s'occupe de la micro-optimisation (SL/TP millimétrés) barre par barre, pendant que le script s'occupe de la macro-optimisation (sélectivité du bot) semaine par semaine.

---

## 6. Mécanique de l'IA : État (State) et Auto-Apprentissage

Il y a souvent une confusion entre la manière dont l'IA de ce bot s'entraîne (micro-optimisation) et la façon dont ses paramètres sont optimisés (macro-optimisation).

### Le modèle s'entraîne-t-il automatiquement ?
**OUI, à 100% et en temps réel.** Le cœur de l'IA est un modèle d'apprentissage en ligne (*Online Learning*). 
- Dès qu'un trade est pris, le système garde sa signature en mémoire (ses 31 features).
- Quelques heures plus tard, quand ce trade se clôture (touche le Stop Loss ou un Take Profit), la stratégie appelle automatiquement `learner.observe()`. 
- À ce moment précis, l'algorithme met immédiatement à jour les poids mathématiques du modèle (*Descente de Gradient Stochastique*).
- **Conséquence :** L'IA s'adapte sans aucune intervention humaine. Si une certaine configuration de S&D couplée à un RSI suracheté perd trois fois de suite aujourd'hui, l'IA va pénaliser cette combinaison et bloquer les prochains trades similaires dès demain.

### L'État de l'IA (State) est-il sauvegardé quelque part ?
Le modèle n'écrit pas un gros fichier de poids de réseau de neurones (`.h5` ou `.pkl`) sur le disque dur, car ce type d'I/O ralentirait énormément le trading haute fréquence. L'état (les poids de l'IA) vit en **RAM (Mémoire vive)**.

**Comment fait-il quand on redémarre le bot ? (Pre-Warming)**
Pour éviter que l'IA ne perde la mémoire si on relance le serveur, le bot utilise une technique de **Réchauffement (Pre-warming / Replay)** :
1. Au redémarrage, le moteur de trading charge instantanément les milliers de bougies des dernières semaines (historique de la base de données).
2. Il simule les trades du passé en vitesse accélérée sans passer d'ordres réels.
3. Le modèle réapprend instantanément son historique pour retrouver le **poids exact de son réseau neuronal** à la seconde près.
Le "state" est donc recréé dynamiquement et est toujours parfait.

### Pourquoi avoir besoin de la page "Optimizer & AI" alors ?
L'auto-apprentissage (*Online Learner*) gère les décisions microscopiques (placer le SL à tel prix, refuser tel trade en fonction de la session HMM, etc). 
Cependant, l'IA ne peut pas changer ses propres **règles de base** (ex: *L'IA ne peut pas décider toute seule de changer le paramètre des Bandes de Bollinger de 2.0 à 2.2*). C'est le but de la nouvelle page `walk_forward_optimizer` : elle t'assiste pour faire évoluer les "Garde-fous" macroscopiques dans lesquels l'IA a le droit d'opérer.
