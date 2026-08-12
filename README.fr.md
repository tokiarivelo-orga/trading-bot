> 🇬🇧 English version: [README.md](README.md)

# Bot de Trading IA — XAUUSD / XAGUSD / BTCUSD

Un bot de trading connecté à MT5 et assisté par IA. Entrées en M5 avec
confirmation sur les unités de temps supérieures, graphique façon TradingView,
stratégies générées à partir de documents PDF par une IA (Claude ou Ollama),
et auto-amélioration automatique toutes les 10 trades.

**Conception complète & feuille de route :** voir
[`IMPLEMENTATION_PLAN.fr.md`](IMPLEMENTATION_PLAN.fr.md).

## Organisation du dépôt

| Chemin | Contenu |
|--------|---------|
| `backend/` | Backend FastAPI — moteur, stratégies, couche IA, journal (modules hexagonaux) |
| `frontend/` | UI Next.js + Tailwind CSS + TypeScript — graphique, contrôles du bot, upload PDF, rapports |
| `gateway/` | Service passerelle MT5 — le **seul** code touchant MetaTrader5 (tourne sous Windows/Wine/VPS, voir `gateway/README.fr.md`) |
| `configs/` | Configuration d'exécution (symboles, plafonds de risque, fournisseurs IA, actualités) |
| `Makefile` | Commandes de dev canoniques — setup, serveurs de dev, vérifications, BDD, docker (`make help`) |
| `.claude/` | Skills et réglages Claude Code |

## Démarrage rapide (développement)

Tout passe par le `Makefile` racine — lancez `make help` pour la liste complète.

```bash
make setup             # backend (uv sync) + frontend (pnpm install) + gateway (uv sync) + .env
make dev               # backend :8000 + frontend :3000 + gateway :8787 — Ctrl-C arrête tout

# ou individuellement :
make dev-backend       # FastAPI avec rechargement auto — http://localhost:8000
make dev-frontend      # serveur de dev Next.js — http://localhost:3000
make dev-gateway       # passerelle MT5 sous Wine — http://localhost:8787
```

La passerelle nécessite un terminal MT5 en cours d'exécution sous Wine
(développement) ou sur un VPS Windows (trading réel). Voir
[`gateway/README.fr.md`](gateway/README.fr.md) pour les instructions complètes.

**Documentation de l'API backend** (une fois `make dev-backend` lancé) :
interface Swagger interactive sur <http://localhost:8000/docs>, ReDoc sur
<http://localhost:8000/redoc>, schéma brut sur
<http://localhost:8000/openapi.json> (ou `make openapi`). Chaque route est
entièrement typée et documentée — voir `backend/src/*/api/schemas.py`.

Sous le capot : le backend est en Python 3.12 via `uv`, le frontend en Next.js
via `pnpm` (version épinglée dans `frontend/package.json`), la passerelle
tourne avec Python 3.12 Windows sous Wine.

## Vérifications

```bash
make check             # lint (ruff + oxlint) + tests backend + build frontend
```

Individuels : `make lint`, `make test`, `make build-frontend` — voir
`make help`.

## Intelligence Artificielle & Deep Learning (SMC)

Le projet intègre un réseau de neurones multicouche qui apprend les concepts de la Smart Money (Order Blocks, FVG) de façon autonome. La documentation complète de son architecture est disponible dans [`docs/SMC_DEEP_LEARNING.md`](docs/SMC_DEEP_LEARNING.md).

**Pour déclencher l'entraînement du modèle IA :**
```bash
make train-dl          # Extrait les données de MT5, entraîne le réseau, et lance le backtest de validation
```

**Pour visualiser les décisions du modèle :**
Ouvrez le *Model Dashboard* dans l'application Web (`make dev` puis via le menu latéral) pour une visualisation interactive de l'activation des neurones pour chaque trade !

## Modèle de sécurité (à ne jamais affaiblir)

- Tout démarre en **mode papier** (`configs/app.yaml : mode: paper`) — les
  ordres sont simulés en mémoire et n'atteignent jamais MT5. Passer à
  `mode: live` envoie de vrais ordres via la passerelle vers votre compte
  réel ; avant cela, le bouton **AutoTrading** du terminal MT5 (barre
  d'outils, ou Outils → Options → Expert Advisors → « Autoriser le trading
  algorithmique ») doit être activé, sinon chaque ordre est rejeté avec le
  code retour `10027` — voir
  [`gateway/README.fr.md`](gateway/README.fr.md#configuration-du-terminal-les-deux-options).
- Les plafonds de risque vivent dans `configs/risk.yaml` et appartiennent à
  l'utilisateur — l'IA et le code généré ne les modifient jamais.
- Les stratégies générées par IA tournent en sandbox : pas d'E/S, pas de
  réseau, pas d'accès au courtier.
- Coupe-circuits au niveau du moteur : limite de perte journalière, pause
  après pertes consécutives, bouton d'arrêt d'urgence (kill switch).
