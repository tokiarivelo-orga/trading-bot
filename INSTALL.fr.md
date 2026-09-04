> 🇬🇧 English version: [INSTALL.md](INSTALL.md)

# Guide d'installation — Bot de Trading IA

Ceci est la référence complète pour faire tourner le bot sur une machine
qui n'est pas votre propre checkout de dev — un PC personnel, un VPS, ou un
serveur que vous administrez pour quelqu'un d'autre. Le
[`README.md`](README.md#installing-on-another-machine) racine en donne la
version courte ; ce fichier est ce vers quoi elle pointe.

## Vue d'ensemble

Il y a trois façons de faire tourner ce projet :

- **L'installateur CLI** (`installer/install.py`) — met en place toute la
  pile, y compris la passerelle MT5, sur n'importe quel OS. Recommandé
  pour la plupart des gens.
- **Docker** (`docker-compose.prod.yml`) — backend + frontend uniquement,
  via des images prêtes à l'emploi, pour quand la passerelle MT5 tourne
  déjà ailleurs.
- **Installation manuelle de dev** — pour celles et ceux qui contribuent
  au code lui-même ; voir [`LAUNCH.md`](LAUNCH.md).

| Votre situation | Chemin |
|---|---|
| Je veux tout faire tourner sur ma machine ou un VPS, passerelle incluse | **Chemin 1 — installateur CLI** |
| J'ai déjà (ou je vais installer séparément) une passerelle MT5 ailleurs et je veux juste l'application | **Chemin 2 — Docker** |
| Je vais travailler sur le code lui-même | **Chemin 3 — installation manuelle de dev** (`LAUNCH.md`) |

## Prérequis

### Système d'exploitation

| OS | Installateur CLI (Chemin 1) | Docker (Chemin 2) |
|---|---|---|
| **Windows 10/11** | Toute la pile, nativement — pas de Wine. Le paquet Python `MetaTrader5` ne publie que des wheels Windows, donc c'est la façon la plus simple de faire tourner la passerelle. | backend + frontend uniquement |
| **Linux** | Toute la pile — l'installateur peut provisionner automatiquement Wine et un terminal MT5 sous Wine pour vous (invite avec opt-in ; installe `wine64`/`wine32`/`winetricks` via `apt`). | backend + frontend uniquement |
| **macOS** | Pas une cible prise en charge : la logique d'autostart/désinstallation de l'installateur ne distingue que « Linux » de « tout le reste est Windows » (`platform.system()`), donc macOS reçoit des invites rédigées pour Windows et de mauvaises instructions de désinstallation. Utilisez Docker à la place, ou les étapes manuelles de la passerelle sur une autre machine. | backend + frontend uniquement — Docker Desktop pour Mac fonctionne bien ; la passerelle elle-même doit quand même tourner sous Windows, sous Wine (Linux), ou sur un VPS joignable par le réseau |

### Logiciels prérequis

| Chemin | Nécessite |
|---|---|
| Installateur CLI | Python 3.12+, [`uv`](https://docs.astral.sh/uv), Node.js + `pnpm`. Le script de l'installateur lui-même ne dépend que de la bibliothèque standard et tourne avant que tout ceci n'existe — mais il vérifie que `uv`/`node`/`pnpm` sont bien sur le PATH dès le départ, et affiche la commande d'installation adaptée pour ce qui manque avant de continuer. |
| Docker | Docker Engine + le plugin `docker compose` (syntaxe v2 — invoqué en `docker compose -f docker-compose.prod.yml ...`). Rien d'autre ; les images embarquent tout. |
| Tous les chemins | Un **compte démo MT5** chez un courtier (numéro de login, mot de passe, nom du serveur — p. ex. `MetaQuotes-Demo`), gratuit chez n'importe quel courtier MT5. Saisi plus tard via le panneau **MT5 Account** de l'UI de l'application — jamais dans `.env`, une invite de l'installateur, ou un fichier Docker ; les identifiants vont directement dans le trousseau (keyring) du système d'exploitation. |

### Matériel & réseau

- **Espace disque** : les dépendances du backend incluent `torch` — prévoyez
  quelques Go libres rien que pour `uv sync`. Un préfixe Wine + une
  installation du terminal MT5 (Chemin 1 sous Linux) nécessitent leur
  propre espace en plus, comparable à une installation native de MT5 sous
  Windows.
- **Réseau/latence** : le trading en temps réel demande une faible latence
  vers le serveur de trading de votre courtier — même recommandation que
  l'option VPS de `gateway/README.md` : pinguez le nom de serveur affiché
  dans la fenêtre de connexion MT5 et choisissez une région de VPS avec le
  temps d'aller-retour le plus bas avant de sortir du mode papier.

## Chemin 1 : installateur CLI (recommandé pour la plupart des gens)

`installer/install.py` est un script Python qui ne dépend que de la
bibliothèque standard — il tourne avant même que `uv sync`/`pnpm install`
n'aient jamais été lancés. Il écrit `.env` et `configs/accounts.yaml`,
provisionne éventuellement Wine + un terminal MT5 sous Linux, et enregistre
éventuellement des services de démarrage automatique.

### 1. Récupérer le code

- Téléchargez une archive source depuis la page **GitHub Releases** du
  dépôt (publiée par `.github/workflows/release.yml` sur un tag de
  version — `trading-bot-<version>.zip` / `.tar.gz`) et extrayez-la, ou
- faites un `git clone` du dépôt.

### 2. Lancer l'installateur

```bash
python3 installer/install.py      # Linux/macOS
python installer\install.py       # Windows
```

Il vérifie d'abord que `uv`, `node` et `pnpm` sont sur le PATH, en
affichant une commande d'installation adaptée à votre OS pour ce qui
manque (et en s'arrêtant, sauf avec `--dry-run`).

Options :

| Option | Effet |
|---|---|
| `--dry-run` | Affiche chaque fichier qu'il écrirait et chaque commande qu'il lancerait, sans rien écrire ni exécuter. |
| `--reconfigure` | Relance l'assistant avec chaque valeur par défaut pré-remplie depuis `installer/state.json` et le `.env`/`configs/accounts.yaml` existants, au lieu des valeurs par défaut codées en dur. Utile plus tard pour ajouter un compte ou changer une réponse précédente. |
| `--non-interactive` | Accepte chaque valeur par défaut sans rien demander — pour un usage scripté/CI. |
| `--uninstall` | Voir [Désinstaller](#désinstaller) ci-dessous. |

### 3. Répondre aux questions

Dans l'ordre :

1. **Répertoire d'installation** — par défaut, l'endroit où vous avez
   extrait/cloné le dépôt.
2. **Par compte MT5** (demande « Add an MT5 account? », puis « Add another
   MT5 account? » jusqu'à ce que vous refusiez) :
   - **Id du compte** (slug) — entrez `default` pour le compte principal
     déjà présent dans `configs/accounts.yaml` ; toute autre valeur devient
     une nouvelle entrée. Réutiliser un id déjà existant est sans risque —
     il est ignoré avec un avertissement, jamais écrasé.
   - **Label** — un nom lisible par un humain.
   - **Mode** — `paper` ou `live` (par défaut `paper` ; toute autre valeur
     retombe aussi sur `paper`).
   - **Hôte de la passerelle** — par défaut `127.0.0.1`.
   - **Port de la passerelle** — par défaut `8787` pour le premier compte,
     puis le prochain port libre pour chaque compte supplémentaire.
   - **Chemin vers le `terminal64.exe` de ce compte** — laissez vide pour
     vous attacher à n'importe quel terminal déjà en cours d'exécution
     (suffisant pour un seul compte). Chaque compte au-delà du premier a
     besoin de son propre chemin ici, sinon sa passerelle peut
     silencieusement déconnecter le terminal d'un autre compte.
3. **Provisionnement automatique de Wine + du terminal MT5** — Linux
   uniquement, et demandé seulement si un compte a encore besoin d'un
   chemin de terminal. Installe `wine64`/`wine32`/`winetricks` via `apt`
   et lance `winetricks corefonts` (nécessite `sudo`). Répondez non et
   l'installateur affiche simplement les mêmes commandes à lancer vous-même
   — l'option A de `gateway/README.fr.md`.
4. **Démarrage automatique** — sous Linux, cela enregistre et active des
   services systemd `--user` pour le backend, le frontend, et la
   passerelle + le terminal de chaque compte, plus un `loginctl
   enable-linger` pour qu'ils survivent à une déconnexion/un redémarrage.
   Sous Windows, cela ne fait qu'enregistrer votre intention dans
   `installer/state.json` — enregistrer réellement des tâches planifiées
   nécessite une étape PowerShell séparée (affichée à la fin), car ce
   n'est pas sûr de s'auto-élever depuis un script.
5. **Confirmation finale** — un écran récapitulatif listant chaque
   modification de `.env`/`configs/accounts.yaml`, les éventuelles
   commandes Wine, et le plan de démarrage automatique. La réponse par
   défaut est **non** — il faut taper `y` pour continuer.
   (`--non-interactive` saute cette confirmation et continue
   automatiquement ; `--dry-run` ne continue jamais.)

### 4. Ce qui est écrit

- **`.env`** — créé à partir de `.env.example` s'il n'existe pas encore,
  avec un `TB_GATEWAY_SHARED_SECRET` fraîchement généré. Chaque compte
  au-delà du tout premier reçoit son propre
  `TB_GATEWAY_SHARED_SECRET_<ID>` généré. Pour tout ce que vous pourriez
  vouloir régler d'autre dans `.env` (clés de fournisseurs IA, identifiants
  d'alerte, `TB_APP_PASSWORD`) et comment obtenir chaque valeur, voir
  [`SECRETS.fr.md`](SECRETS.fr.md).
- **`configs/accounts.yaml`** — un nouveau bloc ajouté par nouveau compte,
  respectant exactement le format existant du fichier. Un id déjà présent
  n'est jamais modifié.
- **Si vous avez accepté** : le préfixe Wine + le terminal MT5, et/ou les
  services de démarrage automatique —
  - *Linux* : des unités systemd `--user` sous
    `~/.config/systemd/user/` : `trading-bot.target`,
    `trading-bot-backend.service`, `trading-bot-frontend.service`, et par
    compte `trading-bot-gateway@<id>.service` /
    `trading-bot-terminal@<id>.service`.
  - *Windows* : lancez la commande affichée par l'installateur à la fin :
    ```powershell
    powershell -ExecutionPolicy Bypass -File installer\services\windows\install_services.ps1 -RepoRoot <répertoire d'installation> -StateJsonPath installer\state.json
    ```
    Cela enregistre des tâches planifiées `TradingBot-*`, chacune pointant
    vers un lanceur `.cmd` généré sous
    `installer\services\windows\generated\`.

    Remarque : les scripts de service Windows sont marqués **NON VÉRIFIÉS**
    (`UNVERIFIED`) dans leurs propres commentaires d'en-tête — écrits en
    suivant le comportement documenté du module PowerShell
    `ScheduledTasks`, mais pas encore exécutés sur une vraie machine
    Windows. Testez sur une VM jetable avant de vous y fier pour un compte
    réel.

### 5. Finaliser l'installation et vérifier que tout fonctionne

```bash
make setup       # uv sync + pnpm install, si ce n'est pas déjà fait
make db-upgrade  # applique les migrations de la base de données
python3 installer/manage.py status   # état de chaque service enregistré
```

`installer/manage.py status|start|stop|restart` pilote `systemctl --user`
sous Linux, ou les tâches planifiées `TradingBot-*` sous Windows ; si rien
n'est encore enregistré, il l'indique clairement au lieu d'une erreur
brute.

Vous n'avez pas activé le démarrage automatique ? Démarrez tout à la main
à la place : `make dev`.

### 6. La seule étape manuelle : se connecter à MT5

L'installateur ne touche jamais aux identifiants du courtier — c'est
délibéré (voir la règle « Installer & distribution » du `CLAUDE.md`). Une
fois la pile lancée, ouvrez le panneau **MT5 Account** de l'UI de
l'application et connectez-vous avec votre login/mot de passe/serveur
démo (ou réel). C'est aussi à ce moment-là que vous activez **Algo
Trading** et ajoutez vos symboles au Market Watch dans le terminal
lui-même — voir la section « Configuration du terminal » de
`gateway/README.fr.md` pour les étapes exactes, puisque rien de tout ça
n'est scriptable depuis l'extérieur de MT5.

## Chemin 2 : Docker (backend + frontend uniquement)

Pour quand la passerelle MT5 tourne déjà ailleurs — un VPS Windows
séparé, ou une passerelle sous Wine que vous avez configurée à la main ou
via les étapes compte/Wine du Chemin 1. `docker-compose.prod.yml` ne
lance jamais la passerelle elle-même : elle n'est délibérément pas
conteneurisée (les soucis de rendu GUI/mise en veille de Wine dans un
conteneur n'en valent pas la peine).

```bash
cp .env.example .env
# éditez .env : réglez TB_GATEWAY_URL vers l'endroit où tourne réellement votre passerelle
docker compose -f docker-compose.prod.yml up -d
# ou : make docker-up-prod
```

Cela récupère `${DOCKERHUB_NAMESPACE:-tradingbot}/backend` et
`.../frontend` en version `${IMAGE_TAG:-latest}` — backend sur le port
8000, frontend sur le 3000, avec `./configs` monté en lecture seule et
`./data` en lecture-écriture dans le conteneur backend.

`tradingbot` est actuellement un espace de noms Docker Hub
**provisoire** (voir les marqueurs `<!-- TODO -->` dans
`docker-compose.prod.yml` et `.github/workflows/docker-publish.yml`) —
tant que le véritable espace de noms n'est pas confirmé et que
`DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` ne sont pas configurés comme
secrets du dépôt, `docker compose ... pull` peut renvoyer une 404. En
attendant, construisez les images en local :

```bash
docker build -f backend/Dockerfile -t tradingbot/backend:local backend
docker build -f frontend/Dockerfile -t tradingbot/frontend:local frontend
DOCKERHUB_NAMESPACE=tradingbot IMAGE_TAG=local docker compose -f docker-compose.prod.yml up -d
```

Configurer la passerelle elle-même est une étape séparée — suivez
`gateway/README.fr.md`, ou lancez l'installateur du Chemin 1 juste pour
les invites compte/Wine. Si vous utilisez l'installateur de cette façon,
**refusez l'invite de démarrage automatique** : le démarrage automatique
de l'installateur enregistre aussi des services backend et frontend, qui
entreraient en conflit avec les conteneurs sur les mêmes ports.

## Chemin 3 : installation manuelle de dev

Pour contribuer au code lui-même. Suivez le guide pas à pas de
[`LAUNCH.md`](LAUNCH.md) — ce sont les mêmes étapes que l'installateur
exécute pour vous, détaillées une par une, avec son propre tableau de
prérequis et sa propre section de dépannage. Non dupliqué ici.

## Configurer plusieurs comptes, papier vs. réel

Chaque entrée de `configs/accounts.yaml` est un login MT5, joint via son
propre processus de passerelle — le paquet Python `MetaTrader5` ne
supporte qu'un seul compte connecté par processus OS, donc N comptes
signifient N processus de passerelle et, au-delà du premier, N
installations de terminal séparées. Par compte :

- `mode: paper | live` — le mode papier simule les ordres en mémoire et
  n'atteint jamais MT5 ; le mode live envoie de vrais ordres. Le dépôt est
  livré avec `default` en mode `live` et `demo-1` en mode `paper` —
  vérifiez `configs/accounts.yaml` avant de passer en réel sur le mauvais
  compte.
- son propre `gateway_url` (hôte:port) et `gateway_shared_secret_env` (un
  nom de variable `.env` distinct — jamais un secret partagé entre
  comptes).
- `mt5_terminal_path` (préféré — fonctionne aussi bien pour une
  installation Windows native que pour Wine) ou l'ancien
  `mt5_terminal_subpath` (résolu par rapport à un préfixe Wine). Chaque
  compte au-delà du premier a besoin que l'un des deux soit renseigné,
  sinon sa passerelle s'attache à n'importe quel terminal déjà en cours
  d'exécution et peut silencieusement déconnecter un autre compte.

L'assistant de l'installateur gère tout ça pour vous dans la boucle des
comptes (étape 2 du Chemin 1 ci-dessus) : chaque compte supplémentaire
reçoit le prochain port de passerelle libre, sa propre variable de secret
générée, et sa propre invite de chemin de terminal. Relancez
`installer/install.py --reconfigure` à tout moment pour ajouter d'autres
comptes — il pré-remplit les réponses précédentes et ne touche jamais à
un compte existant portant le même id.

## Désinstaller

```bash
python3 installer/install.py --uninstall
```

- **Linux** : arrête, désactive et supprime chaque unité systemd `--user`
  `trading-bot.target` / `trading-bot-*.service` enregistrée par
  l'installateur.
- **Windows** : ne désinstalle pas directement (supprimer des tâches
  planifiées n'est pas sûr à auto-élever depuis un script) — il affiche
  la commande à lancer à la place :
  ```powershell
  powershell -ExecutionPolicy Bypass -File installer\services\windows\uninstall_services.ps1 -RepoRoot <répertoire d'installation>
  ```
  qui supprime chaque tâche `TradingBot-*` et, avec `-RepoRoot`, les
  lanceurs `.cmd` générés sous
  `installer\services\windows\generated\` — certains d'entre eux
  embarquent une copie d'un secret de passerelle, donc ça vaut le coup de
  les nettoyer.

Dans tous les cas, **`.env`, `configs/accounts.yaml`, et vos données**
(`backend/data/`, journal, stratégies générées, modèles entraînés) **ne
sont jamais touchés par la désinstallation.** Supprimez-les à la main si
vous voulez repartir sur une base réellement propre.

## Dépannage

| Symptôme | Cause / correctif |
|---|---|
| L'installateur s'arrête avec « Missing required tooling » | `uv`/`node`/`pnpm` ne sont pas sur le PATH — il affiche la commande d'installation exacte pour votre OS (p. ex. `curl -LsSf https://astral.sh/uv/install.sh \| sh` sous Linux, `winget install --id astral-sh.uv -e` sous Windows, `brew install uv` sous macOS). Installez, puis relancez `installer/install.py`. |
| J'ai refusé l'invite de provisionnement automatique de Wine — et maintenant ? | `.env`/`configs/accounts.yaml` ont quand même été écrits, rien n'est perdu. Suivez à la main l'option A de `gateway/README.fr.md`, ou relancez `installer/install.py --reconfigure` et acceptez cette fois-ci (vos autres réponses sont pré-remplies). |
| Une tâche planifiée Windows est enregistrée mais le processus ne tourne pas | Vérifiez le lanceur généré sous `installer\services\windows\generated\<nom>.cmd` — c'est la commande réellement exécutée par la tâche. Lancez-le directement dans un terminal pour voir la vraie erreur. `python installer\manage.py status` affiche l'état de chaque tâche. |
| `terminal_connected: false` (dans le `/health` de la passerelle, ou `make doctor`) | Le terminal MT5 n'est pas ouvert ou pas connecté — démarrez-le et connectez-vous ; la passerelle se reconnecte au prochain `/login`. Même cause racine et même correctif que dans le tableau de dépannage de `LAUNCH.md`. |
| `docker compose ... pull` échoue / image introuvable | `tradingbot` est encore un espace de noms Docker Hub provisoire — les vraies images ne sont peut-être pas encore publiées. Construisez-les en local à la place (voir la remarque Docker Hub du Chemin 2 ci-dessus), ou réglez `DOCKERHUB_NAMESPACE`/`IMAGE_TAG` pour pointer vers des images qui existent. |
| Erreurs `502`/`401` juste après l'installation | Mêmes causes que le tableau de dépannage de `gateway/README.fr.md` — mauvais login/mot de passe/nom de serveur, ou `TB_GATEWAY_SHARED_SECRET` qui ne correspond pas entre le backend et la passerelle. |
| Les ordres échouent avec le code retour `10027` | Algo Trading n'est pas activé dans le terminal — voir l'étape 2 de « Configuration du terminal » dans `gateway/README.fr.md`. Ce n'est jamais quelque chose que l'installateur peut faire à votre place ; ça doit se passer dans l'UI du terminal MT5 lui-même. |

## FAQ

**Mon mot de passe de courtier est-il en sécurité ?** Il ne va jamais dans
`.env`, une invite de l'installateur, ou un fichier Docker. Il n'est saisi
que via le panneau MT5 Account de l'UI de l'application, chiffré avec une
clé conservée dans le trousseau (keyring) du système d'exploitation, et
transmis à la passerelle uniquement au moment de la connexion.

**Puis-je faire tourner ça sans surveillance sur un VPS ?** Oui — c'est
exactement à ça que servent les services de démarrage automatique. Chacun
redémarre en cas de plantage (Windows : jusqu'à 999 redémarrages, espacés
d'une minute ; Linux : comportement de redémarrage systemd sur échec) et
revient après un redémarrage de la machine ; `loginctl enable-linger`
(Linux) garde les services utilisateur actifs sans session de connexion
ouverte.

**Windows ou Linux — lequel choisir ?** Windows natif évite complètement
Wine — le chemin le plus simple, sans les caprices de préfixe. Linux
fonctionne bien aussi (c'est comme ça que le projet lui-même a été
développé) et l'installateur provisionne Wine automatiquement pour vous
si vous l'acceptez ; ça ajoute juste une pièce mobile en plus. Dans tous
les cas, pour le trading réel, un VPS proche des serveurs de trading de
votre courtier compte plus que le choix de l'OS — voir l'option B de
`gateway/README.fr.md`.

**Ai-je besoin de `git` ?** Non — téléchargez une archive de release
(Chemin 1, étape 1) et extrayez-la si vous préférez ne pas cloner.

**Puis-je ajouter un deuxième (ou troisième) compte de courtier plus
tard ?** Oui — relancez `installer/install.py --reconfigure` à tout
moment ; voir « Configurer plusieurs comptes » ci-dessus.
