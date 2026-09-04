> 🇬🇧 English version: [SECRETS.md](SECRETS.md)

# Secrets & variables de configuration — Bot de Trading IA

Ceci est la référence complète pour chaque secret et variable de
configuration utilisés par ce projet — à quoi sert chacun, s'il est
obligatoire, où obtenir une vraie valeur, et comment le régler concrètement
sur chaque plateforme prise en charge par ce dépôt (Linux natif, Windows
natif, Docker, GitHub Actions).

Ce document part du principe que vous mettez en place une installation, pas
forcément que vous contribuez au code. Pour le déroulé pas à pas de
l'installation elle-même (lancer l'installateur, répondre à ses invites,
Docker vs natif, désinstaller), voir [`INSTALL.fr.md`](INSTALL.fr.md) — ce
document ne répète pas ce contenu, il ne couvre que les secrets/valeurs de
configuration en détail.

## La règle la plus importante

**Votre login MT5, votre mot de passe et le nom de votre serveur de
courtier ne vont jamais dans `.env`, dans un fichier sous `configs/`, ni
dans une invite de l'installateur.** Ils sont saisis une seule fois, via le
panneau **MT5 Account** de l'UI de l'application en cours d'exécution, qui
les envoie à `POST /accounts/{account_id}/broker/connect`
(`backend/src/broker/api/routes.py`). À partir de là, ils sont chiffrés
avec une clé conservée dans le trousseau (keyring) de votre système
d'exploitation et transmis à la passerelle uniquement au moment de la
connexion — ils ne sont jamais journalisés, jamais renvoyés dans une
réponse d'API, et jamais écrits dans un fichier de configuration où que ce
soit dans ce dépôt. Si un jour un guide, un script ou une invite vous
demande de mettre un mot de passe de courtier dans un fichier texte, ce
n'est pas le fonctionnement de ce projet — arrêtez-vous et utilisez le
panneau de l'UI à la place.

Tout le reste dans ce document — clés d'API, secrets partagés, identifiants
d'alerte — est un type de valeur différent et *va* effectivement dans
`.env` ou dans le magasin de secrets propre à une plateforme, comme décrit
ci-dessous.

---

## 1. Configuration backend principale (`.env`, préfixe `TB_`)

Ce sont des champs de `Settings` dans
`backend/src/shared/config/settings.py`, chargés depuis le fichier `.env`
à la racine du dépôt (via `pydantic-settings`, préfixe `TB_`).
`.env.example` est le gabarit — copiez-le en `.env` et remplissez ce dont
vous avez besoin ; chaque champ a une valeur par défaut fonctionnelle sauf
indication contraire.

| Variable | Rôle | Obligatoire ? | Défaut | Comment obtenir une valeur | Plateformes |
|---|---|---|---|---|---|
| `TB_DATABASE_URL` | URL de base de données SQLAlchemy async | Optionnel | `sqlite+aiosqlite:///./data/trading.db` | Laissez tel quel sauf si vous migrez vers une autre base de données — choix personnel | Toutes |
| `TB_GATEWAY_URL` | URL de base de la passerelle du compte principal | Optionnel | `http://127.0.0.1:8787` | Choix personnel — doit correspondre à l'endroit où écoute réellement la passerelle de ce compte | Toutes |
| `TB_GATEWAY_SHARED_SECRET` | Secret partagé envoyé par le backend dans l'en-tête `X-Gateway-Secret` à la passerelle du compte principal | Obligatoire pour toute installation non strictement locale | aucune (vide = la passerelle ignore la vérification, dev local uniquement) | **Auto-généré.** L'installateur en génère un automatiquement (`secrets.token_hex(32)`) ; pour en générer un vous-même : `openssl rand -hex 32` (c'est aussi ce qu'utilise `make env`) | Toutes |
| `TB_GATEWAY_SHARED_SECRET_<COMPTE>` (p. ex. `TB_GATEWAY_SHARED_SECRET_DEMO_1`) | Idem, un par compte MT5 supplémentaire au-delà du premier | Obligatoire par compte supplémentaire | aucune | Même mécanisme — générez une valeur aléatoire distincte par compte, jamais réutilisée | Toutes |
| `TB_FRONTEND_PORT` | Port du frontend en dev/prod | Optionnel | `3000` | Choix personnel | Linux (Makefile, unité systemd), Windows (lanceur `.cmd`), non lu par Docker (le port est fixe dans `docker-compose.prod.yml`) |
| `TB_WINEPREFIX` | Préfixe Wine contenant le terminal MT5 + le Python Windows | Optionnel | `$(HOME)/.mt5` | Choix personnel, ou acceptez l'invite de l'installateur | Linux/Wine uniquement |
| `TB_ANTHROPIC_API_KEY` | Clé d'API Anthropic, pour le fournisseur IA `claude` | Optionnel (seulement si `claude` est sélectionné dans `configs/ai.yaml` et qu'aucune clé n'est définie sur la page Settings) | aucune | [console.anthropic.com](https://console.anthropic.com) | Toutes (backend uniquement) |
| `TB_OPENAI_API_KEY` | Clé d'API OpenAI, pour le fournisseur `openai` | Optionnel | aucune | [platform.openai.com](https://platform.openai.com) | Toutes |
| `TB_GEMINI_API_KEY` | Clé d'API Google Generative Language, pour le fournisseur `gemini` | Optionnel | aucune | Google AI Studio (ou un projet GCP avec l'API Generative Language activée) | Toutes |
| `TB_MISTRAL_API_KEY` | Clé d'API Mistral, pour le fournisseur `mistral` | Optionnel | aucune | La console « La Plateforme » de Mistral | Toutes |
| `TB_GROQ_API_KEY` | Clé d'API Groq, pour le fournisseur `groq` | Optionnel | aucune | La console de Groq | Toutes |
| `TB_DEEPSEEK_API_KEY` | Clé d'API DeepSeek, pour le fournisseur `deepseek` | Optionnel | aucune | La console de la plateforme DeepSeek | Toutes |
| `TB_XAI_API_KEY` | Clé d'API xAI (Grok), pour le fournisseur `xai` | Optionnel | aucune | La console de xAI | Toutes |
| `TB_OLLAMA_URL` | URL d'un serveur Ollama local, pour le fournisseur `ollama`/« Hermes Agent » | Optionnel | `http://127.0.0.1:11434` | Choix personnel — pointe vers votre propre installation Ollama, aucune inscription nécessaire | Toutes |
| `TB_CLAUDE_CODE_BINARY` | Chemin/nom du binaire CLI `claude`, pour le fournisseur `claude_code` | Optionnel | `claude` | Choix personnel — s'appuie sur votre propre abonnement via `claude login`, **pas** sur `TB_ANTHROPIC_API_KEY` | Toutes |
| `TB_CLAUDE_CODE_EXTRA_ARGS` | Options CLI supplémentaires pour le fournisseur `claude_code` (p. ex. `--agent`) | Optionnel | vide | Choix personnel | Toutes |
| `TB_CLAUDE_CODE_TIMEOUT_S` | Délai d'expiration, en secondes, pour les appels du fournisseur `claude_code` | Optionnel | `480.0` | Choix personnel — augmentez-le si vos appels `claude_code` sont interrompus ; notez que le délai du proxy du frontend doit rester supérieur à cette valeur (`frontend/next.config.ts`) | Toutes |
| `TB_OPENCLAW_URL` | URL de base d'une instance OpenClaw (intégration bêta/non vérifiée) | Optionnel (obligatoire avec la clé ci-dessous si `openclaw` est sélectionné) | aucune | Choix personnel — pointe vers votre propre déploiement OpenClaw | Toutes |
| `TB_OPENCLAW_API_KEY` | Clé d'API pour cette instance OpenClaw | Optionnel, associé à l'URL ci-dessus | aucune | Ce qui délivre les clés pour votre déploiement OpenClaw — spécifique à chaque opérateur | Toutes |
| `TB_FOREXFACTORY_CALENDAR_URL` | URL de base du calendrier d'actualités ForexFactory | Optionnel | `https://nfs.faireconomy.media` | Choix personnel — rarement modifié | Toutes |
| `TB_FINNHUB_CALENDAR_URL` | URL de base de l'API de calendrier économique Finnhub | Optionnel | `https://finnhub.io/api/v1` | Choix personnel — rarement modifié | Toutes |
| `TB_FINNHUB_API_KEY` | Clé d'API Finnhub | Optionnel — obligatoire seulement si `configs/news.yaml: calendar.source` vaut `finnhub` (le défaut `forexfactory` ne nécessite aucune clé) | aucune | [finnhub.io](https://finnhub.io) — clé du niveau gratuit depuis leur tableau de bord après inscription | Toutes |
| `TB_APP_PASSWORD` | Mot de passe unique protégeant toute l'application, sauf `/health` et `/auth/*` | Optionnel, mais **fortement recommandé** dès que l'application est accessible depuis autre chose que localhost | vide (aucune connexion requise) | Vous le choisissez vous-même — un mot de passe simple, pas quelque chose à récupérer ailleurs. **Voir la section « Notes de sécurité » ci-dessous — l'installateur ne le demande pas.** | Toutes |
| `TB_LOG_FORMAT` | Format des lignes de log, `human` ou `json` | Optionnel | `human` | Choix personnel | Toutes |
| `TB_TELEGRAM_BOT_TOKEN` | Jeton de bot Telegram pour les alertes | Optionnel — utilisé seulement si `configs/alerting.yaml: telegram.enabled` vaut `true` | aucune | Telegram — créez un bot via **BotFather** dans Telegram et copiez le jeton fourni | Toutes |
| `TB_TELEGRAM_CHAT_ID` | Identifiant du chat de destination des alertes Telegram | Optionnel, associé au jeton ci-dessus | aucune | Telegram — envoyez un message à votre bot, puis utilisez un bot/outil « get my chat id », ou l'API de Telegram elle-même, pour le récupérer | Toutes |
| `TB_SMTP_USERNAME` | Nom d'utilisateur SMTP pour les alertes par e-mail | Optionnel — utilisé seulement si `configs/alerting.yaml: email.enabled` vaut `true` | aucune | Votre propre compte fournisseur SMTP (l'hôte par défaut livré est `smtp.gmail.com`, mais n'importe quel fournisseur SMTP fonctionne) | Toutes |
| `TB_SMTP_PASSWORD` | Mot de passe SMTP pour les alertes par e-mail | Optionnel, associé au nom d'utilisateur ci-dessus | aucune | Votre fournisseur SMTP. **Avec Gmail**, ce doit être un **mot de passe d'application** Gmail, pas votre mot de passe de compte habituel — Gmail rejette les mots de passe classiques pour l'auth SMTP une fois la 2FA activée | Toutes |

**Une clé saisie sur la page Settings l'emporte toujours.** Chaque clé de
fournisseur IA ci-dessus (sauf `ollama`, qui n'en a pas besoin, et
`claude_code`, qui utilise votre connexion CLI) peut aussi être saisie sur
la page Settings de l'application plutôt que dans `.env`. Une clé saisie
sur cette page est chiffrée avec Fernet au repos, prend effet
immédiatement sans redémarrage, et n'est jamais réécrite dans `.env`. Si
les deux sont définies, la valeur de la page Settings l'emporte.

---

## 2. Variables d'environnement du processus passerelle, par compte (sans préfixe `TB_`)

La passerelle MT5 est un processus séparé avec son propre environnement —
elle ne lit **pas** le fichier `.env` à la racine du dépôt elle-même (seule
la classe `Settings` du backend fait cela). Ces variables sont fournies
directement à l'environnement de chaque processus passerelle — par les
lignes `Environment=` d'une unité systemd, les lignes `set VAR=` d'un
lanceur `.cmd` Windows, ou le préfixe d'environnement en ligne de
`make dev-gateway` pour un usage en dev manuel.

| Variable | Rôle | Obligatoire ? | Défaut | Comment obtenir une valeur | Plateformes |
|---|---|---|---|---|---|
| `GATEWAY_HOST` | Hôte d'écoute du serveur propre de la passerelle | Optionnel | `127.0.0.1` | Choix personnel (invite de l'installateur, ou défaut du Makefile) | Linux, Windows, `make dev-gateway` |
| `GATEWAY_PORT` | Port d'écoute de la passerelle | Optionnel | `8787` (auto-incrémenté par compte supplémentaire : `8788`, `8789`, …) | Choix personnel | Idem |
| `GATEWAY_SHARED_SECRET` | La moitié côté passerelle du secret partagé — doit correspondre exactement au `TB_GATEWAY_SHARED_SECRET[_<COMPTE>]` du backend pour ce même compte | Idem `TB_GATEWAY_SHARED_SECRET` ci-dessus | aucune | Même valeur, copiée depuis `.env` au moment de la génération/du lancement du service — voir §1 | Toutes |
| `MT5_TERMINAL_PATH` | Chemin absolu vers le `terminal64.exe` propre à ce compte | Optionnel pour un compte unique/principal ; **obligatoire** pour chaque compte concurrent supplémentaire | aucune (non défini = s'attacher à n'importe quel terminal déjà en cours d'exécution) | Déterminé par vous-même — où que vous ayez installé/installé sous Wine le terminal MT5 de ce compte | Toutes |
| `MT5_TERMINAL_SUBPATH` | Alias historique — chemin relatif au `drive_c/` du préfixe Wine | Optionnel, historique (`MT5_TERMINAL_PATH` est prioritaire si les deux sont définis) | aucune | Déterminé par vous-même | Linux/Wine uniquement |

`GATEWAY_HOST`/`GATEWAY_PORT` par compte ne sont pas des lignes `.env`
séparées — ils vivent dans le champ `gateway_url` de ce compte dans
`configs/accounts.yaml` et sont dérivés à nouveau dans l'environnement
propre du processus passerelle au moment de la génération/du lancement du
service.

---

## 3. Identifiants du courtier (login / mot de passe / serveur MT5)

Couvert en détail en tête de ce document. Résumé pour référence : ces
trois valeurs proviennent de votre courtier MT5 (compte démo ou réel), ne
sont saisies que via le panneau **MT5 Account** de l'UI de l'application
(`backend/src/broker/api/routes.py`), et sont stockées — si vous cochez
« remember » — sous forme d'un fichier chiffré avec Fernet par compte
(`backend/src/broker/adapters/credential_store.py`), la clé de chiffrement
elle-même étant conservée dans le trousseau (keyring) du système
d'exploitation, jamais dans un fichier à côté du texte chiffré. Ce ne sont
jamais des variables d'environnement et elles n'apparaissent jamais dans
`installer/`, `.env`, `.env.example`, ni aucun fichier Docker.

---

## 4. Autres secrets conservés dans le trousseau (keyring) du système

Deux autres secrets vivent uniquement dans le trousseau de votre système
d'exploitation (GNOME Keyring/KWallet sous Linux, Gestionnaire
d'identification sous Windows) sous le nom de service `"trading-bot"` —
jamais dans `.env`, jamais quelque chose que vous choisissez ou saisissez.
Les deux sont générés automatiquement dès la première utilisation
(`backend/src/shared/security/keyring_store.py`) :

| Élément | Rôle | Nom de clé dans le trousseau |
|---|---|---|
| Clé Fernet de chiffrement des identifiants | Chiffre à la fois vos identifiants de courtier MT5 et toute clé d'API IA enregistrée via la page Settings — une seule clé partagée entre les deux magasins | `credential-encryption-key` |
| Clé Fernet de signature de session | Signe/chiffre le jeton de session émis lorsque `TB_APP_PASSWORD` est défini | `session-signing-key` |

Il n'y a rien à configurer ici — sachez simplement que perdre l'accès au
trousseau du système (p. ex. un profil OS neuf, ou une réinitialisation du
trousseau) invalide silencieusement chaque identifiant et clé d'API
stockés ; il suffit alors de les ressaisir via l'UI.

---

## 5. `configs/*.yaml` — aucune valeur secrète, seulement des sélecteurs

Aucun fichier YAML sous `configs/` ne contient de véritable valeur secrète.
Quelques champs *nomment* la variable `.env` à utiliser, ou *sélectionnent*
si un secret est nécessaire :

- `configs/accounts.yaml` — le champ `gateway_shared_secret_env` de chaque
  compte nomme la variable `.env` contenant le secret de ce compte (p. ex.
  `TB_GATEWAY_SHARED_SECRET_DEMO_1`) — jamais la valeur du secret
  elle-même. Le commentaire d'en-tête du fichier énonce cette règle
  explicitement.
- `configs/news.yaml` — `calendar.source: forexfactory | finnhub` décide
  si `TB_FINNHUB_API_KEY` est nécessaire (le défaut, `forexfactory`, n'a
  besoin d'aucune clé).
- `configs/alerting.yaml` — `telegram.enabled` / `email.enabled`
  déterminent si les variables `TB_TELEGRAM_*` / `TB_SMTP_*`
  correspondantes sont réellement utilisées ; le fichier lui-même ne
  contient que des réglages non secrets d'hôte/port/adresse.
- `configs/ai.yaml` — `provider_per_task` nomme quel fournisseur/modèle
  traite chaque tâche IA ; ses commentaires indiquent quelle
  `TB_*_API_KEY` chaque fournisseur nécessite, mais le fichier ne
  contient aucune valeur de clé.

---

## 6. Variables du frontend

| Variable | Rôle | Obligatoire ? | Défaut | Plateformes |
|---|---|---|---|---|
| `BACKEND_URL` | URL de base du backend vers laquelle le serveur Next.js réachemine les réécritures `/api/*` (côté serveur uniquement — jamais exposée au navigateur) | Optionnel | `http://127.0.0.1:8000` | Choix personnel ; seul Docker le surcharge (`docker-compose.prod.yml` règle `BACKEND_URL=http://backend:8000` pour le réseau conteneur-à-conteneur) |
| `NEXT_PUBLIC_WS_URL` | URL de base Socket.IO pour le streaming de données de marché en temps réel — contourne le proxy de réécriture de Next.js, qui ne gère pas les WebSockets | Optionnel | `http://127.0.0.1:8000` | Choix personnel, seulement si votre backend n'est pas à l'adresse de dev par défaut |

**Lacune connue sur `NEXT_PUBLIC_WS_URL`** : il n'existe actuellement
aucun chemin de configuration pour cette variable sous Docker — elle n'est
ni dans `.env.example`, ni définie dans le bloc `environment:` d'aucun des
fichiers Compose, ni câblée par l'installateur. Dans la topologie Docker
de production (frontend et backend dans des conteneurs séparés, sur des
noms d'hôte séparés), la valeur par défaut compilée
`http://127.0.0.1:8000` est incorrecte du point de vue du navigateur.
Notez aussi que les variables `NEXT_PUBLIC_*` sont intégrées dans le
bundle JavaScript au moment de la **compilation** Next.js — la définir
comme entrée `environment:` de Compose sur l'image déjà construite n'aurait
aucun effet ; il faudrait qu'elle devienne un `ARG` de build Docker à la
place. Si vous rencontrez ce problème, la solution de contournement
aujourd'hui est de reconstruire l'image frontend avec la valeur intégrée
au moment de la compilation, ou de faire tourner le frontend nativement là
où la valeur par défaut est correcte.

---

## 7. Secrets CI Docker Hub (propriétaire du dépôt uniquement)

Ce ne sont pas des secrets d'exécution — ce sont des secrets/variables de
dépôt GitHub qui permettent à
`.github/workflows/docker-publish.yml` de publier des images sur Docker
Hub. Voir la section « GitHub Actions » ci-dessous pour savoir exactement
où les configurer.

| Nom | Type | Rôle |
|---|---|---|
| `DOCKERHUB_USERNAME` | Secret du dépôt | Nom d'utilisateur de connexion Docker Hub |
| `DOCKERHUB_TOKEN` | Secret du dépôt | **Jeton d'accès** Docker Hub — générez-le depuis les réglages de compte Docker Hub, pas votre mot de passe de compte |
| `DOCKERHUB_NAMESPACE` | Variable du dépôt (pas un secret) | Sous quel org/utilisateur Docker Hub publier les images ; retombe sur le placeholder `tradingbot` si non défini |

Obtenez un jeton d'accès Docker Hub depuis
[hub.docker.com](https://hub.docker.com) → Account Settings → Security →
New Access Token.

---

## Spécifique à chaque plateforme : comment je règle ça concrètement ?

### (a) Linux natif

1. `cp .env.example .env`, puis éditez `.env` avec n'importe quel éditeur
   — c'est un simple fichier `CLÉ=valeur`, une variable par ligne.
2. Si vous avez enregistré des services de démarrage automatique via
   l'installateur, chaque unité systemd `--user` sous
   `~/.config/systemd/user/` (`trading-bot-backend.service`,
   `trading-bot-gateway@<id>.service`, etc.) a le secret de passerelle du
   compte et le chemin du terminal **intégrés directement dans ses lignes
   `Environment=`** au moment de la génération (`render_all()` de
   `installer/services_linux.py`) — elle ne relit pas `.env` à chaque
   démarrage. Le backend et le frontend, eux, lisent normalement le reste
   de `.env` au démarrage du processus (via `pydantic-settings`), puisque
   ce sont de simples processus Python/Node avec `.env` dans leur
   répertoire de travail.
3. Après avoir édité `.env`, redémarrez le(s) service(s) concerné(s) —
   `python3 installer/manage.py restart` (ou directement `systemctl
   --user restart trading-bot-backend.service`) — une simple modification
   de fichier ne se propage pas à un processus déjà en cours d'exécution.

### (b) Windows natif

1. `.env` vit à la racine du dépôt, même format que sous Linux — éditez-le
   directement.
2. Le démarrage automatique fonctionne différemment sous Windows :
   `install_services.ps1` lit `.env` **une seule fois, au moment de la
   génération**, via son assistant `Read-DotEnvValue` (un analyseur minimal
   qui renvoie la première ligne `Nom=valeur` non commentée pour une clé
   donnée), et écrit la valeur résolue directement dans un lanceur `.cmd`
   généré sous `installer\services\windows\generated\` (p. ex. `set
   GATEWAY_SHARED_SECRET=...`). La tâche planifiée exécute ensuite
   simplement ce `.cmd` — elle ne relit jamais `.env` elle-même au moment
   de l'exécution de la tâche.
3. À cause de cela, éditer `.env` après l'enregistrement des services ne
   met **pas** à jour un lanceur `.cmd` déjà généré — voir « Faire tourner
   un secret compromis » ci-dessous.
4. Les scripts de service Windows sont explicitement marqués **NON
   VÉRIFIÉS** (`UNVERIFIED`) dans leurs propres commentaires d'en-tête
   (écrits en suivant le comportement documenté du module PowerShell
   `ScheduledTasks`, mais pas encore exécutés sur une vraie machine
   Windows) — testez sur une VM jetable avant de vous y fier pour un
   compte réel.

### (c) Docker / Compose

1. `cp .env.example .env`, éditez-le, puis `docker compose -f
   docker-compose.prod.yml up -d`. Les services `backend` et `frontend`
   déclarent tous deux `env_file: .env`, donc Compose lit le fichier
   entier et injecte chaque variable dans l'environnement des deux
   conteneurs au démarrage — aucun filtrage par service n'a lieu.
2. **Limitation connue, pas quelque chose à corriger en modifiant le
   fichier compose** : le conteneur frontend reçoit actuellement
   l'intégralité du `.env` backend — chaque clé de fournisseur IA, chaque
   secret de passerelle, `TB_APP_PASSWORD`, les identifiants d'alerte —
   alors que le frontend n'a en réalité besoin que de `BACKEND_URL`. C'est
   une vraie considération de rayon d'impact : si le conteneur frontend
   était un jour compromis (une vulnérabilité de dépendance, un problème
   de chaîne d'approvisionnement dans un paquet npm, etc.), tout ce qui se
   trouve dans votre `.env` serait accessible depuis l'intérieur. Tant que
   ce n'est pas resserré, considérez que tout ce que vous mettez dans
   `.env` est accessible depuis les deux conteneurs, et évitez d'y mettre
   quoi que ce soit que vous ne voudriez pas voir exposé de cette façon.
3. La passerelle n'est jamais conteneurisée (voir `gateway/README.fr.md`)
   — un déploiement basé sur Docker atteint une passerelle exécutée en
   externe via `TB_GATEWAY_URL` dans `.env`, comme toute autre valeur de
   configuration du backend.

### (d) GitHub Actions

Configuration réservée au propriétaire du dépôt — allez dans **Settings →
Secrets and variables → Actions** sur GitHub.

| Workflow | Déclencheur | Consomme |
|---|---|---|
| `.github/workflows/ci.yml` | Chaque push sur `main`, chaque pull request | Aucun secret — le job installer-smoke-test copie `.env.example` → `.env` tel quel ; aucun vrai identifiant n'est jamais nécessaire pour le lint/les tests/le build |
| `.github/workflows/release.yml` | Push d'un tag `v*`, ou déclenchement manuel `workflow_dispatch` | `GITHUB_TOKEN` uniquement — fourni automatiquement par GitHub Actions, aucune configuration nécessaire au-delà du bloc `permissions:` du workflow lui-même |
| `.github/workflows/docker-publish.yml` | Push d'un tag `v*`, ou déclenchement manuel `workflow_dispatch` | Les secrets du dépôt `DOCKERHUB_USERNAME` et `DOCKERHUB_TOKEN` (sous « Secrets ») ; la variable du dépôt `DOCKERHUB_NAMESPACE` (sous « Variables », pas « Secrets » — elle n'est pas sensible) |

`docker-publish.yml` est explicitement documenté comme **inerte** tant que
les deux secrets Docker Hub ne sont pas configurés — jusque-là, il échoue
simplement à l'étape de connexion au lieu de publier quoi que ce soit.

---

## Notes de sécurité

**Ne commitez jamais `.env`.** Il est déjà listé dans `.gitignore` (entrée
`.env` du `.gitignore` racine) — vérifiez avec `git check-ignore .env` en
cas de doute. Seul `.env.example` (sans aucune valeur réelle) est destiné à
être commité.

**Faire tourner un secret compromis n'est pas qu'une simple édition de
`.env`.** Si `TB_GATEWAY_SHARED_SECRET` (ou une variante par compte) fuite,
éditer `.env` seul ne suffit pas :
- Sous **Linux**, la valeur est intégrée dans la ligne `Environment=` d'une
  unité systemd déjà générée — il faut relancer la génération de
  l'installateur/du service (`installer/services_linux.py`, ou
  `python3 installer/install.py --reconfigure`) pour que l'unité soit
  réécrite, puis redémarrer le service.
- Sous **Windows**, la valeur est intégrée dans un lanceur `.cmd` déjà
  généré sous `installer\services\windows\generated\` — relancez
  `install_services.ps1` pour le régénérer.
- Mettez aussi à jour la valeur `.env` correspondante côté passerelle (ou
  la variable `GATEWAY_SHARED_SECRET` équivalente, où que tourne ce
  processus passerelle) pour que les deux moitiés correspondent à nouveau.

Un secret de fournisseur IA ou un identifiant d'alerte compromis est plus
simple à traiter — remplacez-le simplement dans `.env` (ou sur la page
Settings, pour les clés de fournisseur IA) et redémarrez le backend ; rien
d'autre n'en a une copie intégrée ailleurs.

**Lacune sur `TB_APP_PASSWORD` — réglez-le vous-même avant d'exposer
l'application.** L'assistant de l'installateur (`installer/wizard.py`) ne
demande **pas** actuellement `TB_APP_PASSWORD`, alors que c'est le seul
contrôle protégeant toutes les routes sauf `/health` et `/auth/*` sur une
application capable de passer des ordres réels. Si vous lancez uniquement
l'installateur CLI et n'ouvrez jamais `.env` manuellement ensuite, vous
pouvez vous retrouver avec une instance accessible depuis Internet, en
trading réel, **sans aucune connexion requise**. Avant de rendre cette
application accessible au-delà de `127.0.0.1` — un VPS, une redirection de
port, un reverse proxy — ouvrez `.env` vous-même et réglez
`TB_APP_PASSWORD` avec un mot de passe de votre choix, puis redémarrez le
backend.

---

## Dépannage

| Symptôme | Cause probable |
|---|---|
| Les appels à la passerelle renvoient `401` | `TB_GATEWAY_SHARED_SECRET[_<COMPTE>]` dans `.env` ne correspond pas au `GATEWAY_SHARED_SECRET` avec lequel le processus passerelle a réellement été lancé — vérifiez si vous avez édité `.env` après que les services aient déjà été générés (voir « Faire tourner un secret compromis » ci-dessus) |
| Les appels à la passerelle renvoient `502` | Mauvais `TB_GATEWAY_URL` (ou `gateway_url` par compte dans `configs/accounts.yaml`), ou le processus passerelle/le terminal ne tourne pas du tout |
| L'application se charge sans invite de connexion, même à distance | `TB_APP_PASSWORD` n'est pas défini — voir la note de sécurité ci-dessus |
| Une tâche IA renvoie `503` en mentionnant un nom `TB_*_API_KEY` | La clé de ce fournisseur est absente/vide à la fois dans `.env` et sur la page Settings |
| Une tâche IA renvoie `502` ou une erreur d'adaptateur | La clé est présente mais invalide, ou l'identifiant de modèle dans `configs/ai.yaml`/Settings n'est pas reconnu par le fournisseur |
| Le calendrier d'actualités échoue quand `configs/news.yaml: calendar.source: finnhub` | `TB_FINNHUB_API_KEY` est absente — la source par défaut `forexfactory` ne nécessite aucune clé, `finnhub` si |
| Les alertes Telegram/e-mail n'arrivent jamais | Soit le canal est `enabled: false` dans `configs/alerting.yaml`, soit les variables `TB_TELEGRAM_*`/`TB_SMTP_*` correspondantes sont vides/erronées |
| `docker compose ... pull` échoue avec image introuvable | `DOCKERHUB_NAMESPACE` pointe encore vers le placeholder `tradingbot`, ou le propriétaire du dépôt n'a pas encore configuré `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` — construisez les images en local à la place (voir `INSTALL.fr.md`) |
| Éditer `.env` ne change pas le comportement d'un service de démarrage automatique déjà lancé | Comportement attendu — voir les sections par plateforme ci-dessus ; redémarrez (Linux) ou régénérez le lanceur (Windows) après avoir édité `.env` |
