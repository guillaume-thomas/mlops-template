## 🚢 Titanic Chatbot

Chatbot Streamlit avec LLM (GitHub Models - gpt-4o-mini) qui utilise MCP pour appeler l'API d'inférence Titanic.

### ℹ️ Architecture MCP - Microservices séparés

Le serveur MCP est déployé comme un **service Kubernetes indépendant** qui communique avec le chatbot via **SSE (Server-Sent Events)** sur HTTP.

**Architecture microservices** :
- 🔧 **Serveur MCP** : Service séparé exposant les tools via SSE (port 8000)
- 💬 **Chatbot** : Client MCP qui se connecte au serveur via HTTP/SSE
- 🌐 **Communication** : JSON-RPC sur SSE (au lieu de stdio)
- 📦 **Déploiements** : Deux images Docker distinctes, deux workflows CI/CD

**Avantages** :
- ✅ Scalabilité : Le serveur MCP peut être scalé indépendamment
- ✅ Résilience : Restart du chatbot sans affecter le serveur MCP
- ✅ Production-ready : Architecture microservices standard
- ✅ Observabilité : Logs et métriques séparés

---

## 🚀 Quick Start

### 1. Obtenir un token GitHub Models

```bash
# Ouvrir https://github.com/settings/tokens
# Créer "Personal access token (classic)" - Aucune permission nécessaire
# Copier le token (ghp_...)
```

### 2. Configurer GitHub Secrets/Variables

**Settings → Secrets and variables → Actions**

Secrets :
- `GITHUB_MODELS_TOKEN` : token de l'étape 1
- `OPENSHIFT_TOKEN` : `oc whoami --show-token`
- `QUAY_ROBOT_TOKEN` : token du robot Quay.io

Variables :
- `OPENSHIFT_SERVER` : `oc whoami --show-server`
- `OPENSHIFT_USERNAME` : `gthomas59800`
- `QUAY_ROBOT_USERNAME` : `gthomas59800+robot`

### 3. Déployer

```bash
git add .
git commit -m "Deploy chatbot"
git push origin main
```

GitHub Action déploie automatiquement !

Voir : **Actions → Deploy Titanic Chatbot**

---

### 📂 Structure

```
src/summit/chatbot/
├── app.py          # Application Streamlit
└── agent.py        # Agent LangChain avec FastMCP client

src/summit/mcp_server/
└── server.py       # Serveur MCP FastMCP avec Streamable HTTP

k8s/chatbot/
├── Dockerfile      # Image Docker chatbot
└── chatbot.yaml    # Manifests Kubernetes chatbot

k8s/mcp_server/
├── Dockerfile      # Image Docker serveur MCP
└── mcp-server.yaml # Manifests Kubernetes serveur MCP

.github/workflows/
├── deploy-chatbot.yml     # CI/CD chatbot
└── deploy-mcp-server.yml  # CI/CD serveur MCP
```

---

## 🚀 Déploiement avec GitHub Actions

### Configuration des secrets GitHub

Aller dans : **Repository → Settings → Secrets and variables → Actions**

#### Secrets requis

| Secret | Exemple | Comment l'obtenir |
|--------|---------|-------------------|
| `GITHUB_MODELS_TOKEN` | `ghp_xxx...` | https://github.com/settings/tokens → "Personal access token (classic)" → Aucune permission nécessaire |
| `OPENSHIFT_TOKEN` | `sha256~xxx...` | `oc whoami --show-token` ou créer un service account |
| `QUAY_ROBOT_TOKEN` | `xxx...` | https://quay.io → Account Settings → Robot Accounts |

#### Variables requises

| Variable | Exemple | Comment l'obtenir |
|----------|---------|-------------------|
| `OPENSHIFT_SERVER` | `https://api.rm3.7wse.p1.openshiftapps.com:6443` | `oc whoami --show-server` |
| `OPENSHIFT_USERNAME` | `gthomas59800` | Votre username OpenShift (le namespace sera `{username}-dev`) |
| `QUAY_ROBOT_USERNAME` | `gthomas59800+robot` | Nom du robot account Quay.io |

#### Détails : Token GitHub Models

1. Aller sur https://github.com/settings/tokens
2. Créer un "Personal access token (classic)"
3. Nom: "Titanic Chatbot"
4. **Aucune permission nécessaire** (le token suffit pour GitHub Models)
5. Copier le token (commence par `ghp_`)

#### Détails : Robot Quay.io

1. Aller sur https://quay.io/user/{username}?tab=settings
2. Section "Robot Accounts" → Create Robot Account
3. Nom: `summit_chatbot`
4. Permissions: Write sur le repository `summit/chatbot`
5. Copier le token et le username

### Déploiement automatique via GitHub Actions

#### Déclencheurs

Le workflow `.github/workflows/deploy-chatbot.yml` se déclenche automatiquement :
- **Push sur `main`** modifiant `src/summit/chatbot/**` ou `k8s/chatbot/**`
- **Manuellement** via Actions → Deploy Titanic Chatbot → Run workflow

#### Étapes du workflow

1. **Checkout code** - Récupère le code source
2. **Configure docker and kubectl**
   - Login Quay.io avec robot account
   - Configure kubectl pour OpenShift
3. **Build and push Docker image**
   - Build avec Docker (pas buildx, comme dans ct-ci-cd.yaml)
   - Tags: `latest` et `{git-sha}`
   - Push vers `quay.io/gthomas59800/summit/chatbot`
4. **Create or update Secret**
   - Crée/met à jour `chatbot-secrets` dans OpenShift
   - Injecte `GITHUB_MODELS_TOKEN` depuis GitHub Secrets
5. **Deploy to OpenShift**
   - Applique les manifests Kubernetes
   - Met à jour l'image du deployment avec le tag `{git-sha}`
6. **Get route URL**
   - Récupère l'URL du chatbot
   - Affiche dans le summary GitHub

#### Après le déploiement

```bash
# Voir le workflow
Repository → Actions → Deploy Titanic Chatbot

# Récupérer l'URL manuellement
kubectl get route titanic-chatbot -o jsonpath='{.spec.host}'
```

### Déploiement manuel (optionnel)

```bash
# Build et push l'image
docker build -t quay.io/gthomas59800/summit/chatbot -f k8s/chatbot/Dockerfile .
docker push quay.io/gthomas59800/summit/chatbot

# Déployer sur OpenShift
kubectl apply -f k8s/chatbot/chatbot.yaml

# Obtenir l'URL
oc get route titanic-chatbot
```

---

## 💡 GitHub Models - Configuration

### Pourquoi GitHub Models ?

✅ **Gratuit avec compte GitHub** - Pas besoin de carte bancaire
✅ **Pas de nouveau compte** - Utilise votre compte GitHub existant
✅ **Simple** - Un seul token pour tout
✅ **Sécurisé** - Token géré via GitHub Secrets

### Modèle utilisé : gpt-4o-mini

**Le moins gourmand en rate limit** - Parfait pour un exemple simple !

| Critère | Valeur |
|---------|--------|
| Rate limit | ~15 requêtes/minute |
| Tokens/jour | ~150K |
| Contexte | 128K tokens |
| Vitesse | ⚡ Très rapide |
| Qualité | ✅ Excellente pour chatbot |

### Autres modèles disponibles

- **gpt-4o** : Plus puissant mais plus de rate limit
- **Phi-3** : Léger, Microsoft
- **Llama 3.1** : Open source

---

## 🧪 Test local

### Test du serveur MCP seul

```bash
# Tester le serveur MCP directement (sans LLM)
export TITANIC_API_URL="http://localhost:8080"
uv run --group chatbot python tests/mcp_server/test_server.py
```

Vous verrez :
- ✅ Connexion au serveur MCP
- 📋 Liste des tools disponibles
- 🧪 Tests d'appels au tool `predict_survival`

### Test du chatbot complet (avec MCP + LLM)

```bash
# Variables d'environnement
export OPENAI_API_KEY="ghp_YOUR_GITHUB_TOKEN"
export OPENAI_BASE_URL="https://models.inference.ai.azure.com"
export TITANIC_API_URL="http://localhost:8080"
export LLM_MODEL="gpt-4o-mini"

# Lancer le chatbot
uv run --group chatbot chatbot
```

Ouvrir http://localhost:8501

Le chatbot utilise maintenant :
1. Le LLM (GitHub Models) pour comprendre la question
2. Le client MCP pour appeler le serveur
3. Le serveur MCP pour faire la prédiction via l'API Titanic

---

## 💬 Exemples de questions

Une fois déployé, posez des questions comme :

- "Would a first-class female passenger with 1 sibling survive?"
- "Predict survival for a third-class male with no family"
- "What are the survival chances for a middle-class woman with 2 children?"
- "Can you predict if a rich man traveling alone would survive?"

---

## 🔧 Architecture technique avec MCP

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Streamlit UI   │ ──────> │  LangChain Agent │ ──────> │ GitHub Models   │
│  (app.py)       │         │  (agent.py)      │         │ (gpt-4o-mini)   │
└─────────────────┘         └────────┬─────────┘         └─────────────────┘
                                     │
                                     │ invoke tools
                                     ▼
                            ┌─────────────────┐
                            │   MCP Client    │
                            │   (SSE/HTTP)    │
                            └────────┬────────┘
                                     │
                              SSE (JSON-RPC)
                                     │
                            ┌────────▼────────┐
                            │   MCP Server    │  ← Service Kubernetes séparé
                            │  (FastAPI/SSE)  │
                            └────────┬────────┘
                                     │
                                    HTTP
                                     │
                            ┌────────▼────────┐
                            │  Titanic API    │
                            │  (FastAPI)      │
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │ Random Forest   │
                            │     Model       │
                            └─────────────────┘
```

### Composants MCP

- **MCP Server** (`titanic-mcp-server`) : Service Kubernetes exposant les tools via SSE sur port 8000
- **MCP Client** (intégré dans `agent.py`) : Se connecte au serveur via HTTP/SSE
- **Protocol** : JSON-RPC sur SSE (Server-Sent Events) pour communication asynchrone
- **Kubernetes** : Deux déploiements indépendants avec leurs propres images Docker

**Avantage pédagogique** : Architecture microservices avec Model Context Protocol !

---

## 🔒 Gestion des secrets

### Dans GitHub (pour CI/CD)

Secrets à configurer dans Settings → Secrets → Actions :

- `GITHUB_MODELS_TOKEN` : Token pour GitHub Models API
- `OPENSHIFT_SERVER` : URL du cluster OpenShift
- `OPENSHIFT_TOKEN` : Token de connexion OpenShift
- `OPENSHIFT_NAMESPACE` : Namespace de déploiement

### Dans OpenShift

Le secret est créé automatiquement par la GitHub Action :

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: chatbot-secrets
type: Opaque
stringData:
  github-models-token: ${GITHUB_MODELS_TOKEN}
```

---

## 📊 Ressources Kubernetes

- **CPU** : 100m (request) → 500m (limit)
- **Mémoire** : 256Mi (request) → 512Mi (limit)
- **Port** : 8501 (Streamlit)
- **Replicas** : 1

---

## ⚠️ Rate Limits

### gpt-4o-mini (gratuit)
- **15 requêtes/minute** - Largement suffisant pour tester
- **150K tokens/jour**
- Réinitialisation : chaque minute

### Gestion des erreurs

Si rate limit atteint :
1. **Attendre** - Les limites se réinitialisent chaque minute
2. **Optimiser** - Réduire température ou tokens max
3. **Changer de modèle** - Essayer Phi-3

---

## 🔗 Liens utiles

- **GitHub Models** : https://github.com/marketplace/models
- **Playground** : https://github.com/marketplace/models
- **Status** : https://www.githubstatus.com/
- **Tokens** : https://github.com/settings/tokens

---

## 🛡️ Sécurité

⚠️ **Important** :
- Ne commitez **JAMAIS** votre token dans Git
- Utilisez des **secrets GitHub** pour le CI/CD
- Utilisez des **secrets Kubernetes** en production
- Révocez le token si compromis : https://github.com/settings/tokens

---

## 🐛 Dépannage

### Rate limit atteint
```
Erreur: "Rate limit exceeded"
Solution: Attendre 1 minute, les limites se réinitialisent
```

### Token invalide
```
Erreur: "Unauthorized" ou "Invalid authentication"
Solution: Vérifier que le token commence par ghp_ et est valide
```

### API GitHub down
```
Solution: Vérifier https://www.githubstatus.com/
```

### Pod ne démarre pas
```bash
# Vérifier les logs
kubectl logs -f deployment/titanic-chatbot

# Vérifier le secret
kubectl get secret chatbot-secrets -o yaml
```
