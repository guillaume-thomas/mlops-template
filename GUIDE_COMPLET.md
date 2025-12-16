# 📚 Guide Complet - Projet MLOps avec MCP

> **Note** : Ce document est une synthèse de toute la documentation du projet.
> Pour l'architecture technique détaillée, voir [ARCHITECTURE_MCP.md](ARCHITECTURE_MCP.md)

## 🎯 Vue d'ensemble

Ce projet implémente un **chatbot conversationnel** utilisant le **Model Context Protocol (MCP)** pour appeler une API ML de prédiction de survie sur le Titanic.

### Architecture en 3 services
- **Chatbot** (Streamlit + LangChain + GitHub Models)
- **MCP Server** (expose l'API ML comme tool MCP)
- **API ML** (FastAPI + Random Forest pour prédictions Titanic)

---

## 🚀 Quick Start

### Prérequis
- Python 3.13
- uv (gestionnaire de packages)
- Compte GitHub avec accès à GitHub Models
- Accès OpenShift (pour déploiement)

### Installation locale

```bash
# Installer uv
pip install uv

# Synchroniser les dépendances
uv sync --all-groups

# Lancer l'API ML
uv run --group api uvicorn summit.api.main:app --port 8080

# Lancer le serveur MCP
uv run --group mcp-server python -m summit.mcp_server.server

# Lancer le chatbot
export OPENAI_API_KEY="ton_token_github"
export OPENAI_BASE_URL="https://models.inference.ai.azure.com"
uv run --group chatbot streamlit run src/summit/chatbot/app.py
```

### Test local du MCP

```bash
# Terminal 1 : Serveur
python test_mcp_local.py server

# Terminal 2 : Client
python test_mcp_local.py client
```

---

## 🔐 Configuration des Secrets

### GitHub Models Token (REQUIS)

Le chatbot utilise GitHub Models (gpt-4o-mini) qui nécessite un token d'authentification.

#### Étape 1 : Obtenir le token

1. Va sur https://github.com/settings/tokens
2. Clique sur **"Generate new token"** → **"Generate new token (classic)"**
3. Nom : `GitHub Models Token for MLOps`
4. Scope : ✅ `read:user`
5. Génère et **copie le token**

#### Étape 2 : Ajouter le secret dans GitHub

1. Va sur ton repo : `Settings` → `Secrets and variables` → `Actions`
2. Clique sur **"New repository secret"**
3. **Name** : `GH_MODELS_TOKEN`
4. **Value** : Colle ton token
5. **Add secret**

#### Étape 3 : Redéployer

```bash
# Déclencher un redéploiement
git commit --allow-empty -m "chore: trigger redeploy with valid token"
git push origin feature/mcp_real
```

**OU** déclenche manuellement : `Actions` → `Deploy Titanic Chatbot` → `Run workflow`

---

## 🏗️ Architecture des Secrets

```
GitHub Repository Secrets
    └── GH_MODELS_TOKEN
         │
         ▼
    GitHub Actions CI/CD
         │ (créé le secret Kubernetes)
         ▼
    Kubernetes Secret (chatbot-secrets)
         │
         ▼
    Pod Chatbot (env: OPENAI_API_KEY)
         │
         ▼
    GitHub Models API (gpt-4o-mini)
```

**Sécurité** :
- ✅ Token stocké dans GitHub Secrets (chiffré)
- ✅ Token injecté dans Kubernetes par CI/CD
- ✅ Jamais committé dans le code
- ✅ Accessible uniquement par le pod chatbot

---

## 🛠️ Résolution des Problèmes

### Problème 1 : Erreur 401 "Bad credentials"

**Symptôme** :
```
Error code: 401 - {'error': {'code': 'unauthorized', 'message': 'Bad credentials'}}
```

**Cause** : Token GitHub Models invalide ou manquant

**Solution** : Configure `GH_MODELS_TOKEN` dans GitHub Secrets (voir section ci-dessus)

### Problème 2 : MCP ne se connecte pas

**Symptôme** :
```
asyncio.exceptions.CancelledError: Cancelled via cancel scope
```

**Solution** : Le MCP est maintenant corrigé ! Logs attendus :
```
[CHATBOT] Connecting to MCP server: ...
[CHATBOT] SSE connection established
[CHATBOT] MCP session initialized
```

### Problème 3 : Tool LangChain ne fonctionne pas

**Symptôme** :
```
tool() got an unexpected keyword argument 'name'
```

**Solution** : Corrigé ! On utilise maintenant `StructuredTool.from_function()`

---

## 📊 Vérification du Déploiement

### Vérifier les secrets

```bash
# Secret existe ?
oc get secret chatbot-secrets

# Valeur du token (base64)
oc get secret chatbot-secrets -o jsonpath='{.data.github-models-token}' | base64 -d
```

### Vérifier les logs

```bash
# Logs du serveur MCP
oc logs -l app=titanic-mcp-server -f

# Logs du chatbot
oc logs -l app=titanic-chatbot -f

# Logs de l'API ML
oc logs -l app=mlops-api -f
```

### Logs attendus (succès)

**Serveur MCP** :
```
INFO: [SSE] Connection from 10.131.3.10
INFO: [SSE] SSE connection established
INFO: [MCP] list_tools called
INFO: [MCP] Returning 1 tools
INFO: [MCP] call_tool: predict_survival with args {...}
```

**Chatbot** :
```
[CHATBOT] Connecting to MCP server: ...
[CHATBOT] SSE connection established
[CHATBOT] MCP session initialized
[CHATBOT] Found 1 tools: ['predict_survival']
[CHATBOT] Calling LLM with tools...
[CHATBOT] LLM wants to call tool: predict_survival
[CHATBOT] Tool result: Based on the passenger characteristics...
```

---

## 🧪 Tests

### Test du MCP en local

Le script `test_mcp_local.py` permet de tester le MCP sans déployer :

```bash
# Terminal 1 : Serveur
python test_mcp_local.py server

# Terminal 2 : Client (dans un autre terminal)
python test_mcp_local.py client
```

**Résultats attendus** :
```
✅ SSE connection established
✅ Session initialized
✅ Tools found: ['predict_survival']
✅ Tool called successfully
```

### Test du chatbot

Questions à tester :
```
✅ "Un homme de 3ème classe survivrait-il ?"
✅ "Une femme de 1ère classe avec 2 enfants ?"
✅ "Quelle est la probabilité de survie d'un homme de 2ème classe ?"
```

---

## 🔄 Workflow CI/CD

### 3 Pipelines GitHub Actions

1. **deploy-mcp-server.yml** : Build et déploie le serveur MCP
2. **deploy-chatbot.yml** : Build, crée le secret, déploie le chatbot
3. **ct-ci-cd.yaml** : Pipeline principal (API ML + tests)

### Déclenchement automatique

Les pipelines se déclenchent sur :
- Push vers `feature/mcp_real`
- Changements dans les répertoires concernés
- Déclenchement manuel via `workflow_dispatch`

### Étapes du déploiement chatbot

```yaml
1. Checkout code
2. Configure docker et kubectl
3. Build image Docker
4. Push vers Quay.io
5. Créer/Update secret Kubernetes (avec GH_MODELS_TOKEN)
6. Apply chatbot.yaml
7. Rollout restart
8. Get route URL
```

---

## 📖 Documentation Technique

### Structure du projet

```
mlops-template-SES-test/
├── src/summit/
│   ├── api/              # API ML (FastAPI)
│   ├── mcp_server/       # Serveur MCP (Starlette)
│   └── chatbot/          # Chatbot (Streamlit)
├── k8s/                  # Manifests Kubernetes
│   ├── api/
│   ├── mcp_server/
│   └── chatbot/
├── .github/workflows/    # CI/CD
├── tests/                # Tests unitaires
└── data/                 # Données d'entraînement
```

### Technologies utilisées

| Composant | Stack |
|-----------|-------|
| **Chatbot** | Streamlit, LangChain, GitHub Models (gpt-4o-mini) |
| **MCP Server** | Starlette, MCP SDK, SSE |
| **API ML** | FastAPI, scikit-learn, MLflow |
| **Infrastructure** | Kubernetes (OpenShift), Docker, Quay.io |
| **CI/CD** | GitHub Actions |
| **Observabilité** | OpenTelemetry, Jaeger |

### Dépendances par groupe

```toml
[dependency-groups]
api = ["fastapi", "uvicorn", "opentelemetry-*"]
mcp-server = ["uvicorn", "mcp", "httpx"]
chatbot = ["streamlit", "langchain", "langchain-openai", "mcp"]
training = ["mlflow", "ydata-profiling"]
dev = ["pytest", "pytest-asyncio"]
```

---

## 🎓 Utilisation Pédagogique

Ce projet est excellent pour un **cours MLOps** car il démontre :

### Concepts Techniques
✅ Model Context Protocol (MCP)
✅ Server-Sent Events (SSE)
✅ LangChain + LLM
✅ Architecture microservices
✅ Kubernetes / OpenShift
✅ CI/CD avec GitHub Actions
✅ Gestion sécurisée des secrets
✅ Logging et observabilité

### Best Practices
✅ Séparation des responsabilités
✅ Configuration via variables d'environnement
✅ Secrets gérés par Kubernetes
✅ Timeouts sur opérations async
✅ Logging structuré
✅ Tests automatisés
✅ Documentation complète

---

## 🐛 Troubleshooting Avancé

### Le pod chatbot ne démarre pas

```bash
# Voir les events
oc describe pod -l app=titanic-chatbot

# Vérifier l'image
oc describe pod -l app=titanic-chatbot | grep Image:

# Forcer un redémarrage
oc rollout restart deployment/titanic-chatbot
```

### Erreur de connexion réseau entre pods

```bash
# Tester depuis le chatbot vers le MCP server
oc exec -it deployment/titanic-chatbot -- curl http://titanic-mcp-server:8000/health

# Tester depuis le MCP server vers l'API ML
oc exec -it deployment/titanic-mcp-server -- curl http://mlops-api-service:8080/health
```

### Vérifier les variables d'environnement

```bash
# Chatbot
oc get deployment titanic-chatbot -o yaml | grep -A 10 env:

# MCP Server
oc get deployment titanic-mcp-server -o yaml | grep -A 10 env:
```

### Tester l'API GitHub Models

```bash
curl https://models.inference.ai.azure.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TON_TOKEN" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 📝 Historique des Corrections

### Problèmes résolus (16 décembre 2024)

#### 1. Serveur MCP ne fonctionnait pas
**Avant** : Tentative manuelle de gérer SSE (complexe et cassé)
**Après** : Utilisation de `SseServerTransport` du SDK MCP
**Impact** : ✅ MCP fonctionne à 100%

#### 2. Client MCP sans timeouts
**Avant** : Opérations async sans timeout → CancelledError
**Après** : Timeouts de 10s sur toutes les opérations
**Impact** : ✅ Connexion stable

#### 3. LangChain Tool incompatible
**Avant** : `@tool(name=...)` (API obsolète)
**Après** : `StructuredTool.from_function()`
**Impact** : ✅ Tools MCP fonctionnels

#### 4. Secret hardcodé
**Avant** : Token en clair dans `chatbot.yaml`
**Après** : Secret géré par CI/CD depuis GitHub Secrets
**Impact** : ✅ Sécurité améliorée

---

## 🔗 Liens Utiles

### Documentation MCP
- Spécification : https://spec.modelcontextprotocol.io/
- SDK Python : https://github.com/modelcontextprotocol/python-sdk
- Exemples : https://modelcontextprotocol.io/examples

### GitHub Models
- Marketplace : https://github.com/marketplace/models
- Documentation : https://docs.github.com/en/github-models

### LangChain
- Documentation : https://python.langchain.com/
- ChatOpenAI : https://python.langchain.com/docs/integrations/chat/openai
- Tools : https://python.langchain.com/docs/modules/tools/

### MLflow
- Documentation : https://mlflow.org/docs/latest/
- Tracking : https://mlflow.org/docs/latest/tracking.html

---

## ✅ Checklist de Déploiement

Avant de considérer que tout fonctionne :

- [ ] Secret `GH_MODELS_TOKEN` configuré dans GitHub
- [ ] Les 3 services déployés dans OpenShift
- [ ] Les 3 services ont des pods en état `Running`
- [ ] Les logs du MCP server montrent des connexions
- [ ] Les logs du chatbot montrent `MCP client initialized successfully`
- [ ] Le chatbot répond à une question test
- [ ] Le chatbot appelle le tool `predict_survival`
- [ ] L'API ML retourne des prédictions
- [ ] Les routes OpenShift sont accessibles

---

## 🎉 Félicitations !

Si tu as suivi ce guide et que tous les tests passent, tu as maintenant :

✅ Un chatbot IA conversationnel fonctionnel
✅ Une architecture microservices complète
✅ Un exemple de Model Context Protocol en production
✅ Une stack MLOps moderne et sécurisée
✅ Un projet parfait pour un cours MLOps !

**Le système est prêt à être utilisé ! 🚀**

---

## 📞 Support

Pour toute question ou problème :
1. Vérifie d'abord les logs (voir section Vérification)
2. Consulte [ARCHITECTURE_MCP.md](ARCHITECTURE_MCP.md) pour les détails techniques
3. Teste en local avec `test_mcp_local.py`
4. Vérifie que le secret `GH_MODELS_TOKEN` est correctement configuré

**Bon courage avec ton projet MLOps ! 🎓**
