# 🚢 MLOps Template - Titanic Survival Prediction with MCP

Template MLOps complet démontrant une architecture moderne avec **Model Context Protocol (MCP)** pour l'enseignement.

## 🎯 Vue d'ensemble

Ce projet implémente une solution MLOps complète pour prédire la survie des passagers du Titanic, avec :

- 🤖 **Chatbot Streamlit** avec LangChain et GitHub Models (GPT-4o-mini)
- 🔧 **Serveur MCP** exposant les prédictions via le protocole MCP (Streamable HTTP)
- 🎯 **API ML FastAPI** avec monitoring OpenTelemetry + Jaeger
- 📊 **MLflow** pour le tracking des expériences
- ☸️ **Déploiement Kubernetes/OpenShift** complet
- 🔄 **CI/CD GitHub Actions** avec Quay.io

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   CHATBOT (Streamlit)                    │
│  - Interface utilisateur conversationnelle               │
│  - LangChain + GitHub Models (gpt-4o-mini)               │
│  - Client FastMCP                                        │
└─────────────────────────┬────────────────────────────────┘
                          │
                          │ MCP Protocol (Streamable HTTP)
                          │
┌─────────────────────────▼────────────────────────────────┐
│                   MCP SERVER (FastMCP)                   │
│  - Expose tool predict_survival via MCP                  │
│  - Transport: streamable-http                            │
│  - Health check: /health                                 │
└─────────────────────────┬────────────────────────────────┘
                          │
                          │ HTTP REST
                          │
┌─────────────────────────▼────────────────────────────────┐
│                   API ML (FastAPI)                       │
│  - Prédiction Random Forest                              │
│  - Monitoring OpenTelemetry + Jaeger                     │
│  - Health check                                          │
└──────────────────────────────────────────────────────────┘
```

**Voir [ARCHITECTURE_MCP.md](ARCHITECTURE_MCP.md) pour les détails techniques complets.**

## 🚀 Quick Start

### Prérequis

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de packages moderne)
- Compte GitHub (pour GitHub Models)

### Installation locale

```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Cloner le projet
git clone <repo-url>
cd mlops-template-SES-test

# Installer les dépendances
uv sync --all-groups

# Configurer les variables d'environnement
export OPENAI_API_KEY="your-github-token"
export OPENAI_BASE_URL="https://models.github.ai/inference"
```

### Lancer les services localement

#### 1. API ML
```bash
uv run python -m summit.api.main
# Accessible sur http://localhost:8080
```

#### 2. Serveur MCP
```bash
uv run python -m summit.mcp_server.server
# Accessible sur http://localhost:8000/mcp
```

#### 3. Chatbot
```bash
uv run streamlit run src/summit/chatbot/app.py
# Accessible sur http://localhost:8501
```

## 📊 MLflow Training

```bash
# Lancer un training local
uv run mlflow run ./src/summit/training \
  -e main \
  -P path=all_titanic.csv \
  --env-manager=local

# Accéder à MLflow UI (si déployé)
# http://mlflow-<namespace>.apps.openshift.com
```

## 🧪 Tests

```bash
# Lancer tous les tests
uv run pytest

# Tests avec coverage
uv run pytest --cov=src/summit --cov-report=html

# Linter
uv run ruff check src/
uv run ruff format src/
```

## 🐳 Déploiement Kubernetes/OpenShift

### Structure des manifests

```
k8s/
├── api/                    # API ML
│   ├── Dockerfile
│   └── api.yaml
├── chatbot/                # Chatbot Streamlit
│   ├── Dockerfile
│   └── chatbot.yaml
├── mcp_server/             # Serveur MCP
│   ├── Dockerfile
│   └── mcp-server.yaml
└── monitoring/             # Jaeger
    └── jaeger.yaml
```

### Déploiement manuel

```bash
# Déployer l'API ML
oc apply -f k8s/api/api.yaml

# Déployer le serveur MCP
oc apply -f k8s/mcp_server/mcp-server.yaml

# Déployer le chatbot
oc apply -f k8s/chatbot/chatbot.yaml

# Déployer Jaeger (monitoring)
oc apply -f k8s/monitoring/jaeger.yaml
```

### CI/CD

Le projet utilise GitHub Actions pour le déploiement automatique :

- **`.github/workflows/ct-ci-cd.yaml`** : Build et déploiement automatique sur push
- Images stockées sur **Quay.io**
- Déploiement automatique sur **OpenShift**

## 🔧 Configuration

### Variables d'environnement

#### Chatbot
- `MCP_SERVER_HOST` : URL du serveur MCP (défaut: service Kubernetes)
- `OPENAI_API_KEY` : Token GitHub pour GitHub Models
- `OPENAI_BASE_URL` : URL de l'API (défaut: GitHub Models)
- `LLM_MODEL` : Modèle à utiliser (défaut: gpt-4o-mini)

#### MCP Server
- `TITANIC_API_URL` : URL de l'API ML (défaut: service Kubernetes)

#### API ML
- `JAEGER_ENDPOINT` : URL de Jaeger pour les traces (défaut: service Kubernetes)

## 📁 Structure du projet

```
src/summit/
├── api/                    # API ML FastAPI
│   ├── main.py            # Point d'entrée
│   ├── infer.py           # Endpoint de prédiction
│   └── resources/         # Modèle pickle
├── mcp_server/            # Serveur MCP
│   └── server.py          # Serveur FastMCP (59 lignes)
├── chatbot/               # Interface Streamlit
│   ├── app.py             # Interface Streamlit
│   └── agent.py           # Agent LangChain (92 lignes)
└── training/              # Pipeline MLflow
    ├── main.py            # Workflow principal
    └── steps/             # Steps de training

tests/                     # Tests unitaires
k8s/                       # Manifests Kubernetes
.github/workflows/         # CI/CD GitHub Actions
```

## 🎓 Points pédagogiques

Ce template démontre :

### 1. **Model Context Protocol (MCP)**
- Communication standardisée entre agents et outils
- Transport Streamable HTTP moderne
- Bibliothèque FastMCP (simple et élégante)

### 2. **Architecture Microservices**
- Services découplés et indépendants
- Communication via APIs REST et MCP
- Scalabilité horizontale

### 3. **MLOps Best Practices**
- Tracking avec MLflow
- Monitoring avec OpenTelemetry + Jaeger
- CI/CD automatisé
- Tests unitaires

### 4. **Code Simple et Maintenable**
- **MCP Server** : 59 lignes seulement
- **Chatbot Agent** : 92 lignes ultra-propres
- Pattern async/await moderne
- Type hints partout

## 📚 Ressources

### Documentation
- [Architecture MCP détaillée](ARCHITECTURE_MCP.md)
- [FastMCP Documentation](https://gofastmcp.com)
- [GitHub Models](https://github.com/marketplace/models)
- [LangChain](https://python.langchain.com/)

### Dépendances principales
- **fastmcp** : Serveur et client MCP
- **langchain-openai** : Agent LLM
- **streamlit** : Interface chatbot
- **mlflow** : Tracking ML
- **opentelemetry** : Monitoring distribué

## 🤝 Contribution

Ce projet est un template d'enseignement. N'hésite pas à l'adapter pour tes besoins !

## 📝 Licence

Template éducatif - Usage libre pour l'enseignement

---

**Questions fréquentes** : Voir [ARCHITECTURE_MCP.md](ARCHITECTURE_MCP.md)

**Auteur** : Template MLOps pour SES
