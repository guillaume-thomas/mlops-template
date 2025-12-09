# MCP Server - Titanic Prediction

Ce module implémente un **serveur MCP (Model Context Protocol)** déployé comme service Kubernetes indépendant qui expose un outil de prédiction de survie des passagers du Titanic.

## Architecture Microservices

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Streamlit UI   │ ──────> │  LangChain Agent │ ──────> │   MCP Client    │
│  (chatbot)      │         │  (agent.py)      │         │   (SSE/HTTP)    │
└─────────────────┘         └──────────────────┘         └────────┬────────┘
                                                                   │
                                                            SSE (JSON-RPC)
                                                                   │
                                                          ┌────────▼────────┐
                                                          │   MCP Server    │  ← Service K8s
                                                          │  (FastAPI/SSE)  │
                                                          └────────┬────────┘
                                                                   │
                                                                  HTTP
                                                                   │
                                                          ┌────────▼────────┐
                                                          │  Titanic API    │
                                                          │  (FastAPI)      │
                                                          └─────────────────┘
```

## Structure des fichiers

```
src/summit/mcp_server/
├── __main__.py          # Point d'entrée pour lancer le serveur
├── server.py            # Serveur MCP avec FastAPI et SSE
├── titanic_tool.py      # Logique métier (appel API Titanic)
└── README.md            # Documentation

k8s/mcp_server/
├── Dockerfile           # Image Docker du serveur MCP
└── mcp-server.yaml      # Manifests Kubernetes (Deployment, Service)

.github/workflows/
└── deploy-mcp-server.yml # CI/CD du serveur MCP
```

## Fonctionnement

### 1. Serveur MCP (`server.py`)

Le serveur est une application **FastAPI** qui expose :
- `/sse` : Endpoint SSE pour la communication MCP (JSON-RPC)
- `/health` : Endpoint de health check pour Kubernetes

Le serveur expose un **tool** via le protocole MCP :

**Tool : `predict_survival`**
- **Description** : Prédit si un passager du Titanic aurait survécu
- **Paramètres** :
  - `pclass` (int) : Classe du passager (1=Upper, 2=Middle, 3=Low)
  - `sex` (string) : Genre ('male' ou 'female')
  - `sibsp` (int) : Nombre de frères/sœurs/conjoints à bord
  - `parch` (int) : Nombre de parents/enfants à bord
- **Retour** : Texte décrivant la prédiction

### 2. Communication stdio

Le serveur utilise **stdio (Standard Input/Output)** pour communiquer avec le client :
- Messages JSON-RPC échangés via stdin/stdout
- Le client lance le serveur en subprocess
- Communication asynchrone (asyncio)

### 3. Client MCP (`agent.py`)

L'agent LangChain intègre le client MCP :
1. Lance le serveur MCP en subprocess
2. Établit une session via stdio
3. Liste les tools disponibles
4. Convertit les tools MCP en tools LangChain
5. Utilise les tools via le LLM

## Variables d'environnement

### Serveur MCP
- `TITANIC_API_URL` : URL de l'API Titanic (requis)
- `PORT` : Port du serveur (défaut: 8000)

### Chatbot (client)
- `MCP_SERVER_URL` : URL SSE du serveur MCP (défaut: service Kubernetes)
- `OPENAI_API_KEY` : Token GitHub Models
- `OPENAI_BASE_URL` : Base URL pour GitHub Models
- `LLM_MODEL` : Modèle à utiliser (défaut: gpt-4o-mini)

## Test local du serveur MCP

### 🚀 Lancer le serveur MCP en local

```bash
# Terminal 1 : Lancer le serveur MCP
export TITANIC_API_URL="http://mlops-api-service.gthomas59800-dev.svc.cluster.local:8080"
uv run --group mcp-server python -m summit.mcp_server.server
```

Le serveur démarre sur `http://localhost:8000`

Endpoints disponibles :
- `GET /health` : Health check
- `GET /sse` : Endpoint SSE pour MCP

### 🧪 Tester le serveur

#### 1. Health check
```bash
curl http://localhost:8000/health
```

#### 2. Tests unitaires avec pytest

Assurez-vous que le serveur MCP est lancé, puis :

```bash
# Installer les dépendances de test
uv sync --group dev --group mcp-server

# Lancer tous les tests du serveur MCP
uv run --group dev pytest tests/mcp_server/ -v

# Lancer un test spécifique
uv run --group dev pytest tests/mcp_server/test_server.py::test_mcp_server_connection -v

# Lancer les tests avec output détaillé
uv run --group dev pytest tests/mcp_server/ -v -s
```

Si le serveur MCP n'est pas disponible, les tests seront automatiquement skippés.

#### 3. Tests du chatbot (agent)

```bash
# Lancer les tests de l'agent
uv run --group dev --group chatbot pytest tests/chatbot/ -v
```

#### 4. Lancer tous les tests

```bash
# Tests complets (nécessite serveur MCP et API Titanic lancés)
uv run --group dev pytest tests/ -v
```

#### 5. Tester manuellement avec le client complet (chatbot)
```bash
# Terminal 2 : Lancer le chatbot
export MCP_SERVER_URL="http://localhost:8000/sse"
export OPENAI_API_KEY="ghp_YOUR_GITHUB_TOKEN"
export TITANIC_API_URL="http://localhost:8080"
uv run --group chatbot streamlit run src/summit/chatbot/app.py
```

Ouvrir http://localhost:8501

## Avantages de MCP

### Pour un cours pédagogique

✅ **Protocole standardisé** : Les étudiants apprennent un vrai standard industriel
✅ **Architecture distribuée** : Séparation client/serveur claire
✅ **Réutilisabilité** : Le serveur MCP peut être utilisé par plusieurs clients
✅ **Isolation** : La logique métier est séparée de l'agent
✅ **Asynchrone** : Apprentissage de asyncio et communication async

### Concepts enseignés

1. **Model Context Protocol** : Standard émergent pour exposer des outils aux LLM
2. **JSON-RPC** : Communication via messages structurés
3. **stdio** : Communication inter-processus via standard input/output
4. **asyncio** : Programmation asynchrone en Python
5. **LangChain Tools** : Intégration d'outils externes dans un agent
6. **Architecture microservices** : Séparation des responsabilités

## Déploiement

Dans Kubernetes, le serveur MCP tourne dans le même pod que le chatbot :
- Le client lance le serveur en subprocess via stdio
- Pas besoin de service Kubernetes séparé
- Communication locale (pas de réseau)

## Comparaison avec l'approche simple

### Avant (approche simple)

```python
# Direct HTTP call
result = httpx.post(api_url, json=data)
```

### Maintenant (avec MCP)

```python
# Via MCP protocol
result = await mcp_session.call_tool("predict_survival", args)
```

**Différence** : 
- Avant : Appel HTTP direct depuis l'agent
- Maintenant : Appel via protocole MCP standardisé, le serveur MCP fait l'appel HTTP

**Avantage pédagogique** : Montre une architecture plus complexe et réaliste !

---

## 📊 Flux de données complet (Documentation pédagogique)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERACTION                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          "Would a first-class
                          woman survive?"
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT UI (app.py)                              │
│  - Affiche l'interface chat                                                 │
│  - Gère l'historique des messages                                           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          agent.chat(message)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGCHAIN AGENT (agent.py)                           │
│  1. Envoie le message au LLM avec les tools disponibles                     │
│  2. Le LLM décide d'appeler le tool "predict_survival"                      │
│  3. LLM génère les arguments : {pclass: 1, sex: "female", ...}              │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          tool_call(name, args)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP CLIENT (agent.py)                              │
│  - Communique avec le serveur MCP via stdio                                 │
│  - Envoie requête JSON-RPC                                                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          stdio (JSON-RPC)
                          {
                            "method": "tools/call",
                            "params": {
                              "name": "predict_survival",
                              "arguments": {...}
                            }
                          }
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCP SERVER (server.py)                              │
│  1. Reçoit la requête JSON-RPC                                              │
│  2. Extrait le nom du tool et les arguments                                 │
│  3. Appelle la logique métier (TitanicInferenceTool)                        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          predict_survival(...)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TITANIC TOOL (titanic_tool.py)                         │
│  - Prépare la requête HTTP                                                  │
│  - Appelle l'API Titanic                                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          HTTP POST /infer
                          {
                            "pclass": 1,
                            "sex": "female",
                            "sibSp": 0,
                            "parch": 0
                          }
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TITANIC API (FastAPI)                              │
│  - Reçoit les caractéristiques du passager                                  │
│  - Appelle le modèle Random Forest                                          │
│  - Retourne la prédiction                                                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                          [1] (survived)
                                 │
                                 ▼
                        (Remonte la chaîne)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER SEES                                       │
│  "Based on the characteristics provided, the passenger survived ✅"          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Détails techniques

### 1. Communication MCP (stdio)

Le **Model Context Protocol** utilise **stdio** (Standard Input/Output) pour la communication :

```python
# Le client lance le serveur en subprocess
server_params = StdioServerParameters(
    command="python",
    args=["-m", "summit.mcp_server.server"]
)

# Communication via stdin/stdout
async with stdio_client(server_params) as (read_stream, write_stream):
    # Création d'une session MCP
    async with ClientSession(read_stream, write_stream) as session:
        # Appel d'un tool
        result = await session.call_tool("predict_survival", {...})
```

**Avantages** :
- ✅ Pas besoin de réseau (local)
- ✅ Isolation des processus
- ✅ Communication asynchrone
- ✅ Standard JSON-RPC

### 2. Messages JSON-RPC

Exemple de requête du client vers le serveur :

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "predict_survival",
    "arguments": {
      "pclass": 1,
      "sex": "female",
      "sibsp": 0,
      "parch": 0
    }
  }
}
```

Réponse du serveur :

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Based on the passenger characteristics, they survived. (Prediction: 1)"
      }
    ]
  }
}
```

### 3. Déclaration des tools

Le serveur MCP déclare ses tools avec un **schéma JSON** :

```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="predict_survival",
            description="Predict if a Titanic passenger would survive",
            inputSchema={
                "type": "object",
                "properties": {
                    "pclass": {"type": "integer", "enum": [1, 2, 3]},
                    "sex": {"type": "string", "enum": ["male", "female"]},
                    ...
                },
                "required": ["pclass", "sex", "sibsp", "parch"]
            }
        )
    ]
```

Le LLM utilise ce schéma pour :
- Comprendre quels tools sont disponibles
- Connaître les paramètres requis
- Générer les arguments corrects

---

## 🎯 Concepts pédagogiques enseignés

### 1. Model Context Protocol (MCP)
Standard émergent pour connecter des LLM à des outils externes de manière sécurisée et standardisée.

### 2. JSON-RPC
Protocole de communication basé sur JSON pour appeler des méthodes distantes.

### 3. stdio (Standard Input/Output)
Communication inter-processus via stdin/stdout, alternative aux sockets réseau.

### 4. asyncio
Programmation asynchrone en Python pour gérer les I/O non-bloquantes.

### 5. LangChain Tools
Framework pour intégrer des outils externes dans un agent LLM.

### 6. Tool Calling
Capacité d'un LLM à décider quand et comment appeler des outils externes.

---

## 📚 Comparaison détaillée : Simple vs MCP

### Approche Simple (sans MCP)

```python
# L'agent appelle directement l'API
@tool
def predict_survival(...):
    response = httpx.post(api_url, json=data)
    return response.json()
```

**Avantages** :
- ✅ Simple à comprendre
- ✅ Moins de code

**Inconvénients** :
- ❌ Couplage fort agent/API
- ❌ Pas de protocole standard
- ❌ Difficile à réutiliser

### Approche MCP (avec serveur)

```python
# L'agent appelle le serveur MCP
result = await mcp_session.call_tool("predict_survival", args)

# Le serveur MCP appelle l'API
response = httpx.post(api_url, json=data)
```

**Avantages** :
- ✅ Protocole standardisé
- ✅ Séparation des responsabilités
- ✅ Réutilisable par d'autres clients
- ✅ Plus pédagogique !

**Inconvénients** :
- ❌ Plus de code
- ❌ Plus complexe

---

## 🚀 Évolutions possibles pour le cours

Pour enrichir le cours, vous pourriez ajouter :

1. **Plusieurs tools** : Ajouter d'autres tools MCP (météo, news, calculs)
2. **Ressources MCP** : Exposer des ressources (fichiers, base de données) via MCP
3. **Prompts MCP** : Exposer des prompts prédéfinis
4. **Multiple clients** : Montrer comment plusieurs agents utilisent le même serveur
5. **Sécurité** : Ajouter de l'authentification entre client et serveur
6. **Monitoring** : Logger les appels MCP pour debugging et analyse

---

## 📖 Ressources externes

- **MCP Spec** : https://spec.modelcontextprotocol.io/
- **MCP SDK Python** : https://github.com/modelcontextprotocol/python-sdk
- **LangChain MCP** : https://python.langchain.com/docs/integrations/tools/

