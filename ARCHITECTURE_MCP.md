# 🏗️ Architecture MCP - Explication Simple

## Vue d'ensemble

Ce projet utilise le **Model Context Protocol (MCP)** pour permettre à un chatbot IA d'appeler des outils de manière standardisée.

## Les 3 services

```
┌─────────────────────────────────────────────────────────────────┐
│                         CHATBOT                                 │
│  Interface Streamlit + LangChain + GitHub Models (gpt-4o-mini) │
│                    (Client MCP SSE)                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ SSE (Server-Sent Events)
                             │ JSON-RPC over HTTP
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      MCP SERVER                                 │
│          FastAPI exposant des "tools" via protocole MCP         │
│                    Tool: predict_survival                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP REST
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        API ML                                   │
│         FastAPI avec modèle Random Forest (Titanic)             │
│                  Endpoint: POST /predict                        │
└─────────────────────────────────────────────────────────────────┘
```

## Comment ça fonctionne ?

### 1. API ML (`src/summit/api/`)
**Rôle** : Faire des prédictions de survie sur le Titanic

- Exposé sur le port **8080**
- Endpoint : `POST /predict`
- Reçoit : `{pclass: 3, sex: "male", sibsp: 0, parch: 0}`
- Retourne : `{prediction: 0, probability: 0.15}`

**C'est un service ML classique.**

---

### 2. MCP Server (`src/summit/mcp_server/`)
**Rôle** : Exposer l'API ML comme un "tool" MCP

- Exposé sur le port **8000**
- Endpoint SSE : `GET /sse` (pour le protocole MCP)
- Tool MCP : `predict_survival`
- Appelle l'API ML en interne

**C'est un wrapper MCP autour de l'API ML.**

#### Qu'est-ce qu'un "tool" MCP ?
Un tool MCP est une fonction que l'IA peut appeler :
```json
{
  "name": "predict_survival",
  "description": "Predict if a Titanic passenger would survive",
  "inputSchema": {
    "pclass": "integer",
    "sex": "string",
    "sibsp": "integer",
    "parch": "integer"
  }
}
```

#### Qu'est-ce que SSE ?
**Server-Sent Events** : un protocole HTTP qui permet au serveur d'envoyer des messages en continu au client. Utilisé par MCP pour la communication asynchrone.

#### Pourquoi GET pour `/sse` ?

Le endpoint SSE utilise **GET** (et non POST) car :

1. **Flux unidirectionnel** : SSE est conçu pour des flux de données serveur → client
2. **Standard HTTP** : SSE est basé sur la spécification HTML5 qui utilise GET
3. **Connexion persistante** : Le client ouvre une connexion HTTP longue durée avec GET
4. **Media type spécial** : `text/event-stream` indique au navigateur/client que c'est du SSE

**Comment ça fonctionne avec le SDK MCP** :

```python
from mcp.server.sse import SseServerTransport
from starlette.routing import Route, Mount

# Créer le transport SSE
sse = SseServerTransport("/messages")

# Handler SSE utilisant le SDK
async def handle_sse(request: Request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp.run(
            streams[0],  # Read stream
            streams[1],  # Write stream  
            mcp.create_initialization_options()
        )
    return Response()  # Important : retourner Response vide

# Routes Starlette
routes = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount("/messages", app=sse.handle_post_message),  # ← Gère les POSTs automatiquement
]
```

**Ce qui se passe** :
1. Le client fait `GET /sse` et garde la connexion ouverte
2. Le serveur envoie `event: endpoint` avec `data: /messages`
3. Le client envoie ses messages JSON-RPC via `POST /messages/*`
4. Le SDK `SseServerTransport` gère automatiquement la liaison entre les deux
5. Les réponses sont streamées via SSE
6. **Tout est géré par le SDK !** (sessions, queues, format SSE, etc.)

---

### 3. Chatbot (`src/summit/chatbot/`)
**Rôle** : Interface utilisateur avec IA

- Exposé sur le port **8501**
- Interface : Streamlit
- IA : GitHub Models (gpt-4o-mini) via LangChain
- Client MCP : Se connecte au MCP Server

**C'est l'interface utilisateur.**

---

## Flux complet - Exemple concret

### Question utilisateur
```
"Un homme de 3ème classe avec 1 frère survivrait-il ?"
```

### Étape par étape

1. **Streamlit** reçoit la question
   - Fichier : `src/summit/chatbot/app.py`
   
2. **LangChain Agent** analyse la question avec le LLM
   - Fichier : `src/summit/chatbot/agent.py`
   - LLM : GitHub Models (gpt-4o-mini)
   - Décision : "Je dois appeler le tool `predict_survival`"

3. **Client MCP** se connecte au serveur
   - Connexion SSE : `http://titanic-mcp-server:8000/sse`
   - Fichier : `src/summit/chatbot/agent.py` (ligne 34-37)

4. **Client MCP** appelle le tool
   ```python
   result = await mcp_session.call_tool(
       "predict_survival",
       {"pclass": 3, "sex": "male", "sibsp": 1, "parch": 0}
   )
   ```
   - Fichier : `src/summit/chatbot/agent.py` (ligne 69)

5. **MCP Server** reçoit l'appel
   - Endpoint : `GET /sse`
   - Handler : `call_tool` dans `src/summit/mcp_server/server.py` (ligne 40)

6. **Titanic Tool** fait l'appel à l'API ML
   ```python
   response = requests.post(
       "http://titanic-api:8080/predict",
       json={"pclass": 3, "sex": "male", "sibsp": 1, "parch": 0}
   )
   ```
   - Fichier : `src/summit/mcp_server/titanic_tool.py` (ligne 45)

7. **API ML** retourne la prédiction
   ```json
   {"prediction": 0, "probability": 0.18}
   ```

8. **MCP Server** retourne au client
   ```json
   {
     "prediction": "Did not survive",
     "probability": 0.18,
     "confidence": "low"
   }
   ```

9. **LangChain Agent** reformule pour l'utilisateur
   - "Non, un homme de 3ème classe avec 1 frère n'aurait probablement pas survécu (18% de chances)"

10. **Streamlit** affiche la réponse

---

## Pourquoi utiliser MCP ?

### Sans MCP (approche directe)
```
Chatbot → LLM → Code custom pour appeler API ML
```
❌ Chaque outil nécessite du code spécifique
❌ Pas de standard
❌ Difficile à maintenir

### Avec MCP (approche standardisée)
```
Chatbot → LLM → MCP Client → MCP Server → API ML
```
✅ Standard ouvert (créé par Anthropic)
✅ L'IA découvre automatiquement les outils disponibles
✅ Ajout de nouveaux outils facile
✅ Architecture microservices
✅ Réutilisable par d'autres clients (Claude Desktop, Continue, etc.)

---

## Déploiement Kubernetes

### 3 déploiements indépendants

```yaml
# API ML
Service: titanic-api
Port: 8080
Image: quay.io/gthomas59800/summit/api

# MCP Server
Service: titanic-mcp-server
Port: 8000
Image: quay.io/gthomas59800/summit/mcp-server

# Chatbot
Service: titanic-chatbot
Port: 8501
Image: quay.io/gthomas59800/summit/chatbot
Route: titanic-chatbot-{namespace}.apps.{cluster}.com
```

### Communication entre services

```
Chatbot → titanic-mcp-server:8000 (SSE/MCP)
MCP Server → titanic-api:8080 (HTTP/REST)
```

---

## Avantages pédagogiques

Ce projet est **excellent pour un cours MLOps** car il montre :

1. **Architecture microservices** : 3 services indépendants
2. **Protocole standardisé** : MCP pour l'intégration IA-outils
3. **CI/CD** : 3 workflows GitHub Actions séparés
4. **Kubernetes** : Déploiement, services, routes
5. **Observabilité** : Traces OpenTelemetry (Jaeger)
6. **MLflow** : Gestion du modèle ML
7. **IA moderne** : LangChain + GitHub Models (gratuit)

---

## Technologies utilisées

| Composant | Technologies |
|-----------|-------------|
| **API ML** | FastAPI, scikit-learn, MLflow |
| **MCP Server** | FastAPI, MCP SDK, SSE |
| **Chatbot** | Streamlit, LangChain, GitHub Models (gpt-4o-mini) |
| **Infrastructure** | Kubernetes (OpenShift), Docker, Quay.io |
| **CI/CD** | GitHub Actions |
| **Observabilité** | OpenTelemetry, Jaeger |

---

## Pour aller plus loin

### Documentation MCP
- Spécification : https://spec.modelcontextprotocol.io/
- SDK Python : https://github.com/modelcontextprotocol/python-sdk
- Exemples : https://modelcontextprotocol.io/examples

### Ajouter un nouveau tool MCP

1. Créer le tool dans `src/summit/mcp_server/`
2. L'enregistrer dans `server.py`
3. Redéployer le MCP Server
4. Le chatbot le découvre automatiquement !

**Aucune modification du chatbot nécessaire** grâce à MCP ! 🎉

