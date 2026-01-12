# 🏗️ Architecture MCP - Explication Simple

## Vue d'ensemble

Ce projet utilise le **Model Context Protocol (MCP)** pour permettre à un chatbot IA d'appeler des outils de manière standardisée.

## Les 3 services

```
┌─────────────────────────────────────────────────────────────────┐
│                         CHATBOT                                 │
│  Interface Streamlit + LangChain + GitHub Models (gpt-4o-mini) │
│              (Client MCP Streamable HTTP)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Streamable HTTP Transport
                             │ JSON-RPC over HTTP POST
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                      MCP SERVER                                 │
│          FastMCP exposant des "tools" via protocole MCP         │
│                    Tool: predict_survival                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP REST
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                        API ML                                   │
│         FastAPI avec modèle Random Forest (Titanic)             │
│                  Endpoint: POST /infer                          │
└─────────────────────────────────────────────────────────────────┘
```

## Comment ça fonctionne ?

### 1. API ML (`src/summit/api/`)
**Rôle** : Faire des prédictions de survie sur le Titanic

- Exposé sur le port **8080**
- Endpoint : `POST /infer`
- Reçoit : `{pclass: 3, sex: "male", sibSp: 0, parch: 0}`
- Retourne : `[0]` ou `[1]` (liste)

**C'est un service ML classique.**

---

### 2. MCP Server (`src/summit/mcp_server/`)
**Rôle** : Exposer l'API ML comme un "tool" MCP

- Exposé sur le port **8000**
- Endpoint HTTP Streamable : `POST /mcp` (pour le protocole MCP)
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

#### Qu'est-ce que le transport Streamable HTTP ?
**Streamable HTTP** est le transport moderne recommandé pour MCP. Il permet :
- Communication bidirectionnelle sur HTTP standard
- JSON-RPC pour les messages structurés
- Support du streaming pour les réponses longues
- Compatibilité avec les infrastructures HTTP existantes

C'est le transport par défaut pour tous les nouveaux projets MCP.

#### Pourquoi utiliser FastMCP ?

**FastMCP** est une API haut niveau qui simplifie la création de serveurs MCP :

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="titanic-mcp-server",
    streamable_http_path="/mcp",  # Endpoint HTTP Streamable
)

@mcp.tool()
def predict_survival(pclass: int, sex: str, sibsp: int, parch: int) -> str:
    # Votre logique ici
    return "survived" or "did not survive"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

**Avantages de FastMCP** :
- Décorateurs simples (`@mcp.tool()`)
- Gère automatiquement le transport HTTP Streamable
- Validation automatique des schémas
- Moins de code boilerplate

---

### 3. Chatbot (`src/summit/chatbot/`)
**Rôle** : Interface utilisateur avec IA

- Exposé sur le port **8501**
- Interface : Streamlit
- IA : GitHub Models (gpt-4o-mini) via LangChain
- Client MCP : Se connecte au MCP Server via HTTP Streamable

**C'est l'interface utilisateur.**

#### Comment le client se connecte au serveur MCP ?

```python
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Se connecter au serveur MCP
async with streamablehttp_client("http://mcp-server:8000/mcp") as (read, write, _):
    session = ClientSession(read, write)
    await session.initialize()

    # Lister les tools disponibles
    tools = await session.list_tools()

    # Appeler un tool
    result = await session.call_tool("predict_survival", {
        "pclass": 3,
        "sex": "male",
        "sibsp": 1,
        "parch": 0
    })
```

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
       "http://titanic-api:8080/infer",
       json={"pclass": 3, "sex": "male", "sibSp": 0, "parch": 0}
   )
   ```
   - Fichier : `src/summit/mcp_server/server.py` (fonction `predict_survival`)

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
