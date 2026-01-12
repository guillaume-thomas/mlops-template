#!/usr/bin/env python3
"""
Script de test simple pour vérifier le fonctionnement du serveur MCP avec HTTP Streamable.
"""

import asyncio
import sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def test_mcp_server():
    """Test de connexion au serveur MCP et appel du tool predict_survival."""

    mcp_url = "http://localhost:8000/mcp"

    print("🔍 Test de connexion au serveur MCP...")
    print(f"   URL: {mcp_url}")

    try:
        # Connexion au serveur MCP via HTTP Streamable
        async with streamablehttp_client(mcp_url, timeout=5.0) as (read_stream, write_stream, _):
            print("✅ Connexion établie au serveur MCP")

            # Créer une session client
            session = ClientSession(read_stream, write_stream)

            async with session:
                # Initialiser la session
                await asyncio.wait_for(session.initialize(), timeout=5.0)
                print("✅ Session MCP initialisée")

                # Lister les tools disponibles
                tools_result = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                print(f"✅ Tools disponibles: {len(tools_result.tools)}")

                for tool in tools_result.tools:
                    print(f"   - {tool.name}: {tool.description}")

                # Tester le tool predict_survival
                print("\n🧪 Test du tool 'predict_survival'...")
                result = await asyncio.wait_for(
                    session.call_tool("predict_survival", {
                        "pclass": 3,
                        "sex": "male",
                        "sibsp": 0,
                        "parch": 0
                    }),
                    timeout=5.0
                )

                if result.content:
                    print(f"✅ Résultat: {result.content[0].text}")
                else:
                    print("❌ Aucun résultat retourné")

    except asyncio.TimeoutError:
        print("❌ Timeout - Le serveur MCP ne répond pas")
        print("   Assurez-vous que le serveur MCP est démarré : uv run python -m summit.mcp_server.server")
        sys.exit(1)
    except ConnectionRefusedError:
        print("❌ Connection refused - Le serveur MCP n'est pas accessible")
        print("   Assurez-vous que le serveur MCP est démarré sur le port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✨ Tous les tests ont réussi !")


if __name__ == "__main__":
    print("=" * 60)
    print("Test MCP avec HTTP Streamable")
    print("=" * 60)
    print()

    asyncio.run(test_mcp_server())

