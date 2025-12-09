#!/usr/bin/env python3
"""
Script de test local pour vérifier la connexion MCP server/client.
Usage:
  Terminal 1: python test_mcp_local.py server
  Terminal 2: python test_mcp_local.py client
"""
import sys
import asyncio
import os

async def run_server():
    """Lance le serveur MCP en local"""
    print("[SERVER] Starting MCP server on http://localhost:8000")
    os.environ["TITANIC_API_URL"] = "http://localhost:8080"
    os.environ["PORT"] = "8000"

    from summit.mcp_server.server import app
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def run_client():
    """Teste la connexion au serveur MCP"""
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    url = "http://localhost:8000/sse"
    print(f"[CLIENT] Connecting to {url}")

    try:
        async with sse_client(url) as (read_stream, write_stream):
            print("[CLIENT] ✅ SSE connection established")

            async with ClientSession(read_stream, write_stream) as session:
                print("[CLIENT] ✅ Client session created")

                print("[CLIENT] Initializing...")
                await asyncio.wait_for(session.initialize(), timeout=5.0)
                print("[CLIENT] ✅ Session initialized")

                print("[CLIENT] Listing tools...")
                result = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                print(f"[CLIENT] ✅ Tools found: {[tool.name for tool in result.tools]}")

                if result.tools:
                    tool = result.tools[0]
                    print(f"\n[CLIENT] Calling tool '{tool.name}'...")
                    response = await asyncio.wait_for(
                        session.call_tool(
                            tool.name,
                            {"pclass": 3, "sex": "male", "sibsp": 0, "parch": 0}
                        ),
                        timeout=5.0
                    )
                    print(f"[CLIENT] ✅ Tool response: {response.content[0].text if response.content else 'No content'}")

                print("\n[CLIENT] ✅ All tests passed!")

    except asyncio.TimeoutError:
        print("[CLIENT] ❌ Timeout - le serveur ne répond pas")
        print("[CLIENT] Vérifiez que:")
        print("  1. Le serveur MCP est lancé (python test_mcp_local.py server)")
        print("  2. Le serveur répond sur http://localhost:8000/health")
    except Exception as e:
        print(f"[CLIENT] ❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_mcp_local.py server  # Lance le serveur MCP")
        print("  python test_mcp_local.py client  # Teste la connexion au serveur")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "server":
        asyncio.run(run_server())
    elif mode == "client":
        asyncio.run(run_client())
    else:
        print(f"Mode inconnu: {mode}")
        print("Utilisez 'server' ou 'client'")
        sys.exit(1)

if __name__ == "__main__":
    main()

