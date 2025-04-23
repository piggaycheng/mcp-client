import asyncio
from mcp_client.main import run, test, run_sse
from open_webui.proxy_server import proxy_server


if __name__ == '__main__':
    proxy_server.run('sse', host='0.0.0.0', port=8000)
    # asyncio.run(run())
    # asyncio.run(test())
    # asyncio.run(run_sse())
