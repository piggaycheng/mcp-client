import os

from fastmcp import FastMCP, Client
from fastmcp.client.transports import StdioTransport

from dotenv import load_dotenv


load_dotenv()
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

transport = StdioTransport(
    command='wsl',
    args=[
        '-u',
        'root',
        'docker',
        'run',
        '--rm',
        '-i',
        '-e',
        'GITHUB_PERSONAL_ACCESS_TOKEN',
        'mcp/github',
    ],
    env={
        'GITHUB_PERSONAL_ACCESS_TOKEN': GITHUB_PERSONAL_ACCESS_TOKEN,
    }
)

client = Client(transport=transport)
proxy_server = FastMCP.from_client(client=client)
