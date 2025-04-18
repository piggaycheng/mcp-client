import os
from typing import Optional, Literal

from dotenv import load_dotenv

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
# from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.agent.react import ReActAgent
from llama_index.core.agent.function_calling import FunctionCallingAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama


# Load environment variables from .env file
load_dotenv()
DEMO_MCP_SERVER_URL = os.getenv("DEMO_MCP_SERVER_URL")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, request_timeout=60)


class MCPClient():
    def __init__(self):
        self.mcp_client: Optional[BasicMCPClient] = None

    async def connect_to_server(self, transport_type=Literal['stdio', 'sse']):
        """Connect to an MCP server"""

        if transport_type == 'sse':
            self.mcp_client = BasicMCPClient(DEMO_MCP_SERVER_URL)
            self.mcp_tool_spec = McpToolSpec(client=self.mcp_client)
        elif transport_type == 'stdio':
            self.mcp_client = BasicMCPClient(
                command_or_url='wsl',
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
            self.mcp_tool_spec = McpToolSpec(client=self.mcp_client)

        response = await self.mcp_client.list_tools()
        print("Tools: ", [tool.name for tool in response.tools], end="\n\n")

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available tools"""

        tools = await self.mcp_tool_spec.to_tool_list_async()
        agent = FunctionCallingAgent.from_tools(
            llm=llm,
            tools=tools,
            verbose=True,
        )
        response = await agent.aquery(query)
        print("Response:", response)


async def run():
    client = MCPClient()
    await client.connect_to_server(transport_type='stdio')
    await client.process_query("Get repo mcp-server which owned by piggaycheng in github, branch main")


# -------------------------test----------------------------
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


tools = [
    FunctionTool.from_defaults(multiply),
]
# print(tools[0].metadata)


async def test():
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
    )
    print(await agent.achat("What is 1234 * 4567?"))
