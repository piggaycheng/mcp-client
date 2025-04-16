import os
from typing import Optional, Literal
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from ollama import AsyncClient
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


# Load environment variables from .env file
load_dotenv()
DEMO_MCP_SERVER_URL = os.getenv("DEMO_MCP_SERVER_URL")
DEMO_MCP_SERVER_PYTHON_PATH = os.getenv("DEMO_MCP_SERVER_PYTHON_PATH")
DEMO_MCP_SERVER_SCRIPT_PATH = os.getenv("DEMO_MCP_SERVER_SCRIPT_PATH")


class MCPClient():
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.mcp_client: Optional[Client] = None

    async def connect_to_server(self, transport_type=Literal['stdio', 'sse']):
        """Connect to an MCP server"""

        if transport_type == 'sse':
            self.mcp_client = await self.exit_stack.enter_async_context(Client(DEMO_MCP_SERVER_URL))
        elif transport_type == 'stdio':
            transport = StdioTransport(
                command=DEMO_MCP_SERVER_PYTHON_PATH,
                args=[DEMO_MCP_SERVER_SCRIPT_PATH],
            )
            self.mcp_client = await self.exit_stack.enter_async_context(Client(transport=transport))

        response = await self.mcp_client.list_tools()
        tools = response
        print("Connected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available tools"""
        response = await self.mcp_client.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response]
        print("Available tools:", available_tools)
        messages = [
            {
                "role": "system",
                "content": """
                    You are an assistant that should decide which tool to use based on the query and the available tools.
                    Should use the following JSON schema to format the response:
                    {{
                        "properties": {{
                            "tool": {{
                                "description": "Tool to use",
                                "type": "string"
                            }},
                            "arguments": {{
                                "description": "Function arguments",
                                "type": "object"
                            }}
                        }}
                    }}
                    Available tools: {available_tools}
                    If no tool is suitable, tool should be empty string and parameters should be empty.
                """.format(available_tools=str(available_tools))
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response = await AsyncClient(
            host=os.getenv("OLLAMA_HOST"),
        ).chat(
            model=os.getenv("OLLAMA_MODEL"),
            messages=messages,
            # tools=available_tools,
        )
        print("Response from Ollama:", response.message.content)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def run():
    client = MCPClient()
    try:
        await client.connect_to_server(transport_type='stdio')
        await client.process_query("what is the weather in Taitong?")
    finally:
        await client.cleanup()
