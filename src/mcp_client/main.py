import os
from typing import Optional
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, types
from mcp.client.sse import sse_client
from ollama import AsyncClient


# Load environment variables from .env file
load_dotenv()


class MCPClient():
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self):
        """Connect to an MCP server"""

        sse_transport = await self.exit_stack.enter_async_context(sse_client(os.getenv("DEMO_MCP_SERVER_URL")))
        read, write = sse_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))

        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("Connected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama and available tools"""

        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
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
        await client.connect_to_server()
        await client.process_query("what is the weather in Taitong?")
    finally:
        await client.cleanup()
