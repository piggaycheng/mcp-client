import asyncio
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from llama_index.core.agent.react import ReActAgent
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.llms.ollama import Ollama


class Pipeline:
    class Valves(BaseModel):
        MODEL: str = Field(default="")
        OLLAMA_HOST: str = Field(default="")
        DEMO_MCP_SERVER_URL: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()
        print("Pipeline initialized")

    async def on_startup(self):
        self.llm = Ollama(
            model=self.valves.MODEL,  # type: ignore
            base_url=self.valves.OLLAMA_HOST,
            request_timeout=60
        )
        self.mcp_client = BasicMCPClient(self.valves.DEMO_MCP_SERVER_URL)
        self.mcp_tool_spec = McpToolSpec(
            client=self.mcp_client,
            allowed_tools=[
                'search_repositories',
                'search_users',
                'list_commits'
            ]
        )
        print('startup complete')

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        tools = asyncio.run(self.mcp_tool_spec.to_tool_list_async())
        tools_str = asyncio.run(self.mcp_tool_spec.fetch_tools())
        print(tools_str, end="\n\n")
        agent = ReActAgent.from_tools(
            llm=self.llm,
            tools=tools,
            verbose=True,
        )
        response = asyncio.run(agent.achat(user_message))
        print(response)

        return str(response)
