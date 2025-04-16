# server.py
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent
import logfire
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
logfire.configure()

fetch_server = MCPServerStdio('uv', ['run','mcp-server-fetch'])
r2j_server = MCPServerStdio('uv',['run','main.py'])

provider = OpenAIProvider(
    base_url=os.getenv("LM_STUDIO_IP"),
    api_key="LMSTUDIO",
)
model = OpenAIModel(
    model_name=os.getenv("LM_STUDIO_MODEL_AGENT"),
    provider=provider,
)

agent = Agent(
    model=model,
    instrument=True,
    mcp_servers=[
        fetch_server,
        r2j_server,
    ],
    
)


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run("hello!")
        while True:
            print(f"\n{result.output}")
            user_input = input("\n>")
            result = await agent.run(
                user_input,
                message_history=result.new_messages()
            )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())