# server.py
from mcp.server.fastmcp import FastMCP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, ImageUrl
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create an MCP server
mcp = FastMCP("R2J-MCP")

SYSTEM_PROMPT = """
You are a receipt parser. You will be given a image of receipt. Your task is to extract the following information from the receipt:
1. Name of the product or service
2. Date of payment
3. Currency of the payment
4. Amount paid
"""


@dataclass
class ReceiptContent:
    name : str
    date_of_payment : str
    currency : str
    amount : float


provider = OpenAIProvider(
    base_url=os.getenv("LM_STUDIO_IP"),
    api_key="LMSTUDIO",
)
model = OpenAIModel(
    model_name=os.getenv("LM_STUDIO_MODEL_VISION"),
    provider=provider,
)

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    output_type=ReceiptContent,
)


# input is image path, output is ReceiptContent
@mcp.tool()
async def parse_receipt(image_path: str) -> str:
    """
    Parse the receipt image and extract the required information.
    :param image_path: Path to the receipt image.
    :return: ReceiptContent object containing the extracted information.
    """
    result = agent.run_sync(
        [
            "What is the content?",
            ImageUrl(image_path),
        ],
    )
    return str(result.output)

# Run the server
if __name__ == "__main__":
    mcp.run()
