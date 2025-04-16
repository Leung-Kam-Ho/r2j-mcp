# server.py
from mcp.server.fastmcp import FastMCP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent, BinaryContent
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create an MCP server
mcp = FastMCP("R2J-MCP")

SYSTEM_PROMPT = f"""
Today's date is {datetime.now().strftime('%Y-%m-%d')}.
You are a receipt parser. You will be given a image of receipt. Your task is to extract information from the receipt:
"""


@dataclass
class ReceiptContent:
    """
    name : str (name of the item)
    date_of_payment : str (date of payment in YYYY-MM-DD format)
    currency : str (currency of the payment e.g USD, EUR)
    amount : float (amount of the payment)
    """
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
    # convert image path to binary content
    # check if file exists
    # if not os.path.exists(image_path):
    #     return f"File {image_path} does not exist."
    with open(image_path, "rb") as image_file:
        image_byte = image_file.read()
        print(f"Image byte size: {len(image_byte)}")
    # run the agent
    result = await agent.run(
        [
            "What is the content?",
            BinaryContent(data=image_byte, media_type='image/png'),
        ],
    )
    return result.output

# Run the server
if __name__ == "__main__":
    mcp.run()
