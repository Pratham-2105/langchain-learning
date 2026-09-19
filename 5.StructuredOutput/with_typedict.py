import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# schema
class Review(TypedDict):
    summary: Annotated[str, "A bried summary of the review"]
    sentiment: Annotated[
        str, "Return sentiment of the review either negative, positive, or neutral"
    ]


load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """The hardware is graet, but the software feels bloated. There are too many pre-installed apps athat I can't remove. Also, the UI looks outdated compared to other branhds."""
)

print(result)
print(result["summary"])
print(result["sentiment"])
