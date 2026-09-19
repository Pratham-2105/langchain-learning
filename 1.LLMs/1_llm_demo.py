import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openai/gpt-oss-20b"
)


result = model.invoke("What is the capital of India")

print(result)
