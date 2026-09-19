import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result)
