import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0.7,
    
)

result = model.invoke("Write a quote/saying about someone who has built himself from zero as a 17 year old when he came to college and is now 20 years old, with an internship at a quant firm. Keep it abuot 10 lines")

print(result.content)