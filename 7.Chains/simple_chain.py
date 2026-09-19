import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="openai/gpt-oss-20b"
)   

parser = StrOutputParser()

# Pipeline 
chain = prompt | model | parser

result = chain.invoke({'topic':'basketball'})

print(result)

chain.get_graph().print_ascii()