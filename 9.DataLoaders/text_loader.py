import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="openai/gpt-oss-20b",
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poems']
)

loader = TextLoader('9.DL/cricket.txt', encoding='utf-8')

docs = loader.load()

#print(docs)
# print(type(docs))
# print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({'poem':docs[0].page_content})
