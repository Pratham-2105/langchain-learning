import os  # noqa: I001

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
)   

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}.',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'BMW bikes'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1.content})
result2 = model.invoke(prompt2)

print(result1.content)
