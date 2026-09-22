
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="openai/gpt-oss-20b",
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = "https://www.amazon.in/Apple-2024-Desktop-Computer-10%E2%80%91core/dp/B0DLBTLQZP/ref=sr_1_1_sspa?crid=MXZGI78CCA10&dib=eyJ2IjoiMSJ9.sLkmxoVwMOVTi3JUdpD1VyJBLketSFoyHN3m5rLQuwxSlnOpQIrjx1N2D_2x4_kAH1I7LM5LwKT0bYXdW19LUUW3bYadq0AopvCFV8PSNbZR2Nu-zbGQSdAf0AA5PidBnyZS9rWFL6_QqFjhUoYM4moo-EakKe9NfrDJpZZSF5Ii8E30CaV5KjV-3yI-KZHRzxIORbIPbrz7f2pyhgnyD145gopzza_AuSX93Mm6psU.x9KDE7QcFsUsQMu1kR_DDcYL4Fr3ibHf708zzRmcGg8&dib_tag=se&keywords=mac+mini&qid=1790086615&sprefix=mac+mi%2Caps%2C271&sr=8-1-spons&aref=alOnzPOOaD&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"


loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question':'What are the specifications of this product?', 'text':docs[0].page_content})

print(result)