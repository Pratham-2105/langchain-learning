import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


weak_model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="openai/gpt-oss-20b",
)

strong_model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="meta/muse-glimmer-30b",
)

str_parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback."
    )


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

classify_prompt = PromptTemplate(
    template="Classify the sentiment/emotion of the following text into positive or negative \n {feedback}",
    input_variables=["feedback"],
    partial_variables={'format_instructions':pydantic_parser.get_format_instructions()}
)

positive_feedback_prompt = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

negative_feedback_prompt = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifier_chain = classify_prompt | weak_model | pydantic_parser

branch_chain = RunnableBranch(

    (lambda x:x.sentiment == 'positive', positive_feedback_prompt | strong_model | str_parser),
    (lambda x:x.sentiment == 'negative', negative_feedback_prompt | strong_model | str_parser),
    RunnableLambda(lambda x: "could not find sentiment")
    
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'This is a terrible phone'}))

chain.get_graph().print_ascii()
