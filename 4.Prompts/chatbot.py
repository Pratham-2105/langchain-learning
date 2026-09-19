import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
)

chat_history = [
    SystemMessage(content='You are a helpful AI assisstant in career stuff')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(user_input))

    if user_input == 'exit':
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))

    print("AI: ", result.content)

print(chat_history)


"""
3 types of messages in LangChain

1. System message
2. User message
3. AI message
"""