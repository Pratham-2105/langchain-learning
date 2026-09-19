from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-8B",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Where is China bad to India")

print(result.content)