import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

load_dotenv()

# Only the ID of the video
video_id = "CV0GtUlJ6NM"

# Fetch transcript
try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    # Flatten transcript into one large string
    transcript = " ".join(
        snippet.text
        for snippet in transcript_list
    )

except TranscriptsDisabled:
    print("No captions available for this video.")
    raise SystemExit(1)

# Create chunks of the transcript
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# IMPORTANT:
# create_documents() expects a list of strings, not a set.
chunks = splitter.create_documents([transcript])

print(f"Transcript length: {len(transcript)} characters")
print(f"Number of chunks: {len(chunks)}")

# Embedding model
embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Convert chunks into vectors and store them in FAISS
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# Retrieval
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

query = "What is attraction?"

result = retriever.invoke(query)

print("\n--- Retrieved Documents ---\n")

for i, doc in enumerate(result, start=1):
    print(f"### Result {i}")
    print(doc.page_content)
    print()

# Augmentation
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="openai/gpt-oss-20b",
)

prompt = PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is inufficient, just say you don't know.

        {context}
        Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "is the topic of attraction discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})

# Generation
answer = llm.invoke(final_prompt)
# print(answer.content)

# Building a Chain

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain_result = parallel_chain.invoke('Who is Dr. Paul Eastwick')
parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser
final_result = main_chain.invoke('Can you summarize this video')

print(final_result)