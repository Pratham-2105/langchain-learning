import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

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

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short quiz-type questions and answer from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the following texts into a single document \n {notes} and {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | weak_model | parser,
    'quiz': prompt2 | strong_model | parser
})

merge_chain = prompt3 | strong_model | parser

final_chain = parallel_chain | merge_chain

sample_text = """
1. Tokenization
The input text is divided into smaller units called tokens.

For example:

Machine learning is useful __

may be split into: ["Machine", " learning", " is", " useful"]

2. Embeddings
Each token is converted into a numerical vector called an embedding, which represents its meaning and relationships with other tokens.

3. Transformer Architecture
The tokens pass through multiple Transformer blocks that process their relationships using self-attention and feed-forward networks.

A simplified flow is: Tokens → Embeddings → Transformer Blocks → Output

4. Self-Attention
Self-attention helps the model understand relationships between tokens by determining which tokens are important to each other.

For example:

It helps connect "it" with the relevant word in: The animal crossed the road because it was tired.

5. Next-Token Prediction
The LLM predicts the next token based on the previous tokens.

For example:

Artificial intelligence is → powerful

The process repeats one token at a time until the complete response is generated.

Popular LLMs
Examples of well-known LLM families include : GPT, Llama, Claude, Gemini, Mistral and Qwen

GPT-5.4 (OpenAI): Advanced LLM designed for reasoning, coding, tool use and a wide range of language tasks.
Gemini 3.1 Pro (Google DeepMind): Multimodal LLM capable of processing text, images, audio, video and documents.
Claude Sonnet 5 and Claude Opus 5 (Anthropic): LLMs designed for reasoning, coding, writing and other complex tasks.
Llama 4 (Meta): Open-weight multimodal LLM family that includes models such as Llama 4 Scout and Llama 4 Maverick.
DeepSeek-V4.1-Flash (DeepSeek): Multimodal LLM designed for reasoning, coding and efficient AI applications.
Mistral Large 3 (Mistral AI): Open-weight multimodal LLM designed for general-purpose language, reasoning and coding tasks.
Qwen3 (Alibaba): A family of LLMs designed for language, reasoning, coding and multilingual applications, with multimodal variants such as Qwen3-VL.
Applications
Code Generation: LLMs can generate accurate code based on user instructions for specific tasks.
Debugging and Documentation: They assist in identifying code errors, suggesting fixes and even automating project documentation.
Question Answering: Users can ask both casual and complex questions, receiving detailed, context-aware responses.
Language Translation and Correction: LLMs can translate across many languages (often dozens to 100+).
Prompt-Based Versatility: By crafting creative prompts, users can unlock endless possibilities, as LLMs excel in one-shot and zero-shot learning scenarios.
Advantages
Can perform new tasks using zero-shot and few-shot learning without retraining
Efficiently process and understand large amounts of text data
Adapt easily to specific domains through fine-tuning
Automate repetitive language-based tasks, reducing human effort
Work effectively across multiple domains like healthcare, education and business
Limitations
Require very high computational resources, making them expensive to train
Training can take a long time, often weeks or months
Depend on large amounts of high-quality and unbiased data
Consume significant energy, contributing to environmental impact
Can introduce bias and misinformation, raising ethical concerns

"""
result = final_chain.invoke({'text':sample_text})

print(result)

final_chain.get_graph().print_ascii()