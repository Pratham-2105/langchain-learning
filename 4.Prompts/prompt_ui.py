import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    model="openai/gpt-oss-20b",
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Researh Paper Name",
    [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Difussion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    " Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    " Select Explantion Length",
    [
        "Short (1 - 2 paragraphs)",
        "Medium(3 - 5 paragraphs)",
        "Long (detailed explanation)",
    ],
)


template = load_prompt("4.Prompts/template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke(
        prompt=template.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )
    )
    
    st.write(result.content)
