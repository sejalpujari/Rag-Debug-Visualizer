# llm.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads .env file

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query: str, context: str, model: str = "openai/gpt-oss-120b") -> str:
    """
    Simple RAG generation using Groq.
    You can change the model name to any supported model, e.g.:
    - llama-3.1-8b-instant
    - llama-3.3-70b-versatile
    - mixtral-8x7b-32768
    - gemma2-9b-it
    """
    system_prompt = """You are a helpful assistant that answers questions accurately 
using ONLY the provided context. If the context doesn't contain the answer, 
say "I don't have enough information to answer this." 
Do not make up information."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        model=model,
        temperature=0.3,          # low for more factual answers
        max_tokens=600,
        top_p=0.9,
    )

    return chat_completion.choices[0].message.content.strip()