from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

with open("notes.txt", "r") as f:
    document = f.read()

question = input("Ask a question about the document: ")

SYSTEM_PROMPT = """You are a precise document analyst. Answer questions using ONLY the document provided.

Always respond in this exact format:

QUOTE: [copy the most relevant sentence(s) from the document verbatim]
ANSWER: [your direct answer based on that quote]
CONFIDENCE: [High / Medium / Low — based on how clearly the document addresses this]

If the document does not contain enough information to answer, respond:
QUOTE: N/A
ANSWER: The document does not contain information about this.
CONFIDENCE: Low

Never make up information. Never use outside knowledge."""

# Few-shot example: shows the model exactly what a good response looks like
FEW_SHOT_EXAMPLE = {
    "role": "assistant",
    "content": """QUOTE: "The project deadline is March 15th and all teams must submit final reports by noon."
ANSWER: The deadline is March 15th, with reports due by noon.
CONFIDENCE: High"""
}

response = client.chat.completions.create(
    model="liquid/lfm-2.5-1.2b-instruct:free",
    messages=[
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nDOCUMENT:\n{document}"},
        {"role": "user", "content": "What is the deadline for the project?"},
        FEW_SHOT_EXAMPLE,
        {"role": "user", "content": question}
    ]
)

answer = response.choices[0].message.content
print("\n" + answer)