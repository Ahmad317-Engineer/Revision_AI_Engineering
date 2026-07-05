from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

response = client.chat.completions.create(
    model="liquid/lfm-2.5-1.2b-instruct:free",
    messages=[
        {"role": "user", "content": "What does an AI engineer do? Answer in 2 sentences."}
    ]
)

print(response.choices[0].message.content)
