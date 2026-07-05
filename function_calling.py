from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def calculate(expression: str) -> str:
    """Math expression calculate karo jaise '347 * 89'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def chat_with_tools(user_message):
    print(f"\nUser: {user_message}")

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=user_message,
        config=types.GenerateContentConfig(
            tools=[calculate]
        )
    )

    print(f"Jawab: {response.text}")

chat_with_tools("What is 347 multiplied by 89?")
chat_with_tools("What is the capital of France?")
chat_with_tools("1500 minus 347, then multiply by 3")