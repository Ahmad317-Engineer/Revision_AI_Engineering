from openai import OpenAI
from dotenv import load_dotenv
import os
import requests

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# TOOL 1 — Web Search
def web_search(query: str) -> str:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.getenv("TAVILY_API_KEY"),
            "query": query,
            "max_results": 3
        }
    )
    results = response.json().get("results", [])
    if not results:
        return "Koi result nahi mila."
    output = ""
    for i, r in enumerate(results):
        output += f"[{i+1}] {r['title']}\n{r['content'][:300]}\n\n"
    return output

# TOOL 2 — File Writer
def write_file(filename: str, content: str) -> str:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File '{filename}' save ho gayi!"

# Tool definitions OpenAI format mein
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Save content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        }
    }
]

import json

# Memory — conversation history
conversation_history = []

def chat_with_memory(user_message: str):
    """Memory wala agent — har turn yaad rehta hai"""
    
    # User message history mein add karo
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    messages = [
        {"role": "system", "content": "You are a helpful research agent with memory. Use web_search and write_file tools when needed."}
    ] + conversation_history
    
    while True:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            # Agent ka jawab history mein save karo
            conversation_history.append({
                "role": "assistant",
                "content": msg.content
            })
            return msg.content
        
        for tool_call in msg.tool_calls:
            import json
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            print(f"  → Tool: {name}")
            
            if name == "web_search":
                result = web_search(args["query"])
            elif name == "write_file":
                result = write_file(args["filename"], args["content"])
            else:
                result = "Unknown tool"
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

# Test — memory ke saath
print("=== Memory Agent ===\n")
r1 = chat_with_memory("What is FastAPI?")
print(f"Turn 1: {r1}\n")

r2 = chat_with_memory("Give me more detail about what you just explained")
print(f"Turn 2: {r2}\n")

def run_agent(task: str):
    print(f"\nTask: {task}")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a research agent. Use web_search to find info, then write_file to save the report as report.txt."},
        {"role": "user", "content": task}
    ]

    # Agent loop — jab tak kaam complete na ho
    while True:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        messages.append(msg)

        # Koi tool call nahi — kaam khatam
        if not msg.tool_calls:
            print(f"\nAgent: {msg.content}")
            break

        # Tool calls handle karo
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            print(f"  → Tool: {name} | Args: {args}")

            if name == "web_search":
                result = web_search(args["query"])
            elif name == "write_file":
                result = write_file(args["filename"], args["content"])
            else:
                result = "Unknown tool"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

run_agent("Research: What is RAG in AI? Write a short report and save to report.txt")