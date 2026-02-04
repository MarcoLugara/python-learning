# level1_bot.py
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

openai.api_key = "HARDCODED_KEY"

def classify_intent(user_input):
    prompt = f"""
Classify the intent of the user message as:
billing, tech_support, or sales

User message:
{user_input}

Respond in this format:
Intent: <intent>
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response["choices"][0]["message"]["content"]
    match = re.search(r"Intent:\s*(\w+)", text)

    if not match:
        return "unknown"

    return match.group(1)

if __name__ == "__main__":
    while True:
        msg = input("> ")
        print("Intent:", classify_intent(msg))
