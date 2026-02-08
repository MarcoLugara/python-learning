"""
MODULE 1 - VIDEO 2: Building Your First Chain
Complete working demonstration code - Ollama Edition
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# =============================================================================
# STEP 1: Define Output Structure
# =============================================================================
print("=" * 60)
print("STEP 1: Defining Output Structure")
print("=" * 60)

class TextAnalysis(BaseModel):
    summary: str = Field(description="A concise summary of the text")
    key_points: List[str] = Field(description="List of 3-5 key points from the text")
    word_count: int = Field(description="Approximate word count of original text")

output_parser = JsonOutputParser(pydantic_object=TextAnalysis)

print("✓ Output parser created with schema:")
print("  - summary")
print("  - key_points")
print("  - word_count")
print()

# =============================================================================
# STEP 2: Create Prompt Template
# =============================================================================
print("=" * 60)
print("STEP 2: Creating Prompt Template")
print("=" * 60)

format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="""Analyze and summarize the following text.
Return the output in this exact JSON format:
{format_instructions}

Text: {text}""",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)

print("✓ Prompt template created with placeholders:")
print("  - {text} for input")
print("  - {format_instructions} for output format")
print()

# =============================================================================
# STEP 3: Initialize Model
# =============================================================================
print("=" * 60)
print("STEP 3: Initializing Ollama Model")
print("=" * 60)

# Initialize the model - CHANGED to use OllamaLLM
model = OllamaLLM(model="phi3:mini", temperature=0.3)

print("✓ Model initialized:")
print("  - Model: phi3:mini (lightweight, 3.8B parameters)")
print("  - Temperature: 0.3 (focused, consistent output)")
print("  - Make sure Ollama is running (run 'ollama serve' in terminal)")
print()

# =============================================================================
# STEP 4: Create the Chain
# =============================================================================
print("=" * 60)
print("STEP 4: Creating the Chain")
print("=" * 60)

chain = prompt_template | model | output_parser

print("✓ Chain created!")
print("  Components connected: Prompt → Ollama Model → Parser")
print()

# =============================================================================
# STEP 5: Prepare Sample Text
# =============================================================================
print("=" * 60)
print("STEP 5: Sample Text")
print("=" * 60)

sample_text = """
LangChain is a framework for developing applications powered by language models. 
It enables applications that are context-aware and can reason about complex tasks. 
The framework provides modular components that can be composed together, 
making it easier to build, test, and iterate on LLM applications.
"""

print("Sample text prepared:")
print(sample_text.strip())
print()

# =============================================================================
# STEP 6: Run the Chain
# =============================================================================
print("=" * 60)
print("STEP 6: Running the Chain")
print("=" * 60)
print("Executing chain.invoke()...")
print()

result = chain.invoke({"text": sample_text})

print("=" * 60)
print("RESULT - Structured Output:")
print("=" * 60)
import json
print(json.dumps(result, indent=2))
print()

# =============================================================================
# STEP 7: Show How to Access Structured Data
# =============================================================================
print("=" * 60)
print("BONUS: Accessing Structured Data")
print("=" * 60)

print("✓ You can now access structured fields:")
print(f"  - Summary: {result['summary'][:50]}...")
print(f"  - Key Points: {len(result['key_points'])} points")
print(f"  - Word Count: {result['word_count']}")
print()

print("=" * 60)
print("DEMO COMPLETE!")
print("=" * 60)