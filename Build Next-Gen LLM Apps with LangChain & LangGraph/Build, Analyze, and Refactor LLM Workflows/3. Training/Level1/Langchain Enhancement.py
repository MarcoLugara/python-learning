from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
from typing import List

# =============================================================================
# STEP 1: Define Output Structure
# =============================================================================
print("=" * 60)
print("STEP 1: Defining Output Structure")
print("=" * 60)

class Fetch_intent(BaseModel):
    area_of_interest: str = Field(description="Area of interest between billing, tech support or sales. Only one of those three words" )
    justification_of_choice: str = Field(description="Justify your choice.")

output_parser = JsonOutputParser(pydantic_object=Fetch_intent)

print("✓ Output parser created with schema:")
print("  - area of interest")
print("  - justification of choice")
print()

# =============================================================================
# STEP 2: Creating Prompt Template
# =============================================================================
print("=" * 60)
print("STEP 2: Creating Prompt Template")
print("=" * 60)

format_instructions = output_parser.get_format_instructions()

prompt_template = PromptTemplate(
    template="""Analyze and summarize the following text and classify the area of interest: whether its about billing, tech support or sales. 
    Justify your choice formally.
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
# STEP 5: Prepare Sample Text (Test prompt)
# =============================================================================
print("=" * 60)
print("STEP 5: Sample Text")
print("=" * 60)

sample_text = """
TEST
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
print(json.dumps(result, indent=2))
print()