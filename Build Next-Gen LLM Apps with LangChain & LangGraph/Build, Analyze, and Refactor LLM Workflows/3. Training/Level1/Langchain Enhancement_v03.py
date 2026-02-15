# =============================================================================
# Version Updates: This version implements progressive fallback
#   in the invoke part and time check for the model invoke
# =============================================================================

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
from typing import Literal
import time

# =============================================================================
# STEP 1: Define Output Structure
# =============================================================================
print("=" * 60)
print("STEP 1: Defining Output Structure")
print("We define three different output structures:"
      "- one that returns only the area of interest"
      "- one that returns a summary of the text and the area of interest"
      "- one to return a summary of the text, the area of interest and the reason why the latter has been chosen")
print("=" * 60)


class IntentOnly(BaseModel):
    intent: Literal["billing", "tech_support", "sales", "none"] = Field(
        description="Area of interest between billing, tech support, sales or none of them. Only one of those three words."
    )


intent_parser = JsonOutputParser(pydantic_object=IntentOnly)


class IntentSummary(BaseModel):
    summary_of_the_text: str = Field(
        description="Quick summary of the text. No more than 3 lines in a generic PDF extract. Be concise and formal.")
    intent: Literal["billing", "tech_support", "sales", "none"] = Field(
        description="Area of interest between billing, tech support, sales or none of them. Only one of those three words."
    )


intent_summary_parser = JsonOutputParser(pydantic_object=IntentSummary)


class IntentSummaryExplain(BaseModel):
    summary_of_the_text: str = Field(
        description="Quick summary of the text. No more than 3 lines in a generic PDF extract. Be concise and formal.")
    intent: Literal["billing", "tech_support", "sales", "none"] = Field(
        description="Area of interest between billing, tech support, sales or none of them. Only one of those three words."
    )
    justification_of_choice: str = Field(description="Justify your intent choice.")


intent_summary_explain_parser = JsonOutputParser(pydantic_object=IntentSummaryExplain)

print("✓ Three outputs parser created with these schema:")
print("  - area of interest")
print("  - summary of the text and area of interest")
print("  - summary of the text, area of interest and justification of choice")
print()

# =============================================================================
# STEP 2: Creating Prompt Template
# =============================================================================
print("=" * 60)
print("STEP 2: Creating Prompt Template")
print("We create three prompt templates, by starting with one \"base_prompt\" that simply reads the text. "
      "Then, we define three more advanced prompts that do the three things expected, namely summarizing, "
      "defining the area of interest if any and/or providing a justification")
print("=" * 60)

print("First, we create the base prompt, which reads and analyses internally the given text")
base_prompt = """
Analyze the following customer message.

Text:
{text}
"""

print("Then, we create the three specific prompts for our needs")

print("First, we do the area of interest-only prompt")
intent_prompt = PromptTemplate(
    template=base_prompt + """
Classify only the intent of the given text.

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_parser.get_format_instructions()
    }
)

print("Then, we build the summary plus area of interest prompt")
intent_summary_prompt = PromptTemplate(
    template=base_prompt + """
Give a brief summary of the text and classify the intent of the given text.

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_summary_parser.get_format_instructions()
    }
)

print("Last, we build the summary, intent and explanation prompt")
intent_summary_explain_prompt = PromptTemplate(
    template=base_prompt + """
Give a brief summary of the text and classify the intent of the given text and give a brief explanation of why you chose your answer.

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_summary_explain_parser.get_format_instructions()
    }
)

print("✓ Prompt templates created with placeholders:")
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
model = OllamaLLM(model="phi3:mini", temperature=0)

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
print(" We create three different chains, each for our desired analysis.")
print("=" * 60)

intent_only_chain = intent_prompt | model | intent_parser
intent_summary_chain = intent_summary_prompt | model | intent_summary_parser
intent_summary_explain_chain = intent_summary_explain_prompt | model | intent_summary_explain_parser

print("✓ Chains created!")
print("  Components connected: Prompts → Ollama Model → Parsers")
print()

# =============================================================================
# STEP 5: Prepare Sample Text (Test prompt)
# =============================================================================
print("=" * 60)
print("STEP 5: Sample Text")
print("=" * 60)

sample_text = """
The **green bond market** has grown rapidly over the past decade, reflecting a broader shift toward sustainable finance. Green bonds are debt instruments specifically issued to fund projects with environmental benefits — such as renewable energy installations, climate-resilient infrastructure, or low-carbon transportation. Governments, corporations, and financial institutions all participate, often using independent verification to reassure investors that the funds are genuinely directed toward sustainability goals.

For investors, green bonds offer a way to combine financial returns with environmental impact. Yields are typically comparable to conventional bonds from similar issuers, but demand is often strong because institutional investors — including pension funds and asset managers — increasingly have sustainability mandates. This demand can sometimes allow issuers to secure slightly better financing terms.

Despite the optimism, the sector still faces challenges. Standardization of reporting, prevention of “greenwashing,” and consistent impact measurement remain ongoing concerns. Even so, most analysts expect the green bond segment to remain a central pillar of the evolving sustainable finance landscape.

"""

print("Sample text prepared:")
print(sample_text.strip())
print()

# =============================================================================
# STEP 6: We run the Chain
# =============================================================================
print("=" * 60)
print("STEP 6: Running the Chain")
print("=" * 60)
print("Executing chain.invoke()...")
print()

print("We try subsequentially the three approaches")
print("We start by the sole are of interest, into summary and area of interest, "
      "into summary area of interest and explanation of the choice")

# try-except allows to debug and study hallucinations or errors


# =============================================================================
# DEFINITION OF FALLBACK EXECUTOR
# =============================================================================
def progressive_intent_analysis(text):
    chains = [
        ("intent_summary_explanation", intent_summary_explain_chain),
        ("intent_summary", intent_summary_chain),
        ("intent_only", intent_only_chain),
    ]

    for label, chain in chains:
        try:
            result = chain.invoke({"text": text})

            print(f"✔ Success using {label} chain")
            return result

        except Exception as e:
            print(f"✖ {label} chain failed:", e)

    return {"error": "All fallback chains failed"}


# =============================================================================
# EXECUTION WITH FALLBACK EXECUTOR AND TIME COUNTING
# =============================================================================

#We let the timer begin to analyse the response time
start = time.perf_counter()

#We then run the chain
result = progressive_intent_analysis(sample_text)
print(json.dumps(result, indent=2))

#Finally, we check the time it took
elapsed = time.perf_counter() - start
print(f" \"progressive_intent_analysis\" chain took {elapsed:.2f}s")