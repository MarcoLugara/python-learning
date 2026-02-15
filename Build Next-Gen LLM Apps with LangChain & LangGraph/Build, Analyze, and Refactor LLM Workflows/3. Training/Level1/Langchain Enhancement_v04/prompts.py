# prompts.py

from langchain_core.prompts import PromptTemplate
from schemas import intent_parser, intent_summary_parser, intent_summary_explain_parser

base_prompt = """
Analyze the following customer message.

Text:
{text}
"""

intent_prompt = PromptTemplate(
    template=base_prompt + """
Classify only the intent of the given text. Examples:

Text: "I'm interested in upgrading to a business package.", expected_intent: "sales".
Text: "My router stopped connecting after the update.",      expected_intent: "tech_support".
Text: "The green bond market has grown rapidly...",          expected_intent: "none".

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": intent_parser.get_format_instructions()}
)

intent_summary_prompt = PromptTemplate(
    template=base_prompt + """
Give a brief summary of the text and classify the intent of the given text.

Examples of intent classification:

Text: "I'm interested in upgrading to a business package."
Intent: sales

Text: "The green bond market has grown rapidly over the past decade..."
Intent: none  ← financial market commentary, not a customer request

Text: "How do I reset my router to factory settings?"
Intent: tech_support

Text: "My invoice shows a charge I don't recognize."
Intent: billing

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_summary_parser.get_format_instructions()
    }
)

intent_summary_explain_prompt = PromptTemplate(
    template=base_prompt + """
Give a brief summary of the text and classify the intent of the given text and give a brief explanation of why you chose your answer.

Examples of intent classification:

Text: "I'm interested in upgrading to a business package."
Intent: sales

Text: "The green bond market has grown rapidly over the past decade..."
Intent: none  ← financial market commentary, not a customer request

Text: "How do I reset my router to factory settings?"
Intent: tech_support

Text: "My invoice shows a charge I don't recognize."
Intent: billing

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_summary_explain_parser.get_format_instructions()
    }
)