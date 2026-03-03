# schemas.py

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class IntentOnly(BaseModel):
    intent: Literal["billing", "tech_support", "sales", "none"] = Field(
        description="Area of interest between billing, tech support, sales or none of them."
    )

class IntentSummary(BaseModel):
    summary_of_the_text: str = Field(
        description="Quick summary of the text. No more than 3 lines. Be concise and formal.")
    intent: Literal["billing", "tech_support", "sales", "none"] = Field(
        description="Area of interest between billing, tech support, sales or none of them."
    )

class IntentSummaryExplain(BaseModel):
    summary_of_the_text: str = Field(
        description="Quick summary of the text. No more than 3 lines. Be concise and formal.")
    intent: Literal["billing", "tech_support", "sales", "none"] = Field(
        description="Area of interest between billing, tech support, sales or none of them."
    )
    justification_of_choice: str = Field(description="Justify your intent choice.")


intent_parser                 = JsonOutputParser(pydantic_object=IntentOnly)
intent_summary_parser         = JsonOutputParser(pydantic_object=IntentSummary)
intent_summary_explain_parser = JsonOutputParser(pydantic_object=IntentSummaryExplain)
