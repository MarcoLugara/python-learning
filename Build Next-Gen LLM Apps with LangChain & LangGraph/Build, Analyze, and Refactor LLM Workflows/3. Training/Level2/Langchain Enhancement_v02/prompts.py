# prompts.py

from langchain_core.prompts import PromptTemplate
from schemas import intent_parser, intent_summary_parser, intent_summary_explain_parser

# Base prompt for intent classification
base_prompt = """
Analyze the following customer message.

Text:
{text}
"""

few_shot_examples = """
Examples of intent classification:

Text: "I'm interested in upgrading to a business package."
Intent: sales

Text: "The green bond market has grown rapidly over the past decade..."
Intent: none  ← financial market commentary, not a customer request

Text: "How do I reset my router to factory settings?"
Intent: tech_support

Text: "My invoice shows a charge I don't recognize."
Intent: billing
"""

# Intent classification prompts (unchanged from Level 3)
intent_prompt = PromptTemplate(
    template=base_prompt + """
Classify only the intent of the given text.
""" + few_shot_examples + """
{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": intent_parser.get_format_instructions()}
)

intent_summary_prompt = PromptTemplate(
    template=base_prompt + """
Give a brief summary of the text and classify the intent.
""" + few_shot_examples + """
{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_summary_parser.get_format_instructions()
    }
)

intent_summary_explain_prompt = PromptTemplate(
    template=base_prompt + """
Give a brief summary of the text, classify the intent, and explain your choice.
""" + few_shot_examples + """
{format_instructions}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": intent_summary_explain_parser.get_format_instructions()
    }
)

# NEW: RAG prompt for document-grounded Q&A
rag_prompt = PromptTemplate(
    template="""You are a helpful customer support assistant. Answer the user's question based ONLY on the context provided below. 

If the context doesn't contain enough information to answer the question completely, say "I don't have enough information in my knowledge base to answer that fully" and provide what information you do have.

Do not make up information. Do not use knowledge outside of the provided context.

Context from company documents:
{context}

Customer Question:
{question}

Your Answer:""",
    input_variables=["context", "question"]
)
