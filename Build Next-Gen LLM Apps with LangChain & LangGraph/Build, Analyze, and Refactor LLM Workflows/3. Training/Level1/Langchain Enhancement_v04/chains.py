# chains.py

from langchain_ollama import OllamaLLM
from config import MODEL_NAME, TEMPERATURE, SEED
from schemas import intent_parser, intent_summary_parser, intent_summary_explain_parser
from prompts import intent_prompt, intent_summary_prompt, intent_summary_explain_prompt

model = OllamaLLM(model=MODEL_NAME, temperature=TEMPERATURE, seed=SEED)

intent_only_chain = intent_prompt | model | intent_parser
intent_only_chain.name = "intent"
intent_summary_chain = intent_summary_prompt | model | intent_summary_parser
intent_summary_chain.name = "intent_summary"
intent_summary_explain_chain = intent_summary_explain_prompt | model | intent_summary_explain_parser
intent_summary_explain_chain.name = "intent_summary_explain"

