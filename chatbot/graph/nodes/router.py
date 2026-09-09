from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from chatbot.domain.models import RouteClassification, IsCohort
from chatbot.prompts.router_prompts import CLASSIFICATION_PROMPT, COHORT_PROMPT

async def classify(llm: ChatOpenAI, question: str, history: List[HumanMessage]) -> RouteClassification:
    """Classify whether a question is healthcare-related."""
    messages = [SystemMessage(content=CLASSIFICATION_PROMPT), *history, HumanMessage(content=question)]
    classifier = llm.with_structured_output(RouteClassification)
    return await classifier.ainvoke(messages)

async def cohort(llm: ChatOpenAI, question: str, history: List[BaseMessage], cohorts: Dict[str, str] = None) -> IsCohort:
    """Classify whether a question is asking to build or modify a cohort."""
    prompt = COHORT_PROMPT
    if cohorts:
        cohort_list = "\n".join(f"- {name}: {desc}" for name, desc in cohorts.items())
        prompt += f"\n\nExisting cohorts:\n{cohort_list}"
    messages = [SystemMessage(content=prompt), *history, HumanMessage(content=question)]
    classifier = llm.with_structured_output(IsCohort)
    return await classifier.ainvoke(messages)
