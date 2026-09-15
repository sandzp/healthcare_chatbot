from pydantic import BaseModel, Field


class RouteClassification(BaseModel):
    """LLM-enforced output for the routing classifier."""
    is_relevant: bool = Field(
        description="True if the question is related to healthcare analytics or database queries, False otherwise."
    )

class IsCohort(BaseModel):
    """LLM-enforced output for deciding if a question pertains to building a cohort of patients"""
    is_cohort: bool = Field(
        description="True if the user wants to build, create, update, narrow, or expand a patient cohort. False for everything else."
    )

class ExpandedQuery(BaseModel):
    """The LLM-expanded query or the original query based upon the system prompt"""
    query: str = Field(
        description="The expanded query or the users original query depending on whether the LLM decides the query is too vague"
    )
