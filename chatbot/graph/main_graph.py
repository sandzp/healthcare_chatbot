import json
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any, List
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from openai import RateLimitError, APITimeoutError, APIConnectionError
from chatbot.graph.state import GraphState
from chatbot.graph.nodes.router import classify, cohort
from chatbot.tools.definitions import HealthcareTools
from chatbot.tools.utils import retry_with_backoff
from chatbot.domain.models import ExpandedQuery
from chatbot.prompts.system_prompts import IDENTIFY_COHORT, GENERAL_ANSWER, QUERY_EXPANDER, format_cohort_context

TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)

class ChatbotGraph:
    """LangGraph-based healthcare analytics chatbot with cohort management."""

    def __init__(
        self,
        api_key: str,
        chat_model: str,
        streaming: bool,
        no_retries: int,
        temperature: Optional[float]
    ) -> None:
        """Initialize the chatbot with LLM client, tools, and retry configuration."""
        self.healthcare_tools = HealthcareTools()
        llm_kwargs = {"model": chat_model, "streaming": streaming, "api_key": api_key}
        if temperature is not None:
            llm_kwargs["temperature"] = temperature
        self.llm = ChatOpenAI(**llm_kwargs)
        self.llm_client = self.llm.bind_tools(self.healthcare_tools.all)
        self.tool_node = ToolNode(self.healthcare_tools.all, handle_tool_errors=True)
        self.no_retries = no_retries

    def compile_graph(self, checkpointer=None):
        """Build and return the compiled LangGraph state machine."""
        ## NODES

        ### ROUTER NODES
        @retry_with_backoff(max_retries=self.no_retries, exceptions=TRANSIENT_ERRORS)
        async def is_relevant(state: GraphState) -> Dict[str, bool]:
            """
            Classifies whether the question is healthcare-related.
            """
            history = [msg for msg in state.get("messages", []) if isinstance(msg, HumanMessage)]
            result = await classify(self.llm, state["input"], history)
            return {
                "is_relevant": result.is_relevant,
            }

        @retry_with_backoff(max_retries=self.no_retries, exceptions=TRANSIENT_ERRORS)
        async def is_cohort(state: GraphState) -> Dict[str, bool]:
            """
            Classifies whether the question is asking to build or modify a cohort.
            """
            history = [msg for msg in state.get("messages", []) if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)][-6:]
            result = await cohort(self.llm, state["input"], history, cohorts=state.get("cohorts", {}))
            return {
                "is_cohort": result.is_cohort,
            }

        ### MAIN NODES
        @retry_with_backoff(max_retries=self.no_retries, exceptions=TRANSIENT_ERRORS)
        async def query_expander(state: GraphState) -> Dict[str, Any]:
            history = [msg for msg in state.get("messages", []) if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)][-6:]
            
            response = await self.llm.with_structured_output(ExpandedQuery).ainvoke(
                [
                    SystemMessage(content=QUERY_EXPANDER),
                    *history,
                    HumanMessage(content=state["input"])
                ]
            )
            print(response.query)
            return {"input": response.query}

        @retry_with_backoff(max_retries=self.no_retries, exceptions=TRANSIENT_ERRORS)
        async def exit_graph(state: GraphState) -> Dict[str, Any]:
            """
            Generates a polite rejection response for off-topic questions.
            """
            answer = ""
            async for chunk in self.llm.astream([
                SystemMessage(content="You are a healthcare analytics chatbot. The user asked something outside your scope. Politely explain you can only help with healthcare analytics and database queries."),
                HumanMessage(content=state["input"]),
            ]):
                answer += chunk.content
            return {"answer": answer}
        
        @retry_with_backoff(max_retries=self.no_retries, exceptions=TRANSIENT_ERRORS)
        async def answer_with_tools(state: GraphState) -> Dict[str, List]:
            """
            Calls the LLM with tools bound. May return a direct answer
            or request tool calls. Injects cohort context when cohorts exist.
            """
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None
            if not isinstance(last_msg, ToolMessage):
                prompt = GENERAL_ANSWER
                cohorts = state.get("cohorts", {})
                if cohorts:
                    prompt += format_cohort_context(cohorts)
                human_msg = HumanMessage(content=state["input"])
                messages = [SystemMessage(content=prompt), *messages, human_msg]
                response = await self.llm_client.ainvoke(messages)
                return {"messages": [human_msg, response]}

            response = await self.llm_client.ainvoke(messages)
            return {"messages": [response]}

        @retry_with_backoff(max_retries=self.no_retries, exceptions=TRANSIENT_ERRORS)
        async def identify_cohort(state: GraphState) -> Dict[str, Any]:
            """
            Calls the LLM with tools to identify patient IDs for a cohort.
            On completion, creates a temp table and updates state["cohorts"].
            """
            messages = state.get("messages", [])
            last_msg = messages[-1] if messages else None
            if not isinstance(last_msg, ToolMessage):
                prompt = IDENTIFY_COHORT
                cohorts = state.get("cohorts", {})
                if cohorts:
                    cohort_list = "\n".join(f"- {name}: {desc}" for name, desc in cohorts.items())
                    prompt += f"\n\nExisting cohorts:\n{cohort_list}"
                history = [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)][-6:]
                messages = [SystemMessage(content=prompt), *history, HumanMessage(content=state["input"])]

            response = await self.llm_client.ainvoke(messages)

            if not response.tool_calls:
                cohorts = dict(state.get("cohorts", {}))
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, ToolMessage):
                        try:
                            result = json.loads(msg.content)
                            sql = result.get("sql", "")
                            count = result.get("count", 0)
                            target = result.get("target_cohort", "")
                            if sql and count > 0:
                                table_name = target if target else f"cohort_{len(cohorts) + 1}"
                                self.healthcare_tools.db.conn.execute(
                                    f"CREATE OR REPLACE TEMP TABLE {table_name} AS ({sql})"
                                )
                                cohorts[table_name] = state["input"]
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
                return {"cohorts": cohorts, "messages": [*messages, response]}
            return {"messages": [*messages, response]}

        async def generate_answer(state: GraphState) -> Dict[str, str]:
            """
            Extracts the final text answer from the last LLM message.
            """
            last_message = state["messages"][-1]
            return {"answer": last_message.content}

        ## EDGES
        def route_relevant(state) -> str:
            if state.get("is_relevant") == False:
                return "irrelevant"
            return "relevant"

        def route_tools(state: GraphState) -> str:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return "done"

        def route_cohort(state) -> str:
            if state.get("is_cohort"):
                return "build_cohort"
            return "answer"


        workflow = StateGraph(GraphState)

        ## ADD NODES
        workflow.add_node("is_relevant", is_relevant)
        workflow.add_node("is_cohort", is_cohort)
        workflow.add_node("query_expander", query_expander)
        workflow.add_node("exit_graph", exit_graph)
        workflow.add_node("answer_with_tools", answer_with_tools)
        workflow.add_node("identify_cohort", identify_cohort)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("identify_tools", self.tool_node)
        workflow.add_node("generate_answer", generate_answer)

        ## ADD EDGES
        workflow.add_edge(START, "is_relevant")
        workflow.add_conditional_edges(
            "is_relevant",
            route_relevant,
            {
                "irrelevant": "exit_graph",
                "relevant": "query_expander",
            }
        )
        workflow.add_conditional_edges(
            "is_cohort",
            route_cohort,
            {
                "build_cohort": "identify_cohort",
                "answer": "answer_with_tools",
            }
        )
        workflow.add_conditional_edges(
            "answer_with_tools",
            route_tools,
            {
                "tools": "tools",
                "done": "generate_answer",
            }
        )
        workflow.add_conditional_edges(
            "identify_cohort",
            lambda state: "tools" if hasattr(state["messages"][-1], "tool_calls") and state["messages"][-1].tool_calls else "done",
            {
                "tools": "identify_tools",
                "done": END,
            }
        )
        workflow.add_edge("query_expander", "is_cohort")
        workflow.add_edge("identify_tools", "identify_cohort")
        workflow.add_edge("tools", "answer_with_tools")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("exit_graph", END)

        ## INIT GRAPH
        if checkpointer is None:
            checkpointer = MemorySaver()

        app = workflow.compile(checkpointer=checkpointer)
        return app
