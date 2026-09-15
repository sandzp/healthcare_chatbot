import uuid
import chainlit as cl
from typing import Optional
from chatbot.config.configuration import settings
from chatbot.graph.main_graph import ChatbotGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

ANSWER_NODES = {"exit_graph", "answer_with_tools", "identify_cohort"}

_checkpointer = None
_checkpointer_cm = None

async def get_checkpointer():
    """Return a shared AsyncPostgresSaver instance, initializing on first call."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        _checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)
        _checkpointer = await _checkpointer_cm.__aenter__()
        await _checkpointer.setup()
    return _checkpointer

def new_graph(checkpointer):
    graph = ChatbotGraph(
        api_key=settings.OPENAI_API_KEY,
        chat_model=settings.CHAT_MODEL,
        temperature=settings.TEMPERATURE,
        streaming=settings.STREAMING,
        no_retries=settings.RETRY_ATTEMPTS,
    )
    return graph.compile_graph(checkpointer=checkpointer)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def start():
    """Each new chat = fresh ChatbotGraph instance with persistent checkpointer."""
    cp = await get_checkpointer()
    cl.user_session.set("app", new_graph(cp))
    cl.user_session.set("thread_id", cl.context.session.thread_id)
    await cl.Message(content="Healthcare Analytics Chatbot ready. Ask me anything about the data.").send()

@cl.on_chat_resume
async def on_resume(thread):
    """Resume a previous thread — restores LangGraph state from PostgreSQL."""
    cp = await get_checkpointer()
    cl.user_session.set("app", new_graph(cp))
    cl.user_session.set("thread_id", thread["id"])

@cl.on_message
async def on_message(message: cl.Message):
    """Stream the graph response, showing tool calls as steps."""
    app = cl.user_session.get("app")
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.GRAPH_RECURSION_LIMIT}

    current_node = ""
    node_step = None
    msg = cl.Message(content="")
    await msg.send()

    async for event in app.astream_events(
        {"input": message.content}, config=config, version="v2"
    ):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and event.get("metadata", {}).get("langgraph_node"):
            new_node = event["metadata"]["langgraph_node"]
            if new_node != current_node:
                # close previous node step
                if node_step:
                    await node_step.__aexit__(None, None, None)
                    node_step = None
                current_node = new_node
                # open a new node step (skip for nodes that stream the final answer)
                if current_node not in ANSWER_NODES:
                    node_step = cl.Step(name=current_node, type="run")
                    await node_step.__aenter__()

        elif kind == "on_chain_end" and current_node == "query_expander" and node_step:
            output = event["data"].get("output", {})
            if isinstance(output, dict) and "input" in output:
                node_step.output = output["input"]

        elif kind == "on_tool_start":
            tool_input = event["data"].get("input", {})
            step = cl.Step(name=name, type="tool")
            step.input = str(tool_input)
            cl.user_session.set("current_step", step)
            await step.__aenter__()

        elif kind == "on_tool_end":
            step = cl.user_session.get("current_step")
            if step:
                output = event["data"].get("output", "")
                step.output = output.content if hasattr(output, "content") else str(output)
                await step.__aexit__(None, None, None)

        elif kind == "on_chat_model_stream" and current_node in ANSWER_NODES:
            chunk = event["data"]["chunk"]
            if chunk.content:
                await msg.stream_token(chunk.content)

    # close any remaining node step
    if node_step:
        await node_step.__aexit__(None, None, None)

    await msg.update()
