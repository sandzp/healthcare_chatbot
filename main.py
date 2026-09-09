import asyncio
import uuid
from chatbot.config.configuration import settings
from chatbot.graph.main_graph import ChatbotGraph


async def main():
    """CLI entrypoint for the healthcare analytics chatbot."""
    graph = ChatbotGraph(
        api_key=settings.OPENAI_API_KEY,
        chat_model=settings.CHAT_MODEL,
        temperature=settings.TEMPERATURE,
        streaming=settings.STREAMING,
        no_retries=settings.RETRY_ATTEMPTS
    )
    app = graph.compile_graph()

    print("Healthcare Analytics Chatbot")
    print("Type 'exit' or 'quit' to end the conversation")
    print("-" * 60)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.GRAPH_RECURSION_LIMIT}

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if "exit" in user_input.lower().strip():
                print("Goodbye!")
                break

            if not user_input:
                continue

            ANSWER_NODES = {"exit_graph", "answer_with_tools", "identify_cohort"}
            current_node = ""
            streaming_answer = False

            async for event in app.astream_events(
                {"input": user_input},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")

                if kind == "on_chain_start" and event.get("metadata", {}).get("langgraph_node"):
                    node = event["metadata"]["langgraph_node"]
                    if node != current_node:
                        current_node = node
                        if streaming_answer:
                            streaming_answer = False
                        print(f"\n  [{current_node}]", flush=True)

                elif kind == "on_tool_start":
                    tool_input = event["data"].get("input", {})
                    args = ', '.join(f'{k}={v}' for k, v in tool_input.items()) if isinstance(tool_input, dict) and tool_input else ''
                    print(f"    -> {name}({args})", flush=True)

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    content = output.content if hasattr(output, "content") else str(output)
                    print(f"    <- {content[:500]}", flush=True)

                elif kind == "on_chat_model_stream" and current_node in ANSWER_NODES:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        if not streaming_answer:
                            print("\n  Assistant: ", end="", flush=True)
                            streaming_answer = True
                        print(chunk.content, end="", flush=True)

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break



if __name__ == "__main__":
    asyncio.run(main())
