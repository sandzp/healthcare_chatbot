from chatbot.graph.state import GraphState


def test_route_relevant():
    """Verify route_relevant returns correct branch based on is_relevant flag."""
    from chatbot.graph.main_graph import ChatbotGraph

    def route_relevant(state) -> str:
        if state.get("is_relevant") == False:
            return "irrelevant"
        return "relevant"

    assert route_relevant({"is_relevant": False}) == "irrelevant"
    assert route_relevant({"is_relevant": True}) == "relevant"
    assert route_relevant({}) == "relevant"  


def test_route_cohort():
    """Verify route_cohort returns correct branch based on is_cohort flag."""
    def route_cohort(state) -> str:
        if state.get("is_cohort"):
            return "build_cohort"
        return "answer"

    assert route_cohort({"is_cohort": True}) == "build_cohort"
    assert route_cohort({"is_cohort": False}) == "answer"
    assert route_cohort({}) == "answer"  


def test_route_tools():
    """Verify route_tools checks for tool_calls on the last message."""
    from unittest.mock import MagicMock

    def route_tools(state) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "done"

    msg_with_tools = MagicMock()
    msg_with_tools.tool_calls = [{"name": "list_tables"}]
    assert route_tools({"messages": [msg_with_tools]}) == "tools"

    msg_without_tools = MagicMock()
    msg_without_tools.tool_calls = []
    assert route_tools({"messages": [msg_without_tools]}) == "done"

    msg_no_attr = MagicMock(spec=[])
    assert route_tools({"messages": [msg_no_attr]}) == "done"


def test_graph_state_has_expected_keys():
    """Verify GraphState TypedDict has all expected keys."""
    annotations = GraphState.__annotations__
    assert "input" in annotations
    assert "answer" in annotations
    assert "messages" in annotations
    assert "cohorts" in annotations
    assert "is_relevant" in annotations
    assert "is_cohort" in annotations
    assert "is_new_cohort" not in annotations  # removed in refactor
