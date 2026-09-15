import pytest
from unittest.mock import MagicMock, AsyncMock
from chatbot.graph.nodes.router import classify, cohort
from chatbot.domain.models import RouteClassification, IsCohort


@pytest.mark.asyncio
async def test_classify_calls_llm_with_structured_output():
    """Verify classify() invokes the LLM with RouteClassification schema."""
    mock_llm = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value=RouteClassification(is_relevant=True))
    mock_llm.with_structured_output.return_value = mock_classifier

    result = await classify(mock_llm, "How many patients have diabetes?", [])

    assert result.is_relevant is True
    mock_llm.with_structured_output.assert_called_once_with(RouteClassification)


@pytest.mark.asyncio
async def test_classify_irrelevant():
    """Verify classify() returns is_relevant=False for off-topic questions."""
    mock_llm = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value=RouteClassification(is_relevant=False))
    mock_llm.with_structured_output.return_value = mock_classifier

    result = await classify(mock_llm, "What's the weather?", [])

    assert result.is_relevant is False


@pytest.mark.asyncio
async def test_cohort_without_existing_cohorts():
    """Verify cohort() works without existing cohorts."""
    mock_llm = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value=IsCohort(is_cohort=True))
    mock_llm.with_structured_output.return_value = mock_classifier

    result = await cohort(mock_llm, "Build a cohort of patients with diabetes", [])

    assert result.is_cohort is True


@pytest.mark.asyncio
async def test_cohort_with_existing_cohorts():
    """Verify cohort() appends existing cohort info to the prompt."""
    mock_llm = MagicMock()
    mock_classifier = MagicMock()
    mock_classifier.ainvoke = AsyncMock(return_value=IsCohort(is_cohort=False))
    mock_llm.with_structured_output.return_value = mock_classifier

    existing = {"cohort_1": "FL patients with heart disease"}
    result = await cohort(mock_llm, "What medications are they on?", [], cohorts=existing)

    assert result.is_cohort is False
    call_args = mock_classifier.ainvoke.call_args[0][0]
    system_msg = call_args[0].content
    assert "cohort_1" in system_msg
    assert "FL patients with heart disease" in system_msg


@pytest.mark.asyncio
async def test_cohort_returns_only_is_cohort():
    """Verify IsCohort model only has is_cohort field (no is_new_cohort)."""
    model = IsCohort(is_cohort=True)
    assert hasattr(model, "is_cohort")
    assert not hasattr(model, "is_new_cohort")
