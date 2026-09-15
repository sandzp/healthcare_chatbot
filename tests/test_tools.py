import json
import pytest
from chatbot.tools.definitions import HealthcareTools


@pytest.fixture
def tools():
    """Shared HealthcareTools instance with cleanup."""
    t = HealthcareTools()
    yield t
    t.close()

def test_list_tables(tools):
    result = json.loads(tools.general[0].invoke({}))
    assert "demographics" in result["tables"]
    assert "rx_events" in result["tables"]

def test_query_database(tools):
    result = json.loads(tools.general[1].invoke({"sql": "SELECT COUNT(*) AS cnt FROM demographics"}))
    assert result["columns"] == ["cnt"]
    assert result["total_rows"] == 1
    assert result["rows"][0][0] > 0

def test_describe_table(tools):
    result = json.loads(tools.general[2].invoke({"table_name": "demographics"}))
    assert result["table"] == "demographics"
    columns = [col["column"] for col in result["schema"]]
    assert "PATIENT_NUMBER" in columns

def test_table_info(tools):
    result = json.loads(tools.general[3].invoke({"table_name": "demographics"}))
    assert result["row_count"] > 0
    assert len(result["sample_data"]["rows"]) <= 5

def test_identify_cohort_returns_count_and_sql(tools):
    sql = "SELECT PATIENT_NUMBER FROM demographics LIMIT 5"
    result = json.loads(tools.cohort[0].invoke({"sql": sql}))
    assert result["count"] == 5
    assert result["sql"] == sql
    assert result["target_cohort"] == ""

def test_identify_cohort_with_target(tools):
    sql = "SELECT PATIENT_NUMBER FROM demographics LIMIT 5"
    result = json.loads(tools.cohort[0].invoke({"sql": sql, "target_cohort": "cohort_1"}))
    assert result["target_cohort"] == "cohort_1"

def test_temp_table_persistence(tools):
    sql = "SELECT PATIENT_NUMBER FROM demographics LIMIT 10"
    tools.db.conn.execute(f"CREATE TEMP TABLE test_cohort AS ({sql})")
    count = tools.db.query("SELECT COUNT(*) FROM test_cohort")[0][0]
    assert count == 10

def test_compare_cohorts_returns_stats(tools):
    tools.db.conn.execute("CREATE TEMP TABLE cohort_a AS SELECT PATIENT_NUMBER FROM demographics LIMIT 50")
    tools.db.conn.execute("CREATE TEMP TABLE cohort_b AS SELECT PATIENT_NUMBER FROM demographics LIMIT 50 OFFSET 50")
    result = json.loads(tools.cohort[1].invoke({
        "cohort_a_table": "cohort_a",
        "cohort_b_table": "cohort_b",
        "cohort_a_name": "Group A",
        "cohort_b_name": "Group B",
    }))
    assert "statistical_tests" in result
    assert result["cohort_a"]["size"] == 50
    assert result["cohort_b"]["size"] == 50
    assert result["cohort_a"]["age_mean"] is not None
    assert len(result["cohort_a"]["top_diagnoses"]) > 0


def test_compare_cohorts_empty_cohort(tools):
    tools.db.conn.execute("CREATE TEMP TABLE cohort_empty AS SELECT PATIENT_NUMBER FROM demographics WHERE 1=0")
    tools.db.conn.execute("CREATE TEMP TABLE cohort_full AS SELECT PATIENT_NUMBER FROM demographics LIMIT 10")
    result = json.loads(tools.cohort[1].invoke({
        "cohort_a_table": "cohort_empty",
        "cohort_b_table": "cohort_full",
        "cohort_a_name": "Empty",
        "cohort_b_name": "Full",
    }))
    assert "error" in result
