"""
LangGraph-native tool definitions for the healthcare database agent.

Tools are defined using the @tool decorator from langchain_core.
LangGraph's ToolNode will automatically discover and execute these.
"""

import json
from langchain_core.tools import tool
from chatbot.tools.db_query import DuckDBQuery
from chatbot.tools.utils import _CustomEncoder


class HealthcareTools:
    """Container for healthcare database tools, grouped by purpose."""

    def __init__(self):
        self.db = DuckDBQuery(read_only=False)
        self.general = self._create_general_tools()
        self.cohort = self._create_cohort_tools()
        self.all = self.general + self.cohort

    def close(self):
        """Close the shared database connection."""
        self.db.close()

    def _create_general_tools(self) -> list:
        db = self.db

        @tool
        def list_tables() -> str:
            """List all available tables in the healthcare database."""
            tables = db.list_tables()
            return json.dumps({"tables": tables}, cls=_CustomEncoder)

        @tool
        def query_database(sql: str) -> str:
            """Execute a SQL query on the healthcare database and return results.

            Args:
                sql: The SQL query to execute (e.g., 'SELECT * FROM demographics LIMIT 10')
            """
            columns, rows = db.query_with_columns(sql)
            limited_rows = rows[:100]
            result = {
                "columns": columns,
                "rows": [list(row) for row in limited_rows],
                "total_rows": len(rows),
                "truncated": len(rows) > 100,
            }
            return json.dumps(result, cls=_CustomEncoder)

        @tool
        def describe_table(table_name: str) -> str:
            """Get the schema/structure of a specific table including column names and types.

            Args:
                table_name: The name of the table to describe.
            """
            columns, rows = db.query_with_columns(f"DESCRIBE {table_name}")
            result = {
                "table": table_name,
                "schema": [{"column": row[0], "type": row[1]} for row in rows],
            }
            return json.dumps(result, cls=_CustomEncoder)

        @tool
        def table_info(table_name: str) -> str:
            """Get comprehensive information about a table including row count, schema, and sample data.

            Args:
                table_name: The name of the table to get info about.
            """
            count = db.query(f"SELECT COUNT(*) FROM {table_name}")[0][0]
            _, schema_rows = db.query_with_columns(f"DESCRIBE {table_name}")
            _, sample_rows = db.query_with_columns(f"SELECT * FROM {table_name} LIMIT 5")
            result = {
                "table": table_name,
                "row_count": count,
                "schema": [{"column": row[0], "type": row[1]} for row in schema_rows],
                "sample_data": {
                    "columns": [row[0] for row in schema_rows],
                    "rows": [list(row) for row in sample_rows],
                },
            }
            return json.dumps(result, cls=_CustomEncoder)

        return [list_tables, query_database, describe_table, table_info]

    def _create_cohort_tools(self) -> list:
        db = self.db

        @tool
        def identify_cohort(sql: str, target_cohort: str = "") -> str:
            """Build or update a patient cohort by executing a SQL query that returns PATIENT_NUMBER values. ONLY use this tool when explicitly building, creating, or modifying a cohort. For general data queries, use query_database instead.

            Args:
                sql: A SQL query that returns a single column of patient IDs (e.g., 'SELECT PATIENT_NUMBER FROM demographics WHERE ...')
                target_cohort: If narrowing/expanding an existing cohort, the table name to overwrite (e.g. 'cohort_1'). Leave empty for a new cohort.
            """
            rows = db.query(sql)
            patient_ids = list(dict.fromkeys(str(row[0]) for row in rows))
            return json.dumps({"count": len(patient_ids), "sql": sql, "target_cohort": target_cohort}, cls=_CustomEncoder)

        @tool
        def compare_cohorts(cohort_a_table: str, cohort_b_table: str, cohort_a_name: str, cohort_b_name: str) -> str:
            """Compare two patient cohorts on demographics, mortality, top diagnoses, and top medications.

            Each cohort must be a temporary table containing PATIENT_NUMBER values.

            Args:
                cohort_a_table: Table name for cohort A (e.g. 'cohort_1').
                cohort_b_table: Table name for cohort B (e.g. 'cohort_2').
                cohort_a_name: Human-readable label for cohort A.
                cohort_b_name: Human-readable label for cohort B.
            """
            from scipy import stats

            ids_a = [row[0] for row in db.query(f"SELECT PATIENT_NUMBER FROM {cohort_a_table}")]
            ids_b = [row[0] for row in db.query(f"SELECT PATIENT_NUMBER FROM {cohort_b_table}")]

            if not ids_a or not ids_b:
                return json.dumps({"error": "One or both cohorts returned no patients."})

            def _cohort_stats(patient_ids, label):
                id_list = ",".join(str(i) for i in patient_ids)

                # Demographics
                demo_rows = db.query(
                    f"SELECT YEAR(CURRENT_DATE) - YEAR(PATIENT_YOB_DATE) AS age, PATIENT_SEX "
                    f"FROM demographics WHERE PATIENT_NUMBER IN ({id_list})"
                )
                ages = [r[0] for r in demo_rows]
                sexes = [r[1] for r in demo_rows]

                # Mortality
                death_count = db.query(
                    f"SELECT COUNT(*) FROM mortality WHERE PATIENT_NUMBER IN ({id_list})"
                )[0][0]

                # Top 10 diagnoses
                dx_rows = db.query(
                    f"SELECT dx_code, COUNT(*) AS cnt FROM ("
                    f"  SELECT UNNEST(STRING_SPLIT(TRIM(DIAGNOSIS_CODES, '|'), '|')) AS dx_code "
                    f"  FROM mx_events WHERE PATIENT_NUMBER IN ({id_list})"
                    f") WHERE dx_code != '' GROUP BY dx_code ORDER BY cnt DESC LIMIT 10"
                )

                # Top 10 medications
                rx_rows = db.query(
                    f"SELECT GENERIC_NAME, COUNT(DISTINCT PATIENT_NUMBER) AS cnt "
                    f"FROM rx_events WHERE PATIENT_NUMBER IN ({id_list}) "
                    f"AND GENERIC_NAME IS NOT NULL "
                    f"GROUP BY GENERIC_NAME ORDER BY cnt DESC LIMIT 10"
                )

                return {
                    "name": label,
                    "size": len(patient_ids),
                    "age_mean": round(sum(ages) / len(ages), 1) if ages else None,
                    "age_median": round(sorted(ages)[len(ages) // 2], 1) if ages else None,
                    "age_std": round((sum((a - sum(ages)/len(ages))**2 for a in ages) / len(ages))**0.5, 1) if ages else None,
                    "sex_male": sexes.count("M"),
                    "sex_female": sexes.count("F"),
                    "pct_male": round(sexes.count("M") / len(sexes) * 100, 1) if sexes else None,
                    "pct_female": round(sexes.count("F") / len(sexes) * 100, 1) if sexes else None,
                    "mortality_count": death_count,
                    "mortality_rate": round(death_count / len(patient_ids) * 100, 1),
                    "top_diagnoses": [{"code": r[0], "count": r[1]} for r in dx_rows],
                    "top_medications": [{"name": r[0], "patient_count": r[1]} for r in rx_rows],
                }

            stats_a = _cohort_stats(ids_a, cohort_a_name)
            stats_b = _cohort_stats(ids_b, cohort_b_name)

            # Statistical tests
            statistical_tests = {}

            # T-test on age
            ages_a = [r[0] for r in db.query(
                f"SELECT YEAR(CURRENT_DATE) - YEAR(PATIENT_YOB_DATE) FROM demographics "
                f"WHERE PATIENT_NUMBER IN ({','.join(str(i) for i in ids_a)})"
            )]
            ages_b = [r[0] for r in db.query(
                f"SELECT YEAR(CURRENT_DATE) - YEAR(PATIENT_YOB_DATE) FROM demographics "
                f"WHERE PATIENT_NUMBER IN ({','.join(str(i) for i in ids_b)})"
            )]
            if ages_a and ages_b:
                t_stat, t_p = stats.ttest_ind(ages_a, ages_b)
                statistical_tests["age_ttest"] = {"t_statistic": round(t_stat, 3), "p_value": round(t_p, 4)}

            # Chi-squared on sex
            male_a, female_a = stats_a["sex_male"], stats_a["sex_female"]
            male_b, female_b = stats_b["sex_male"], stats_b["sex_female"]
            if all(v > 0 for v in [male_a + female_a, male_b + female_b]):
                chi2, chi_p, _, _ = stats.chi2_contingency([[male_a, female_a], [male_b, female_b]])
                statistical_tests["sex_chi2"] = {"chi2_statistic": round(chi2, 3), "p_value": round(chi_p, 4)}

            # Relative risk on mortality
            rate_a = stats_a["mortality_count"] / stats_a["size"] if stats_a["size"] else 0
            rate_b = stats_b["mortality_count"] / stats_b["size"] if stats_b["size"] else 0
            if rate_b > 0:
                statistical_tests["mortality_relative_risk"] = round(rate_a / rate_b, 3)

            result = {
                "cohort_a": stats_a,
                "cohort_b": stats_b,
                "statistical_tests": statistical_tests,
            }
            return json.dumps(result, cls=_CustomEncoder)

        return [identify_cohort, compare_cohorts]
