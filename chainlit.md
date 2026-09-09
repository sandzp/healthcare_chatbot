# Healthcare Analytics Chatbot

A natural language interface for querying healthcare claims data. Ask questions, explore the database, and build patient cohorts — all through conversation.

## What can I do?

**Query the database** — Ask questions about patients, diagnoses, prescriptions, and medical events in plain English. The chatbot will write and execute SQL for you.

**Build cohorts** — Define patient populations using natural language (e.g., *"Find patients with diabetes on metformin"*). Cohorts are saved as named groups you can reference in follow-up questions.

**Compare cohorts** — Ask the chatbot to compare two cohorts and get a statistical breakdown across demographics, diagnoses, medications, and mortality.

## Example questions

- "What tables are available in the database?"
- "How many patients are in the database?"
- "Show me the most common diagnosis codes"
- "Build a cohort of patients over 65 with hypertension"
- "Compare cohort_1 and cohort_2"

## Tips

- The chatbot will explore the database schema before writing queries — you can watch its reasoning in the collapsible tool steps below each response.
- Cohorts persist across your conversation. Once created, you can ask follow-up questions about them without rebuilding.
- Start a new chat from the sidebar to begin a fresh session. Previous conversations are saved and can be resumed.
- **Note:** Cohort temp tables live in memory for the current session. If you switch to a different thread or resume an old one, previously created cohorts will need to be rebuilt.
