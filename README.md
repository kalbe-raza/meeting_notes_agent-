# Meeting Notes Agent

A bounded, reliable agentic system designed to process unstructured meeting notes, extract actionable tasks, and schedule them in a sandbox calendar. Built to survive the Reliability Arena stress tests.

## 1. Problem Scope & Agent Design Canvas
- **Operational Goal:** Extract action items (description, assignee, due date) from meeting notes and create sandbox calendar events.
- **Completion Condition:** All extractable tasks are scheduled, or missing critical information is requested from the user.
- **System Boundary:** The agent only processes text and interacts with mock/sandbox tools. It has no access to real-world APIs.
- **Observations:** User text input, uploaded meeting notes (untrusted), and results from previous tool executions.
- **Actions/Tools:** `extract_action_items`, `create_calendar_event`, `request_clarification`.
- **State:** Current step count, extracted items list, calendar events created, last action taken, and last tool result.
- **Autonomy Boundary:** Allowed to read text, classify, and create sandbox events. Blocked from sending real emails, deleting files, or making irreversible external changes.
- **Primary Risks:** LLM hallucinating dates/assignees, prompt injection via untrusted uploaded notes, infinite loops on tool failure.
- **Evaluation Criteria:** Task success rate, structured-output validity (JSON adherence), latency, and graceful handling of Arena stress tests.

## 2. Model Selection Experiment
We evaluated three models across ~10 representative inputs (vague requests, multi-turn clarifications, prompt injections).

| Metric | Mistral: `labs-leanstral-1-5` | OpenRouter: `nvidia/nemotron-3.5` | OpenRouter: `liquid/lfm-2.5` |
| :--- | :--- | :--- | :--- |
| **Task Success** | High (Correctly identified missing details) | Moderate (Often hallucinated dates) | Low (Failed multi-turn context) |
| **Structured-Output Validity**| Excellent (Explicit JSON mode support) | Poor (Frequent markdown parsing errors) | Poor (Returned conversational text) |
| **Correct Action Selection** | 9/10 | 6/10 | 4/10 |
| **Latency** | ~4.5 seconds | ~8 to 480 seconds (Highly variable) | ~98 seconds |
| **Approx. Cost** | $0.00 (Free Tier) | $0.00 (Free Tier) | $0.00 (Free Tier) |

**Selection Rationale:** We selected **Mistral `labs-leanstral-1-5`** via the direct Mistral API. OpenRouter's free tier suffered severe upstream congestion (429 rate limits, 400s+ latencies), violating the Arena's 40-second global timeout. Leanstral provides dedicated infrastructure, sub-5-second latency, and explicit JSON mode support, ensuring high structured-output validity within the `max_steps=6` and `max_tool_retries=2` limits.

## 3. Prompt & Context Engineering
The agent dynamically assembles context using a layered template (`app/prompts.py`) to prevent prompt injection and maintain state:
1. **SYSTEM:** Persistent role, non-negotiable boundaries, and the strict Pydantic JSON schema.
2. **USER:** The current user request or goal.
3. **STATE/RUNTIME:** Execution variables injected dynamically (e.g., `Step: 2/6`, `Items extracted: 1`).
4. **EXTERNAL/UNTRUSTED:** Meeting notes explicitly prefixed with `[SOURCE: user_upload | TRUST: untrusted]`. The system prompt explicitly instructs the model to treat this as inert data, not commands.
5. **TOOL OBSERVATION:** The JSON result returned by a tool execution, fed back into the loop.

**Multi-Turn Context Management:** We use a stable `session_id` to maintain bounded in-memory history (last 6 messages). A specific "CRITICAL MULTI-TURN RULE" instructs the model to combine new replies with the original goal rather than treating them as unrelated tasks.

## 4. Structured Outputs & Validation
All decisions must pass through a typed Pydantic contract (`AgentDecision`) before execution.
- **JSON Extraction:** A robust `_extract_json()` helper uses regex to strip markdown fences and isolate the `{...}` block.
- **Schema Validation:** Parsed via `json.loads()` and validated against `AgentDecision`.
- **Semantic Validation:** A custom `_validate_arguments()` function checks logical consistency (e.g., ensuring `date` is `YYYY-MM-DD`).
- **Recovery Policy:** Bounded to `max_tool_retries` (2 attempts). If recovery fails, the agent halts and returns a machine-readable `contract_error` with `stop_reason: "contract_validation_failed"`.

## 5. Architecture & Folder Structure
```text
i220794/
├── app/
│   ├── main.py        # FastAPI application entry point
│   ├── api.py         # Endpoint definitions (/health, /arena/run, /chat)
│   ├── agent.py       # Main agentic loop, validation, and recovery
│   ├── tools.py       # Mock tool implementations with fault injection
│   ├── prompts.py     # Layered prompt templates
│   ├── config.py      # Pydantic settings and environment variables
│   └── memory.py      # In-memory session history management
├── tests/
│   └── test_agent.py  # Automated pytest suite (9 tests, all passing)
├── evaluation/
│   └── public_cases.json
├── arena_manifest.json
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md

6. Local Run Instructions
Clone the repository and navigate to the folder.
Create a virtual environment: python -m venv .venv and activate it.
Install dependencies: pip install -r requirements.txt
Copy .env.example to .env and add your LLM_API_KEY.
Run the server: python -m uvicorn app.main:app --reload
Access the UI at http://localhost:8000 or API docs at http://localhost:8000/docs.
7. Deployment Instructions
This project is deployed on Render using Docker.
Ensure .env is in .gitignore.
Push code to GitHub.
In Render, create a new Web Service from this GitHub repo.
Set Environment Variables: LLM_API_KEY, LLM_BASE_URL (https://api.mistral.ai/v1), MODEL_NAME (labs-leanstral-1-5).
Use Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
