# 🥗 NutriLens

> AI-powered meal nutrition analyser with human-in-the-loop review.

Upload a photo of any meal and NutriLens will identify every food item, search for nutrition data, and return a detailed breakdown of protein, carbs, fat, and calories — with you in control at every step.

---

## Architecture

```
Streamlit UI  ──────►  FastAPI Backend  ──────►  Planner Agent (LangChain)
   frontend.py            nutrition_api.py              │
                                                        ├── identifier tool
                                                        │     └── Identifier Sub-agent (Groq + LLaMA 4)
                                                        │
                                                        ├── search_agent tool
                                                        │     └── TavilySearch
                                                        │
                                                        └── summarization_agent tool
                                                              └── Summarizer Sub-agent (Groq + LLaMA 4)
```

### Agent Pipeline

```
Image Input
    ↓
identifier          — identifies food items + weight estimates
    ↓
⏸ HITL #1          — human reviews identified items (approve / reject / edit)
    ↓
search_agent        — searches nutrition data for each item via Tavily
    ↓
summarization_agent — formats results into a clean nutrition table
    ↓
⏸ HITL #2          — human reviews the nutrition summary (approve / reject / edit)
    ↓
Final Report        — metric cards + detailed table shown in UI
```

---

## Project Structure

```
nutrilens/
├── nutrition_api.py        # FastAPI backend — agent, tools, middleware, endpoints
├── frontend.py             # Streamlit chat UI
├── user_preferences.json   # Auto-created — stores dietary flags + rejection history
└── images.jpeg             # Sample meal image (replace with your own)
```

---

## Setup

### 1. Install dependencies

```bash
pip install fastapi uvicorn python-multipart streamlit requests \
            langchain langchain-groq langchain-tavily langgraph \
            python-dotenv
```

### 2. Set environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get your keys from:
- Groq — https://console.groq.com
- Tavily — https://app.tavily.com

### 3. Run the backend

```bash
uvicorn nutrition_api:app --reload
# API running at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

### 4. Run the frontend

```bash
streamlit run frontend.py
# UI running at http://localhost:8501
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload a meal image to start analysis. Returns `session_id`. |
| `GET` | `/status/{session_id}` | Poll for current status: `running`, `interrupted`, `done`, or `error`. |
| `POST` | `/resume/{session_id}` | Send a human decision to resume an interrupted session. |
| `GET` | `/preferences` | Read current user preferences. |
| `PUT` | `/preferences` | Update preferred nutrients or dietary flags. |
| `DELETE` | `/preferences/history` | Clear the rejection history. |
| `GET` | `/health` | Health check. |

### Resume payload examples

```json
// Approve
{ "type": "approve" }

// Reject with feedback
{ "type": "reject", "reason": "You missed the sauce on the left" }

// Edit output manually
{ "type": "edit", "content": "rice: 150g\nchicken: 200g\nsauce: 50g" }
```

### Status response

```json
{
  "session_id": "abc-123",
  "status": "interrupted",
  "interrupt": {
    "tool": "identifier",
    "label": "Identified Items",
    "result": "rice: 150g\nchicken: 200g",
    "retry_count": 0,
    "instructions": "approve / reject / edit"
  },
  "result": null,
  "total_tool_calls": null
}
```

---

## Human-in-the-Loop (HITL)

NutriLens pauses twice during every analysis and asks for your input:

| Checkpoint | When | What you review |
|------------|------|-----------------|
| HITL #1 | After `identifier` runs | List of detected food items + weight estimates |
| HITL #2 | After `summarization_agent` runs | Full nutrition table before it's shown |

At each checkpoint you can:

- **Approve** — accept the output and continue
- **Reject** — provide feedback; the agent retries (up to `MAX_RETRIES = 2` times)
- **Edit** — manually correct the output text and continue with your version

If the agent hits max retries it proceeds automatically with the best attempt.

---

## Middleware

Two custom `AgentMiddleware` classes power the HITL and preference injection:

### `HITLAfterToolMiddleware`

Uses `wrap_tool_call` to intercept **after** `identifier` and `summarization_agent` run — so the human reviews the **output**, not the input. On reject, injects a `HumanMessage` with the feedback and returns `Command(goto="model")` to loop back to the LLM.

### `PreferenceInjectionMiddleware`

Uses `wrap_model_call` to inject the user's saved preferences into the system prompt on the very first model call of each session. This means the planner knows your dietary flags and preferred nutrients before it makes any decisions.

---

## User Preferences

Preferences are stored in `user_preferences.json` and persist across sessions.

```json
{
  "preferred_nutrients": ["protein", "carbs", "fat", "calories"],
  "dietary_flags": ["low-carb"],
  "rejection_history": [
    {
      "tool": "identifier",
      "reason": "Missed the sauce",
      "result_rejected": "rice: 150g\nchicken: 200g"
    }
  ]
}
```

The planner reads this at the start of every run and adjusts its output accordingly — for example, highlighting protein if you're tracking muscle gain, or flagging high-carb items if you've set a low-carb flag.

Manage preferences directly from the Streamlit sidebar — no need to edit the JSON file manually.

---

## Models Used

| Role | Model | Provider |
|------|-------|----------|
| Planner, Identifier, Summarizer | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq |
| Web search | Tavily Search API | Tavily |

---

## Configuration

All tuneable constants are at the top of each file:

**`nutrition_api.py`**
```python
MAX_RETRIES = 2   # max human rejections before agent proceeds anyway
```

**`frontend.py`**
```python
API_BASE       = "http://127.0.0.1:8000"
POLL_INTERVAL  = 2      # seconds between status polls
POLL_MAX_TRIES = 90     # timeout = POLL_INTERVAL × POLL_MAX_TRIES (~3 min)
MAX_RETRIES    = 2      # must match nutrition_api.py
```

---

## Known Limitations

- **Single user** — session store is in-memory (`dict`). For multi-user production use, replace with Redis or a database.
- **Image URL** — Groq's API requires the image to be passed as a base64 data URI. Very large images may hit token limits.
- **Nutrition accuracy** — values come from Tavily web search and are approximate. Not a substitute for medical dietary advice.
- **Preference learning** — rejection history is logged but the planner only reads `preferred_nutrients` and `dietary_flags` directly. Richer learning from rejection history requires additional prompt engineering.
