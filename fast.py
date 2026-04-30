

import base64
import uuid
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import NotRequired
from dotenv import load_dotenv
import os

load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nutrition Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Stores: { session_id: { "config": ..., "status": ..., "interrupt": ..., "result": ... } }
sessions: dict[str, dict] = {}

# ── Preference Store ──────────────────────────────────────────────────────────
PREFS_FILE = Path("./user_preferences.json")

def load_preferences() -> dict:
    if PREFS_FILE.exists():
        return json.loads(PREFS_FILE.read_text())
    return {
        "preferred_nutrients": ["protein", "carbs", "fat", "calories"],
        "dietary_flags": [],
        "rejection_history": [],
    }

def save_preferences(prefs: dict):
    PREFS_FILE.write_text(json.dumps(prefs, indent=2))

# ── Custom State ──────────────────────────────────────────────────────────────
class CustomState(AgentState):
    identified_food_items: NotRequired[str]
    search_details:        NotRequired[str]
    nutrition_details:     NotRequired[str]
    identifier_retries:    NotRequired[int]
    summarizer_retries:    NotRequired[int]
    user_preferences:      NotRequired[dict]

MAX_RETRIES = 2

# ── LLM ──────────────────────────────────────────────────────────────────────
llm_groq = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_retries=2,
)

# ── Sub-agents (created once) ─────────────────────────────────────────────────
identifier_agent = create_agent(
    model=llm_groq,
    tools=[],
    system_prompt=(
        "You are a food identifier assistant. "
        "Identify every food item visible in the image and estimate its weight in grams. "
        "Return a plain list, one item per line, formatted as: 'item_name: Xg'"
    ),
)

summarization_agent = create_agent(
    model=llm_groq,
    tools=[],
    system_prompt=(
        "You are a nutrition expert. "
        "Given raw search results about food items, produce a clean table with columns: "
        "Food Item | Protein (g) | Carbs (g) | Fat (g) | Calories. "
        "Include a totals row at the bottom."
    ),
)

tavily_search_tool = TavilySearch(max_results=5, topic="general")

# ── Tools ─────────────────────────────────────────────────────────────────────
# image_data_uri is set per-request via closure variable
_current_image_uri: dict[str, str] = {}  # session_id → data URI

@tool("identifier", description="Identifies food items and their approximate weight from the meal image.")
def identify_food_items(instruction: str) -> str:
    # Uses the last uploaded image — in production, pass session_id via context
    image_uri = list(_current_image_uri.values())[-1] if _current_image_uri else ""
    result = identifier_agent.invoke({
        "messages": [HumanMessage(content=[
            {"type": "text",      "text": f"Identify every food item in this image. {instruction}"},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ])]
    })
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    return ai_msgs[-1].content if ai_msgs else "Could not identify items."

@tool("search_agent", description="Searches nutrition info for a given list of food items.")
def search_food_items(food_items: str) -> str:
    query = f"nutrition details protein carbs fat calories for: {food_items}"
    results = tavily_search_tool.invoke({"query": query})
    if isinstance(results, list):
        return "\n\n".join(r.get("content", "") for r in results)
    return str(results)

@tool("summarization_agent", description="Summarizes raw nutrition search results into a clean table.")
def summarize_nutrition_details(raw_nutrition_data: str) -> str:
    result = summarization_agent.invoke({
        "messages": [HumanMessage(content=[
            {"type": "text", "text": "Here is the raw nutrition data from search results:"},
            {"type": "text", "text": raw_nutrition_data},
            {"type": "text", "text": "Produce a clean nutrition table with totals."},
        ])]
    })
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    return ai_msgs[-1].content if ai_msgs else "Could not summarize."

# ── Middleware ────────────────────────────────────────────────────────────────
class HITLAfterToolMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    @hook_config(can_jump_to=["model", "end"])
    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]
        tool_result = handler(request)

        if tool_name not in ("identifier", "summarization_agent"):
            return tool_result

        result_content = (
            tool_result.content
            if isinstance(tool_result, ToolMessage)
            else str(tool_result)
        )

        retry_key = (
            "identifier_retries" if tool_name == "identifier"
            else "summarizer_retries"
        )
        current_retries = request.state.get(retry_key, 0)

        if current_retries >= MAX_RETRIES:
            return tool_result

        label = "Identified Items" if tool_name == "identifier" else "Nutrition Summary"
        decision = interrupt({
            "tool":         tool_name,
            "label":        label,
            "result":       result_content,
            "retry_count":  current_retries,
            "instructions": "approve / reject / edit",
        })

        decision_type = decision.get("type", "approve")

        if decision_type == "approve":
            return tool_result

        if decision_type == "edit":
            edited_content = decision.get("content", result_content)
            return ToolMessage(
                content=edited_content,
                tool_call_id=request.tool_call["id"],
                name=tool_name,
            )

        if decision_type == "reject":
            reason = decision.get("reason", "Please try again more carefully.")
            prefs = load_preferences()
            prefs["rejection_history"].append({
                "tool": tool_name,
                "reason": reason,
                "result_rejected": result_content[:200],
            })
            save_preferences(prefs)

            feedback_tool_msg = ToolMessage(
                content=f"[REJECTED] Human feedback: {reason}",
                tool_call_id=request.tool_call["id"],
                name=tool_name,
            )
            feedback_human_msg = HumanMessage(
                content=(
                    f"The {tool_name} output was rejected. "
                    f"Reason: {reason}. "
                    f"Please call {tool_name} again, addressing this feedback."
                )
            )
            return Command(
                update={
                    "messages": [feedback_tool_msg, feedback_human_msg],
                    retry_key: current_retries + 1,
                },
                goto="model",
            )

        return tool_result


class PreferenceInjectionMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def wrap_model_call(self, request, handler):
        ai_msgs = [m for m in request.messages if isinstance(m, AIMessage)]
        if len(ai_msgs) == 0:
            prefs = load_preferences()
            extra = (
                f"\n\nUser preferences: {json.dumps(prefs, indent=2)}"
                "\nHighlight the preferred nutrients and respect any dietary flags."
            )
            new_system = SystemMessage(
                content=list(request.system_message.content_blocks) + [
                    {"type": "text", "text": extra}
                ]
            )
            return handler(request.override(system_message=new_system))
        return handler(request)

# ── Agent ─────────────────────────────────────────────────────────────────────
checkpointer = InMemorySaver()

planner_agent = create_agent(
    model=llm_groq,
    tools=[identify_food_items, search_food_items, summarize_nutrition_details],
    system_prompt="""You are a nutrition assistant. Given a meal image, you must:
1. Call `identifier` to identify the food items in the image.
2. Call `search_agent` with the identified items to get nutrition data.
3. Call `summarization_agent` with the raw search results to produce a clean table.
4. Return the final nutrition table to the user.

Always follow this exact sequence. Do not skip any step.

Sample output format:
Food Item      | Protein (g) | Carbs (g) | Fat (g) | Calories
---------------|-------------|-----------|---------|----------
Grilled Chicken|     31      |     0     |    3.6  |   165
TOTAL          |     31      |     0     |    3.6  |   165
""",
    middleware=[
        PreferenceInjectionMiddleware(),
        HITLAfterToolMiddleware(),
    ],
    state_schema=CustomState,
    checkpointer=checkpointer,
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    session_id: str
    message: str

class StatusResponse(BaseModel):
    session_id: str
    # "running" | "interrupted" | "done" | "error"
    status: str
    interrupt: dict | None = None    # present when status == "interrupted"
    result: str | None = None        # present when status == "done"
    total_tool_calls: int | None = None

class ResumeRequest(BaseModel):
    type: Literal["approve", "reject", "edit"]
    reason: str | None = None        # for reject
    content: str | None = None       # for edit

class PreferencesUpdate(BaseModel):
    preferred_nutrients: list[str] | None = None
    dietary_flags: list[str] | None = None

# ── Helper: process agent result into session ─────────────────────────────────
def _process_result(session_id: str, result: Any):
    session = sessions[session_id]

    if result.interrupts:
        session["status"] = "interrupted"
        session["interrupt"] = result.interrupts[0].value
    else:
        final_messages = result.value.get("messages", [])
        ai_msgs = [m for m in final_messages if isinstance(m, AIMessage) and m.content]
        total_calls = sum(
            len(m.tool_calls) for m in final_messages if isinstance(m, AIMessage)
        )
        session["status"] = "done"
        session["result"] = ai_msgs[-1].content if ai_msgs else "No response."
        session["total_tool_calls"] = total_calls
        # Clean up image uri
        _current_image_uri.pop(session_id, None)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    """
    Upload a meal image to start nutrition analysis.
    Returns a session_id to poll /status/{session_id}.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    raw = await file.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = file.content_type  # e.g. image/jpeg
    image_data_uri = f"data:{mime};base64,{b64}"

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    # Store image URI so tools can access it
    _current_image_uri[session_id] = image_data_uri

    # Init session
    sessions[session_id] = {
        "config":   config,
        "status":   "running",
        "interrupt": None,
        "result":   None,
        "total_tool_calls": None,
    }

    try:
        result = planner_agent.invoke(
            {
                "messages": [HumanMessage(content=[
                    {"type": "text",      "text": "Analyze the meal in this image and give me full nutrition details."},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ])],
                "identifier_retries":  0,
                "summarizer_retries":  0,
                "user_preferences":    load_preferences(),
            },
            config=config,
            version="v2",
        )
        _process_result(session_id, result)
    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["result"] = str(e)

    return AnalyzeResponse(session_id=session_id, message="Analysis started.")


@app.get("/status/{session_id}", response_model=StatusResponse)
def get_status(session_id: str):
    """
    Poll this endpoint after /analyze.
    - status = 'interrupted' → human review needed, see `interrupt` field
    - status = 'done'        → see `result` field
    - status = 'error'       → see `result` field for error message
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found.")

    return StatusResponse(
        session_id=session_id,
        status=session["status"],
        interrupt=session.get("interrupt"),
        result=session.get("result"),
        total_tool_calls=session.get("total_tool_calls"),
    )


@app.post("/resume/{session_id}", response_model=StatusResponse)
def resume(session_id: str, body: ResumeRequest):
    """
    Resume an interrupted session with a human decision.

    Body examples:
      { "type": "approve" }
      { "type": "reject", "reason": "Missed the sauce" }
      { "type": "edit",   "content": "rice: 150g\\nchicken: 200g" }
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found.")
    if session["status"] != "interrupted":
        raise HTTPException(400, f"Session is not awaiting input (status: {session['status']}).")

    resume_value: dict = {"type": body.type}
    if body.type == "reject" and body.reason:
        resume_value["reason"] = body.reason
    if body.type == "edit" and body.content:
        resume_value["content"] = body.content

    session["status"] = "running"
    session["interrupt"] = None

    try:
        result = planner_agent.invoke(
            Command(resume=resume_value),
            config=session["config"],
            version="v2",
        )
        _process_result(session_id, result)
    except Exception as e:
        session["status"] = "error"
        session["result"] = str(e)

    return StatusResponse(
        session_id=session_id,
        status=session["status"],
        interrupt=session.get("interrupt"),
        result=session.get("result"),
        total_tool_calls=session.get("total_tool_calls"),
    )


@app.get("/preferences")
def get_preferences():
    """Read current user preferences."""
    return load_preferences()


@app.put("/preferences")
def update_preferences(body: PreferencesUpdate):
    """Update preferred nutrients or dietary flags."""
    prefs = load_preferences()
    if body.preferred_nutrients is not None:
        prefs["preferred_nutrients"] = body.preferred_nutrients
    if body.dietary_flags is not None:
        prefs["dietary_flags"] = body.dietary_flags
    save_preferences(prefs)
    return {"message": "Preferences updated.", "preferences": prefs}


@app.delete("/preferences/history")
def clear_rejection_history():
    """Clear the rejection history from preferences."""
    prefs = load_preferences()
    prefs["rejection_history"] = []
    save_preferences(prefs)
    return {"message": "Rejection history cleared."}


@app.get("/health")
def health():
    return {"status": "ok"}