# Requires Python 3.10+
# pip install --upgrade langchain langgraph langchain-groq langchain-tavily

import os
import base64
import uuid
import json
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
)
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing_extensions import NotRequired
from dotenv import load_dotenv

load_dotenv()



# ── Preference Store (JSON file) ──────────────────────────────────────────────
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
    identifier_retries:    NotRequired[int]   # retry counter for HITL #1
    summarizer_retries:    NotRequired[int]   # retry counter for HITL #2
    user_preferences:      NotRequired[dict]  # loaded once at start

MAX_RETRIES = 2  # max times to retry on reject before proceeding anyway

# ── LLM & Tools ──────────────────────────────────────────────────────────────
llm_groq = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_retries=2,
)

# Sub-agents created once at module level (not inside tools)
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

# Read image once at module level — base64 data URI
with open("./image2.jpeg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
image_data_uri = f"data:image/jpeg;base64,{image_b64}"


# ── Tools (wrapped sub-agents) ────────────────────────────────────────────────

@tool("identifier", description="Identifies food items and their approximate weight from the meal image.")
def identify_food_items(instruction: str) -> str:
    """
    Calls the identifier sub-agent on the stored image.
    Pass any extra instruction (e.g., 'look more carefully at sauces').
    """
    result = identifier_agent.invoke({
        "messages": [HumanMessage(content=[
            {"type": "text",      "text": f"Identify every food item in this image. {instruction}"},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ])]
    })
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    return ai_msgs[-1].content if ai_msgs else "Could not identify items."


@tool("search_agent", description="Searches nutrition info for a given list of food items.")
def search_food_items(food_items: str) -> str:
    """
    Runs a Tavily search for nutrition details of the identified food items.
    Pass the comma-separated food item list.
    """
    query = f"nutrition details protein carbs fat calories for: {food_items}"
    results = tavily_search_tool.invoke({"query": query})
    # results is a list of dicts with 'content' keys
    if isinstance(results, list):
        return "\n\n".join(r.get("content", "") for r in results)
    return str(results)


@tool("summarization_agent", description="Summarizes raw nutrition search results into a clean table.")
def summarize_nutrition_details(raw_nutrition_data: str) -> str:
    """
    Calls the summarization sub-agent on raw search results.
    Pass the raw search result text.
    """
    result = summarization_agent.invoke({
        "messages": [HumanMessage(content=[
            {"type": "text", "text": "Here is the raw nutrition data from search results:"},
            {"type": "text", "text": raw_nutrition_data},
            {"type": "text", "text": "Produce a clean nutrition table with totals."},
        ])]
    })
    ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    return ai_msgs[-1].content if ai_msgs else "Could not summarize."


# ── Custom HITL Middleware ─────────────────────────────────────────────────────
#
# Uses wrap_tool_call so we intercept AFTER the tool runs and can review its result.
# For identifier and summarizer → interrupt() → human approves/rejects/edits.
# On reject  → inject rejection feedback as a HumanMessage, jump_to="model" (retry)
# On approve → return the tool result as-is (normal flow continues)
#
class HITLAfterToolMiddleware(AgentMiddleware[CustomState]):
    """
    Interrupts AFTER identifier and summarization_agent tools run,
    so the human reviews the OUTPUT (not the input).
    Supports approve / reject (with retry up to MAX_RETRIES) / edit.
    """
    state_schema = CustomState

    @hook_config(can_jump_to=["model", "end"])
    def wrap_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]

        # ── Run the tool first ────────────────────────────────────────────────
        tool_result = handler(request)

        # ── Only intercept identifier and summarizer ──────────────────────────
        if tool_name not in ("identifier", "summarization_agent"):
            return tool_result  # search_agent runs freely

        # Extract the text content from the ToolMessage
        result_content = (
            tool_result.content
            if isinstance(tool_result, ToolMessage)
            else str(tool_result)
        )

        # ── Determine which retry counter to use ─────────────────────────────
        retry_key = (
            "identifier_retries" if tool_name == "identifier"
            else "summarizer_retries"
        )
        current_retries = request.state.get(retry_key, 0)

        # If already hit max retries, skip interruption and proceed
        if current_retries >= MAX_RETRIES:
            print(f"\n⚠  Max retries ({MAX_RETRIES}) reached for {tool_name}. Proceeding.")
            return tool_result

        # ── Interrupt: show result to human ──────────────────────────────────
        label = "Identified Items" if tool_name == "identifier" else "Nutrition Summary"
        decision = interrupt({
            "tool":         tool_name,
            "label":        label,
            "result":       result_content,
            "retry_count":  current_retries,
            "instructions": (
                "Type 'approve' to continue, "
                "'reject' to retry with feedback, "
                "or 'edit' to modify the output manually."
            ),
        })

        decision_type = decision.get("type", "approve")

        # ── Approve ───────────────────────────────────────────────────────────
        if decision_type == "approve":
            return tool_result  # normal flow continues

        # ── Edit: human provides corrected text directly ──────────────────────
        if decision_type == "edit":
            edited_content = decision.get("content", result_content)
            # Return a ToolMessage with the human-edited content
            return ToolMessage(
                content=edited_content,
                tool_call_id=request.tool_call["id"],
                name=tool_name,
            )

        # ── Reject: inject feedback and retry via jump_to="model" ─────────────
        if decision_type == "reject":
            reason = decision.get("reason", "Please try again more carefully.")

            # Save rejection to preferences
            prefs = load_preferences()
            prefs["rejection_history"].append({
                "tool": tool_name,
                "reason": reason,
                "result_rejected": result_content[:200],
            })
            save_preferences(prefs)

            # Build a feedback ToolMessage (required — every tool_call needs one)
            # then add a HumanMessage with instructions so the LLM retries correctly
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
                goto="model",   # jump_to equivalent inside Command
            )

        # Fallback
        return tool_result


# ── Preference Injection Middleware ───────────────────────────────────────────
class PreferenceInjectionMiddleware(AgentMiddleware[CustomState]):
    """
    Reads user preferences from disk on the very first model call
    and injects them into the system prompt context.
    """
    state_schema = CustomState

    def wrap_model_call(self, request, handler):
        # Only inject on the first call (no prior AI messages)
        ai_msgs = [m for m in request.messages if isinstance(m, AIMessage)]
        if len(ai_msgs) == 0:
            prefs = load_preferences()
            extra = (
                f"\n\nUser preferences: {json.dumps(prefs, indent=2)}"
                "\nHighlight the preferred nutrients and respect any dietary flags."
            )
            from langchain_core.messages import SystemMessage
            new_system = SystemMessage(
                content=list(request.system_message.content_blocks) + [
                    {"type": "text", "text": extra}
                ]
            )
            return handler(request.override(system_message=new_system))
        return handler(request)


# ── Planner Agent ─────────────────────────────────────────────────────────────
tools = [identify_food_items, search_food_items, summarize_nutrition_details]

planner_agent = create_agent(
    model=llm_groq,
    tools=tools,
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
Brown Rice     |     2.6     |    23     |    0.9  |   111
TOTAL          |     33.6    |    23     |    4.5  |   276
""",
    middleware=[
        PreferenceInjectionMiddleware(),  # reads prefs, injects on first call
        HITLAfterToolMiddleware(),        # HITL after identifier + summarizer
    ],
    state_schema=CustomState,
    checkpointer=InMemorySaver(),
)

# ── Run ───────────────────────────────────────────────────────────────────────
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("=" * 60)
print("Starting nutrition analysis...")
print("=" * 60)

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
    version="v2",  # required for interrupt support with the new API
)

# ── Interrupt Loop ────────────────────────────────────────────────────────────
while result.interrupts:
    interrupt_data = result.interrupts[0].value

    print(f"\n⏸  REVIEW REQUIRED — {interrupt_data['label']}")
    print(f"  Tool        : {interrupt_data['tool']}")
    print(f"  Retry count : {interrupt_data['retry_count']}/{MAX_RETRIES}")
    print(f"\n── Output ──────────────────────────────────────")
    print(interrupt_data["result"])
    print(f"────────────────────────────────────────────────")
    print(f"\n{interrupt_data['instructions']}")

    action = input("\nDecision (approve / reject / edit): ").strip().lower()

    if action == "approve":
        resume_value = {"type": "approve"}

    elif action == "reject":
        reason = input("Rejection reason: ").strip()
        resume_value = {"type": "reject", "reason": reason}

    elif action == "edit":
        print("Paste your corrected version (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        resume_value = {"type": "edit", "content": "\n".join(lines)}

    else:
        print("Unknown input — defaulting to approve.")
        resume_value = {"type": "approve"}

    # Resume the agent with the human decision
    result = planner_agent.invoke(
        Command(resume=resume_value),
        config=config,
        version="v2",
    )

# ── Final Output ──────────────────────────────────────────────────────────────
final_messages = result.value.get("messages", [])
ai_msgs = [m for m in final_messages if isinstance(m, AIMessage) and m.content]
total_tool_calls = sum(
    len(m.tool_calls) for m in final_messages if isinstance(m, AIMessage)
)

print(f"\n{'=' * 60}")
print(f"Total tool calls made: {total_tool_calls}")
print("── Final Nutrition Report ──────────────────────────────────")
print(ai_msgs[-1].content if ai_msgs else "No final response.")
print("=" * 60)