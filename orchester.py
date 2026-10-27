
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from helper import ALL_TOOLS, get_dataset_overview


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

MAX_GRAPH_STEPS = 20

SYSTEM_PROMPT = """
You are an Autonomous Data Analyst Agent.

You have access to Python data-analysis tools using Pandas,
NumPy, and Matplotlib.

RULES:
1. Always use tools for numerical calculations.
2. For dataset overview, use analyze_dataset.
3. For statistics, use calculate_statistics.
4. For correlations, use calculate_correlation.
5. For category analysis, use groupby_analysis.
6. For charts, use generate_chart.
7. For missing values, use find_missing_values.
8. For duplicate detection, use find_duplicates.
9. Use only columns that exist in the dataset.
10. Never invent numerical values or claim a chart succeeded
    unless the tool result supports that claim.
11. Explain results in simple business language.
12. Clearly report unavailable data and tool failures.
"""

REVIEWER_PROMPT = """
You are the Reviewer Agent for an Autonomous Data Analyst.

Review the preceding analysis and available tool results.

Check:
- Whether numerical claims match the available tool results.
- Whether conclusions are supported by the evidence.
- Whether nonexistent columns or fabricated values are used.
- Whether chart creation is claimed without supporting evidence.

Do not invent numbers or claim independent verification
without evidence. Correct unsupported claims and return a
clear final report. If evidence is insufficient, state that.
"""


# --------------------------------------------------
# GRAPH STATE
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# --------------------------------------------------
# GEMINI MODEL
# --------------------------------------------------

def get_api_key() -> str:
    key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not key or not key.strip():
        raise ValueError(
            "API key missing. Set GOOGLE_API_KEY or "
            "GEMINI_API_KEY in your .env file."
        )

    return key.strip()


def get_llm():
    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=get_api_key(),
    )


# --------------------------------------------------
# CREATE LANGGRAPH
# --------------------------------------------------

def create_agent_graph():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Main agent
    def agent_node(state: AgentState):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    # Python tool execution
    tool_node = ToolNode(
        ALL_TOOLS,
        handle_tool_errors=True,
    )

    # Reviewer node
    def reviewer_node(state: AgentState):
        messages = [
            SystemMessage(content=REVIEWER_PROMPT)
        ] + state["messages"]

        try:
            response = llm.invoke(messages)
            return {"messages": [response]}

        except Exception as exc:
            # Preserve the existing agent response if review fails.
            print(f"Reviewer warning: {exc}")
            return {}

    # Route agent to tools or reviewer
    def route_after_agent(state: AgentState):
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        return "reviewer"

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "reviewer": "reviewer",
        },
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("reviewer", END)

    return workflow.compile()


# --------------------------------------------------
# GRAPH INSTANCE
# --------------------------------------------------

graph = None


def get_graph():
    global graph

    if graph is None:
        graph = create_agent_graph()

    return graph


# --------------------------------------------------
# RESPONSE FORMATTER
# --------------------------------------------------

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, dict):
                if part.get("type", "text") == "text":
                    text_part = part.get("text", "")
                    if text_part:
                        parts.append(text_part)
            else:
                parts.append(str(part))

        return "\n".join(parts)

    return str(content)


# --------------------------------------------------
# ERROR HANDLER
# --------------------------------------------------

def _friendly_error(exc: Exception) -> str:
    message = str(exc).lower()

    if (
        "resource_exhausted" in message
        or "quota" in message
        or "429" in message
    ):
        return (
            "Gemini API quota exceeded. Check your API quota, "
            "model limits, and try again later."
        )

    if (
        "unauthenticated" in message
        or "permission_denied" in message
        or "invalid api key" in message
        or "401" in message
        or "403" in message
    ):
        return (
            "Gemini authentication failed. Check your API key "
            "and permissions in the local .env file."
        )

    if "recursion" in message:
        return (
            "The agent reached its execution limit. "
            "Try a simpler question."
        )

    return (
        "Analysis failed. Check the VS Code terminal for "
        "the technical error, then try again."
    )


# --------------------------------------------------
# ASK AGENT
# --------------------------------------------------

def ask_agent(
    user_question: str,
    dataset_context: str = None,
):
    if (
        not isinstance(user_question, str)
        or not user_question.strip()
    ):
        return "Please enter a valid question."

    if dataset_context:
        full_question = (
            f"Dataset Context:\n{dataset_context}\n\n"
            f"User Question: {user_question}"
        )
    else:
        full_question = user_question

    try:
        app = get_graph()

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content=full_question)
                ]
            },
            config={
                "recursion_limit": MAX_GRAPH_STEPS,
            },
        )

        messages = result.get("messages", [])

        if not messages:
            return "The agent returned no response."

        answer = _content_to_text(messages[-1].content)

        if not answer.strip():
            return "The agent returned an empty response."

        return answer

    except Exception as exc:
        print(f"Agent execution error: {exc}")
        return _friendly_error(exc)


# --------------------------------------------------
# AUTOMATIC DATASET INSIGHTS
# --------------------------------------------------

def get_auto_insights():
    try:
        context = get_dataset_overview()

        question = """
        Analyze the currently loaded dataset.

        1. Summarize its structure and important columns.
        2. Identify missing values and duplicate rows.
        3. Calculate useful numerical statistics.
        4. Identify category-wise patterns and trends.
        5. Generate useful charts using available tools.
        6. Provide key findings and business recommendations.

        Use tools for calculations.
        Use actual data only.
        State clearly if any step cannot be completed.
        """

        return ask_agent(question, context)

    except Exception as exc:
        print(f"Automatic insights error: {exc}")
        return _friendly_error(exc)
