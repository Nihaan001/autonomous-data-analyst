import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from helper import ALL_TOOLS, get_dataset_overview

load_dotenv()

SYSTEM_PROMPT = """You are an Autonomous Data Analyst Agent.
You have access to real Python tools (Pandas, NumPy, Matplotlib).

RULES:
1. NEVER calculate numbers yourself. ALWAYS use tools.
2. For overview questions, use analyze_dataset.
3. For numeric stats, use calculate_statistics.
4. For relationships, use calculate_correlation.
5. For category-wise analysis, use groupby_analysis.
6. For charts, use generate_chart.
7. After tool results, explain insights in simple business language.
8. Mention trends, max/min, missing values, and unusual patterns.

Example: If the user asks "Which category has highest sales?", call
groupby_analysis(group_column='Category', agg_column='Sales', agg_func='sum') then explain.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY in .env")
    return key


def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=0,
        google_api_key=get_api_key(),
    )


def create_agent_graph():
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


graph = None


def get_graph():
    global graph
    if graph is None:
        graph = create_agent_graph()
    return graph


def _content_to_text(content) -> str:
    """Newer Gemini models return a list of blocks; convert to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type", "text") == "text":
                    parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def ask_agent(user_question: str, dataset_context: str = None):
    app = get_graph()
    if dataset_context:
        full_question = f"Dataset Context:\n{dataset_context}\n\nUser Question: {user_question}"
    else:
        full_question = user_question

    result = app.invoke({"messages": [HumanMessage(content=full_question)]})
    return _content_to_text(result["messages"][-1].content)


def get_auto_insights():
    context = get_dataset_overview()
    question = (
        "Provide a complete autonomous analysis: "
        "1) Dataset health (missing, duplicates) "
        "2) Key statistics "
        "3) Top insights and trends "
        "4) Useful charts. Use tools. Be concise with bullet points."
    )
    return ask_agent(question, context)