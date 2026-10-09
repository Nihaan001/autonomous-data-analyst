import os

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from helper import load_dataset, get_dataset_overview, LAST_FIG_PATH, set_dataframe
from orchester import ask_agent, get_auto_insights

load_dotenv()

st.set_page_config(page_title="Autonomous Data Analyst", page_icon="📊", layout="wide")

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample.csv")

SUGGESTIONS = [
    "Give me an overview of this dataset",
    "Show missing values",
    "Show descriptive statistics",
    "Plot correlation heatmap",
    "Find duplicate rows",
    "Which columns are most important to look at?",
]

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
footer {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    padding: 2.2rem 2.4rem; border-radius: 22px; color: #fff;
    box-shadow: 0 12px 32px rgba(99, 102, 241, 0.35); margin-bottom: 1.4rem;
}
.hero-title {font-size: 2.3rem; font-weight: 800; line-height: 1.15; margin-bottom: .4rem;}
.hero-sub {font-size: 1.05rem; opacity: .92; margin-bottom: 1rem;}
.badge {
    display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
    padding: .25rem .8rem; border-radius: 999px; font-size: .82rem; margin-right: .45rem; font-weight: 600;
}

.stat {
    display: flex; align-items: center; gap: .9rem; padding: 1rem 1.1rem; border-radius: 16px;
    background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stat:hover {transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,.18);}
.stat-icon {font-size: 1.7rem;}
.stat-value {font-size: 1.5rem; font-weight: 800; line-height: 1.1;}
.stat-label {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}

.feature {
    padding: 1.4rem; border-radius: 18px; height: 100%;
    background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25);
}
.feature-icon {font-size: 2rem; margin-bottom: .4rem;}
.feature-title {font-weight: 700; font-size: 1.1rem; margin-bottom: .3rem;}
.feature-text {opacity: .75; font-size: .92rem;}

[data-testid="stChatMessage"] {
    border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: .6rem;
    background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18);
}
div.stButton > button {
    width: 100%; border-radius: 12px; font-weight: 600; padding: .55rem .9rem;
    border: 1px solid rgba(128,128,128,.35); transition: all .15s ease;
}
div.stButton > button:hover {
    border-color: #7c3aed; color: #7c3aed; transform: translateY(-1px);
}
button[data-baseweb="tab"] {font-weight: 600; font-size: 1rem;}
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
.side-card {
    padding: .8rem 1rem; border-radius: 14px; margin: .6rem 0 1rem 0;
    background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.4); font-size: .9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- state
for key, default in {
    "messages": [],
    "df": None,
    "file_sig": None,
    "dataset_name": None,
    "pending": None,
    "want_insights": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------------------------------- helpers
def set_pending(text: str):
    st.session_state.pending = text


def clear_chat():
    st.session_state.messages = []


def request_insights():
    st.session_state.want_insights = True


def fig_mtime() -> float:
    return os.path.getmtime(LAST_FIG_PATH) if os.path.exists(LAST_FIG_PATH) else 0.0


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return ("⏳ **Gemini quota is used up.** The free tier has a small daily limit. "
                "Try again later, switch `GEMINI_MODEL` in `.env`, or enable billing.")
    if "401" in text or "UNAUTHENTICATED" in text or "API key" in text:
        return "🔑 **API key problem.** Check `GOOGLE_API_KEY` in your `.env` file and restart the app."
    if "404" in text or "NOT_FOUND" in text:
        return "🤖 **Model not found.** Update `GEMINI_MODEL` in your `.env` file and restart the app."
    return f"⚠️ Something went wrong: {text[:300]}"


def run_and_store(user_text: str, fn):
    """Run the agent, store the Q&A, and keep the chart only if this call created a new one."""
    st.session_state.messages.append({"role": "user", "content": user_text})
    set_dataframe(st.session_state.df)
    before = fig_mtime()
    with st.spinner("🤖 Agent is thinking and running tools..."):
        try:
            answer = fn()
        except Exception as exc:  # show a friendly message instead of a crash
            answer = friendly_error(exc)
    image = None
    if fig_mtime() > before:
        with open(LAST_FIG_PATH, "rb") as f:
            image = f.read()
    st.session_state.messages.append({"role": "assistant", "content": answer, "image": image})


def stat_card(icon: str, value, label: str) -> str:
    return (f'<div class="stat"><div class="stat-icon">{icon}</div>'
            f'<div><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div></div>')


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 📂 Your data")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        sig = (uploaded.name, uploaded.size)
        if st.session_state.file_sig != sig:
            try:
                uploaded.seek(0)
                st.session_state.df = load_dataset(uploaded)
                st.session_state.file_sig = sig
                st.session_state.dataset_name = uploaded.name
                st.session_state.messages = []
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    if os.path.exists(SAMPLE_PATH) and st.button("🧪 Try the sample dataset"):
        st.session_state.df = load_dataset(SAMPLE_PATH)
        st.session_state.file_sig = ("sample", 0)
        st.session_state.dataset_name = "sample.csv"
        st.session_state.messages = []

    if st.session_state.df is not None:
        d = st.session_state.df
        st.markdown(
            f'<div class="side-card">✅ <b>{st.session_state.dataset_name}</b><br>'
            f'{d.shape[0]:,} rows · {d.shape[1]} columns</div>',
            unsafe_allow_html=True,
        )
        st.button("✨ Generate insights", on_click=request_insights)
        st.button("🧹 Clear chat", on_click=clear_chat)

    st.divider()
    st.caption("Powered by Gemini · LangGraph · Pandas")

# --------------------------------------------------------------------------- hero
st.markdown(
    """
<div class="hero">
  <div class="hero-title">📊 Autonomous Data Analyst</div>
  <div class="hero-sub">Upload a dataset, ask questions in plain English, and let the agent
  run real Pandas tools, draw charts and explain the results.</div>
  <span class="badge">Gemini</span><span class="badge">LangGraph</span><span class="badge">Pandas</span>
</div>
""",
    unsafe_allow_html=True,
)

df = st.session_state.df

if df is None:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="feature"><div class="feature-icon">💬</div><div class="feature-title">Ask in plain English</div>'
                '<div class="feature-text">No code needed. Just type a question about your data.</div></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="feature"><div class="feature-icon">🛠️</div><div class="feature-title">Real tools, real numbers</div>'
                '<div class="feature-text">Answers come from Pandas calculations, not guesses.</div></div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="feature"><div class="feature-icon">📈</div><div class="feature-title">Charts on demand</div>'
                '<div class="feature-text">Bar, line, scatter, histogram and heatmap in seconds.</div></div>',
                unsafe_allow_html=True)
    st.info("👈 Upload a CSV/Excel file in the sidebar, or click **Try the sample dataset** to start.")
    st.chat_input("Upload a dataset to start chatting...", disabled=True)
    st.stop()

# --------------------------------------------------------------------------- stats row
numeric_cols = df.select_dtypes(include="number").columns
cards = [
    ("🧾", f"{df.shape[0]:,}", "Rows"),
    ("🧱", f"{df.shape[1]}", "Columns"),
    ("🔢", f"{len(numeric_cols)}", "Numeric columns"),
    ("🕳️", f"{int(df.isna().sum().sum()):,}", "Missing cells"),
    ("♻️", f"{int(df.duplicated().sum()):,}", "Duplicate rows"),
]
for col, (icon, value, label) in zip(st.columns(len(cards)), cards):
    col.markdown(stat_card(icon, value, label), unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------- handle new requests
prompt = st.chat_input("Ask anything about your data...")
if st.session_state.pending:
    prompt, st.session_state.pending = st.session_state.pending, None

if st.session_state.want_insights:
    st.session_state.want_insights = False
    run_and_store("✨ Generate autonomous insights", get_auto_insights)
elif prompt:
    run_and_store(prompt, lambda: ask_agent(prompt, get_dataset_overview()))

# --------------------------------------------------------------------------- tabs
tab_chat, tab_data, tab_explore = st.tabs(["💬 Ask the agent", "📄 Data preview", "📈 Explore"])

with tab_chat:
    if not st.session_state.messages:
        st.markdown("#### Try one of these")
        cols = st.columns(3)
        for i, text in enumerate(SUGGESTIONS):
            with cols[i % 3]:
                st.button(text, key=f"sug_{i}", on_click=set_pending, args=(text,))
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("image"):
                st.image(msg["image"], caption="Chart generated by the agent")

with tab_data:
    st.markdown("#### Dataset")
    st.dataframe(df, height=380)
    st.markdown("#### Columns")
    info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(1).values,
        "Unique values": df.nunique().values,
    })
    st.dataframe(info, hide_index=True)

with tab_explore:
    st.caption("These charts are drawn locally, so they don't use any Gemini quota.")
    column = st.selectbox("Pick a column to visualise", list(df.columns))
    series = df[column]
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4))
    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna()
        if len(values):
            sns.histplot(values, kde=len(values) > 2, bins=20, color="#7c3aed", ax=ax)
            ax.set_title(f"Distribution of {column}")
        else:
            ax.text(0.5, 0.5, "No data", ha="center")
    else:
        counts = series.astype(str).value_counts().head(15)
        sns.barplot(x=counts.values, y=counts.index, color="#7c3aed", ax=ax)
        ax.set_title(f"Top values of {column}")
        ax.set_xlabel("Count")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if len(numeric_cols) >= 2 and st.checkbox("Show correlation heatmap"):
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=len(numeric_cols) <= 12,
                    fmt=".2f", cmap="coolwarm", ax=ax2)
        ax2.set_title("Correlation heatmap")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)