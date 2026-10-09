# Autonomous Data Analyst Agent (Gemini)

Agentic AI app: upload CSV/Excel -> LangGraph agent picks tools ->
Pandas/NumPy calculate -> Matplotlib/Seaborn charts -> Gemini explains insights.

## Structure

    app/helper.py      Tools (Pandas / NumPy / charts)
    app/orchester.py   LangGraph + Gemini agent brain
    app/main.py        Streamlit UI + FastAPI backend
    data/sample.csv    Demo dataset

## Setup

    python -m venv venv
    venv\Scripts\activate          # Windows
    # source venv/bin/activate     # Mac/Linux
    pip install -r requirements.txt

Put your Gemini key (from https://aistudio.google.com/apikey) into .env

## Run

    streamlit run app/main.py

Optional API:

    uvicorn app.main:app --reload --port 8000

## Try asking

- Which category has highest sales?
- What is average revenue?
- Show missing values
- Plot correlation heatmap
- Bar chart of Sales by Category

If you get a model-not-found error, set GEMINI_MODEL=gemini-1.5-flash in .env
