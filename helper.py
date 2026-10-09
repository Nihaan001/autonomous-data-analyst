import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

CURRENT_DF = None
LAST_FIG_PATH = "temp_chart.png"


def set_dataframe(df: pd.DataFrame):
    global CURRENT_DF
    CURRENT_DF = df


def get_dataframe():
    return CURRENT_DF


def load_dataset(uploaded_file, filename: str = None):
    name = filename or getattr(uploaded_file, "name", "uploaded.csv")
    name = str(name)
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")
    set_dataframe(df)
    return df


def get_dataset_overview() -> str:
    df = get_dataframe()
    if df is None:
        return "No dataset loaded."
    buffer = io.StringIO()
    df.info(buf=buffer)
    return f"""
Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
Columns: {list(df.columns)}
Data Types:
{df.dtypes.to_string()}
Missing Values:
{df.isnull().sum().to_string()}
Duplicate Rows: {df.duplicated().sum()}
First 5 Rows:
{df.head().to_string()}
Info:
{buffer.getvalue()}
"""


@tool
def analyze_dataset() -> str:
    """Overall dataset overview: rows, columns, dtypes, head, missing values, duplicates."""
    return get_dataset_overview()


@tool
def find_missing_values() -> str:
    """Missing values per column with count and percentage."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return "No missing values found. Dataset is complete."
    percent = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"Missing Count": missing, "Percentage": percent})
    return result.to_string()


@tool
def find_duplicates() -> str:
    """Count duplicate rows."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    count = int(df.duplicated().sum())
    percent = round(count / len(df) * 100, 2) if len(df) else 0
    return f"Duplicate Rows: {count} ({percent}%). Total Rows: {len(df)}"


@tool
def calculate_statistics() -> str:
    """Descriptive statistics for numeric columns (mean, median, std, min, max)."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    try:
        desc = df.describe(include="all").to_string()
        extra = ""
        for col in df.select_dtypes(include=np.number).columns:
            extra += (
                f"\n{col}: Mean={df[col].mean():.2f}, Median={df[col].median():.2f}, "
                f"Std={df[col].std():.2f}, Min={df[col].min()}, Max={df[col].max()}"
            )
        return f"Descriptive Statistics:\n{desc}\n\nDetailed Numeric Summary:{extra}"
    except Exception as e:
        return f"Error calculating stats: {e}"


@tool
def calculate_correlation() -> str:
    """Correlation matrix for numeric columns."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns for correlation (need at least 2)."
    return f"Correlation Matrix:\n{numeric_df.corr().round(2).to_string()}"


@tool
def groupby_analysis(group_column: str, agg_column: str, agg_func: str = "mean") -> str:
    """Group by a column and aggregate another. agg_func: sum, mean, median, count, max, min, std."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    try:
        if group_column not in df.columns or agg_column not in df.columns:
            return f"Columns not found. Available: {list(df.columns)}"
        allowed = ["sum", "mean", "median", "count", "max", "min", "std"]
        if agg_func not in allowed:
            return f"agg_func must be one of {allowed}"
        result = df.groupby(group_column)[agg_column].agg(agg_func).sort_values(ascending=False).round(2)
        return f"Groupby Result ({agg_func} of {agg_column} by {group_column}):\n{result.to_string()}"
    except Exception as e:
        return f"Error in groupby: {e}"


@tool
def filter_data(query: str) -> str:
    """Filter with pandas query, e.g. 'Sales > 500' or \"Category == 'Electronics'\"."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    try:
        filtered = df.query(query)
        return f"Filtered {len(filtered)} rows with query '{query}':\n{filtered.head(10).to_string()}"
    except Exception as e:
        return f"Error filtering: {e}. Available columns: {list(df.columns)}"


@tool
def generate_chart(chart_type: str, x_column: str, y_column: str = None) -> str:
    """Create a chart. chart_type: bar, line, histogram, scatter, heatmap."""
    df = get_dataframe()
    if df is None:
        return "No dataset loaded"
    try:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        if chart_type == "bar":
            if x_column not in df.columns or y_column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"
            grouped = df.groupby(x_column)[y_column].sum().sort_values(ascending=False).head(10)
            sns.barplot(x=grouped.index.astype(str), y=grouped.values,
                        hue=grouped.index.astype(str), palette="viridis", legend=False)
            plt.title(f"{y_column} by {x_column} (Top 10)")
            plt.xticks(rotation=45)

        elif chart_type == "line":
            if x_column not in df.columns or y_column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"
            plt.plot(df[x_column], df[y_column], marker="o")
            plt.title(f"{y_column} over {x_column}")
            plt.xticks(rotation=45)

        elif chart_type == "histogram":
            if x_column not in df.columns:
                return f"Column not found: {x_column}"
            sns.histplot(df[x_column].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of {x_column}")

        elif chart_type == "scatter":
            if x_column not in df.columns or y_column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"
            sns.scatterplot(data=df, x=x_column, y=y_column)
            plt.title(f"{y_column} vs {x_column}")

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                return "Not enough numeric columns for heatmap"
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Heatmap")

        else:
            return "Invalid chart_type. Use: bar, line, histogram, scatter, heatmap"

        plt.tight_layout()
        plt.savefig(LAST_FIG_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        return (
            f"SUCCESS: {chart_type} chart generated for {x_column} vs {y_column} "
            f"and saved to {LAST_FIG_PATH}. Tell the user the chart is ready."
        )
    except Exception as e:
        plt.close()
        return f"Error generating chart: {e}"


ALL_TOOLS = [
    analyze_dataset,
    find_missing_values,
    find_duplicates,
    calculate_statistics,
    calculate_correlation,
    groupby_analysis,
    filter_data,
    generate_chart,
]
