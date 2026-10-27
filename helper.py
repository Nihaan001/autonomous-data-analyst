import io
import os

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_core.tools import tool


# Existing compatibility variables.
# Keep these names because other project files may use them.
CURRENT_DF = None
LAST_FIG_PATH = "temp_chart.png"


def set_dataframe(df: pd.DataFrame):
    """Set the current dataset."""
    global CURRENT_DF

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    CURRENT_DF = df


def get_dataframe():
    """Get the currently loaded dataset."""
    return CURRENT_DF


def load_dataset(uploaded_file, filename: str = None):
    """Load CSV, XLSX, or XLS files safely."""
    name = str(
        filename or getattr(uploaded_file, "name", "uploaded.csv")
    )
    extension = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if extension not in {"csv", "xlsx", "xls"}:
        raise ValueError(
            "Unsupported file format. Please use CSV, XLSX, or XLS."
        )

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif extension == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            df = pd.read_excel(uploaded_file, engine="xlrd")

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        if len(df.columns) == 0:
            raise ValueError("The uploaded file has no columns.")

        # Convert numeric-looking text columns to numbers.
        # Handles values such as "1200", "1,200", "45.5", and "30 min".
        for col in df.select_dtypes(include=["object", "string"]).columns:
            original = df[col]
            text_values = original.astype("string").str.strip()

            cleaned = (
                text_values
                .str.replace(",", "", regex=False)
                .str.extract(
                    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:min)?\s*$",
                    expand=False,
                )
            )

            numeric = pd.to_numeric(cleaned, errors="coerce")
            filled_count = int(original.notna().sum())

            # Convert only when at least half of the non-missing
            # values can be interpreted as numbers.
            if (
                filled_count > 0
                and numeric.notna().sum() >= 0.5 * filled_count
            ):
                df[col] = numeric

        set_dataframe(df)
        return df

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read the {extension.upper()} file. "
            "Check that it is valid and not corrupted."
        ) from exc


def get_dataset_overview() -> str:
    """Return a summary of the current dataset."""
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
        return "No dataset loaded."

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        return "No missing values found. Dataset is complete."

    percent = (missing / len(df) * 100).round(2)

    result = pd.DataFrame({
        "Missing Count": missing,
        "Percentage": percent,
    })

    return result.to_string()


@tool
def find_duplicates() -> str:
    """Count duplicate rows."""
    df = get_dataframe()

    if df is None:
        return "No dataset loaded."

    count = int(df.duplicated().sum())
    percent = round(count / len(df) * 100, 2) if len(df) else 0

    return (
        f"Duplicate Rows: {count} ({percent}%). "
        f"Total Rows: {len(df)}"
    )


@tool
def calculate_statistics() -> str:
    """Descriptive statistics for numeric and categorical columns."""
    df = get_dataframe()

    if df is None:
        return "No dataset loaded."

    try:
        desc = df.describe(include="all").to_string()
        extra = ""

        for col in df.select_dtypes(include=np.number).columns:
            series = df[col]

            extra += (
                f"\n{col}: Mean={series.mean():.2f}, "
                f"Median={series.median():.2f}, "
                f"Std={series.std():.2f}, "
                f"Min={series.min()}, Max={series.max()}"
            )

        return (
            f"Descriptive Statistics:\n{desc}\n\n"
            f"Detailed Numeric Summary:{extra}"
        )

    except Exception as exc:
        return f"Error calculating statistics: {exc}"


@tool
def calculate_correlation() -> str:
    """Correlation matrix for numeric columns."""
    df = get_dataframe()

    if df is None:
        return "No dataset loaded."

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns for correlation (need at least 2)."

    return (
        "Correlation Matrix:\n"
        f"{numeric_df.corr().round(2).to_string()}"
    )


@tool
def groupby_analysis(
    group_column: str,
    agg_column: str,
    agg_func: str = "mean",
) -> str:
    """Group by a column and aggregate another. Functions: sum, mean, median, count, max, min, std."""
    df = get_dataframe()

    if df is None:
        return "No dataset loaded."

    try:
        if group_column not in df.columns or agg_column not in df.columns:
            return f"Columns not found. Available: {list(df.columns)}"

        allowed = [
            "sum", "mean", "median", "count", "max", "min", "std"
        ]

        if agg_func not in allowed:
            return f"agg_func must be one of {allowed}"

        result = (
            df.groupby(group_column)[agg_column]
            .agg(agg_func)
            .sort_values(ascending=False)
            .round(2)
        )

        return (
            f"Groupby Result ({agg_func} of {agg_column} "
            f"by {group_column}):\n{result.to_string()}"
        )

    except Exception as exc:
        return f"Error in groupby: {exc}"


@tool
def filter_data(query: str) -> str:
    """Filter data using a pandas query, e.g. Sales > 500."""
    df = get_dataframe()

    if df is None:
        return "No dataset loaded."

    try:
        filtered = df.query(query)

        return (
            f"Filtered {len(filtered)} rows with query '{query}':\n"
            f"{filtered.head(10).to_string()}"
        )

    except Exception as exc:
        return f"Error filtering: {exc}. Available columns: {list(df.columns)}"


@tool
def generate_chart(
    chart_type: str,
    x_column: str,
    y_column: str = None,
) -> str:
    """Create a bar, line, histogram, scatter, or heatmap chart."""
    df = get_dataframe()

    if df is None:
        return "No dataset loaded."

    fig = None

    try:
        chart_type = chart_type.lower().strip()

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            if x_column not in df.columns or y_column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"

            grouped = (
                df.groupby(x_column)[y_column]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            sns.barplot(
                x=grouped.index.astype(str),
                y=grouped.values,
                ax=ax,
                color="steelblue",
            )
            ax.set_title(f"{y_column} by {x_column} (Top 10)")
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "line":
            if x_column not in df.columns or y_column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"

            sns.lineplot(
                data=df,
                x=x_column,
                y=y_column,
                marker="o",
                ax=ax,
            )
            ax.set_title(f"{y_column} over {x_column}")
            ax.tick_params(axis="x", rotation=45)

        elif chart_type == "histogram":
            if x_column not in df.columns:
                return f"Column not found: {x_column}"

            sns.histplot(
                data=df,
                x=x_column,
                kde=False,
                bins=30,
                ax=ax,
            )
            ax.set_title(f"Distribution of {x_column}")

        elif chart_type == "scatter":
            if x_column not in df.columns or y_column not in df.columns:
                return f"Column not found. Available: {list(df.columns)}"

            sns.scatterplot(
                data=df,
                x=x_column,
                y=y_column,
                ax=ax,
            )
            ax.set_title(f"{y_column} vs {x_column}")

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=np.number)

            if numeric_df.shape[1] < 2:
                return "Not enough numeric columns for heatmap."

            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                ax=ax,
            )
            ax.set_title("Correlation Heatmap")

        else:
            return (
                "Invalid chart_type. Use: "
                "bar, line, histogram, scatter, heatmap"
            )

        fig.tight_layout()

        # Keep the existing output filename for compatibility.
        fig.savefig(LAST_FIG_PATH, dpi=150, bbox_inches="tight")

        return (
            f"SUCCESS: {chart_type} chart generated for "
            f"{x_column} vs {y_column} and saved to {LAST_FIG_PATH}. "
            "Tell the user the chart is ready."
        )

    except Exception as exc:
        return f"Error generating chart: {exc}"

    finally:
        if fig is not None:
            plt.close(fig)


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