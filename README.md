# 📊 Autonomous Data Analyst

An AI-powered data analysis application that allows users to upload datasets, ask questions in natural language, perform data analysis using Python tools, and generate visualizations.

The application combines **Google Gemini, LangGraph, Streamlit, Pandas, and Matplotlib** to simplify data analysis without requiring users to write SQL queries or Python code themselves.

## ✨ Features

- **Dataset Upload:** Upload CSV, XLSX, and XLS datasets.
- **AI-Powered Analysis:** Ask questions about your data in plain English.
- **Dataset Overview:** View dataset dimensions, columns, data types, missing values, and duplicate rows.
- **Descriptive Statistics:** Calculate mean, median, standard deviation, minimum, and maximum values.
- **Missing Value Detection:** Identify missing values and their percentages.
- **Duplicate Detection:** Find duplicate rows in the dataset.
- **Correlation Analysis:** Examine relationships between numeric columns.
- **Data Filtering:** Filter records using supported Pandas query expressions.
- **GroupBy Analysis:** Aggregate data using sum, mean, median, count, maximum, minimum, and standard deviation.
- **Data Visualization:** Generate bar charts, line charts, histograms, scatter plots, and correlation heatmaps.
- **Automated Insights:** Generate AI-assisted summaries and insights from datasets.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Google Gemini API | AI-powered responses |
| LangGraph | Agent workflow orchestration |
| LangChain Core | Tool integration |
| Pandas | Data processing and analysis |
| NumPy | Numerical computations |
| Matplotlib | Chart generation |
| Seaborn | Statistical visualizations |
| Git & GitHub | Version control and source code management |

## 📁 Project Structure

```text
autonomous-data-analyst/
│
├── data/
│   └── sample.csv
│
├── streamlit_app.py
├── helper.py
├── orchester.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> The exact structure may vary depending on your local project files.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Nihaan001/autonomous-data-analyst.git
```

Navigate to the project directory:

```bash
cd autonomous-data-analyst
```

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

### 3. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If you are using Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

After activation, your terminal should display `(venv)`.

### 4. Install Dependencies

Make sure `requirements.txt` exists in the project root.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory.

Copy the following template into `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Replace the placeholder with your own valid Google Gemini API key.

**Security note:** Never commit your actual `.env` file or API key to GitHub. Keep `.env` listed in `.gitignore` and share only `.env.example` with placeholder values.

### 6. Run