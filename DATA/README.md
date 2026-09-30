# AI Data Analyst

AI Data Analyst is a Python-based application that allows users to analyze CSV data by asking questions in natural language.

The application uses a locally running **Llama 3.2** model through **Ollama**. The model converts the user's question into a Pandas expression, which is then processed through a controlled AST-based evaluator. The result is displayed in the Streamlit interface along with a visualization when applicable.

## Features

* Upload a CSV file for analysis
* View the uploaded dataset
* Ask questions using natural language
* Generate Pandas expressions using Llama 3.2
* Analyze data using Pandas
* Display DataFrame and Series results
* Generate basic visualizations
* Use AI locally without a paid API
* Use a restricted AST-based execution layer for generated expressions

## Technologies Used

* Python
* Pandas
* Streamlit
* Ollama
* Llama 3.2
* Python AST

## How the Application Works

```text
CSV Dataset
    ↓
User asks a question
    ↓
Llama 3.2
    ↓
Pandas expression
    ↓
Controlled AST evaluator
    ↓
Pandas analysis
    ↓
Result and visualization
```

## Example Questions

The application can answer questions such as:

* What is the total sales?
* What is the average sales?
* Which region has the highest sales?
* Which region has the lowest sales?
* Show the top 3 products by sales.
* What is the total profit by region?
* Show sales greater than 50000.

## Project Structure

```text
AI-DATA-CHATBOT/
│
└── DATA/
    ├── app.py
    ├── ask_ai.py
    ├── test_api.py
    ├── Sales.csv
    ├── requirements.txt
    ├── README.md
    └── .gitignore
```

## Requirements

* Python 3.x
* Ollama
* Llama 3.2 model

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project directory

```bash
cd AI-Data-Analyst
```

### 3. Install the required Python packages

```bash
pip install -r DATA/requirements.txt
```

### 4. Install and run Ollama

Download Ollama and pull the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before starting the application.

### 5. Start the Streamlit application

```bash
cd DATA
python -m streamlit run app.py
```

The application will then be available locally through Streamlit.

## Security

The application does not directly execute the generated response using Python `eval()`.

Instead, it parses the generated expression using Python's AST module and allows only selected operations and Pandas methods.

The `.env` file is also excluded from Git using `.gitignore`.

The AST evaluator is a restricted execution mechanism for this project and should not be considered a complete security sandbox for arbitrary untrusted code.

## Dataset

A sample `Sales.csv` dataset is included with the project.

The dataset contains information such as:

* Order ID
* Date
* Product
* Category
* Region
* Quantity
* Sales
* Profit

Users can also upload their own CSV file through the Streamlit interface.

## Purpose of the Project

I built this project to understand how Generative AI can be combined with data analysis.

Instead of writing Pandas code manually, a user can ask a question about a dataset in normal language. The local Llama 3.2 model converts the question into a Pandas expression, and the application processes the expression and displays the result.

This project helped me practice Python, Pandas, Streamlit, prompt design, local LLM integration, and controlled execution of generated code.

## Author

**Diksha Ramteke**

B.Tech Computer Science Engineering
