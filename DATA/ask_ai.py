import pandas as pd
import ollama

# Load the dataset
df = pd.read_csv("sales.csv")

# Get dataset information
columns = list(df.columns)
sample = df.head(5).to_dict()

print("Dataset loaded successfully!")
print("Columns:", columns)


def ask_ai(question):

    prompt = f"""
You are a Pandas expert.

You are given a Pandas DataFrame called df.

Columns:
{columns}

Sample data:
{sample}

Convert the user's question into ONE valid Pandas expression.

User question:
{question}

Rules:
- Use only the DataFrame called df.
- Return ONLY one valid Pandas expression.
- Do not use markdown.
- Do not explain the answer.

Examples:

Question: What is the total sales?
Code: df['Sales'].sum()

Question: What is the average sales?
Code: df['Sales'].mean()

Question: How many unique products are there?
Code: df['Product'].nunique()

Question: Which region has the highest sales?
Code: df.groupby('Region')['Sales'].sum().idxmax()

Question: Which region has the lowest sales?
Code: df.groupby('Region')['Sales'].sum().idxmin()

Question: What is the total profit by region?
Code: df.groupby('Region')['Profit'].sum()

Question: What is the total sales by region?
Code: df.groupby('Region')['Sales'].sum()

Important:
- When finding highest or lowest sales by region, ALWAYS group Region and calculate the SUM of Sales.
- Do not group the Region column itself.
- When calculating profit by region, group by Region and calculate the SUM of Profit.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


while True:

    question = input("\nAsk a question about your data (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("\nGoodbye! 👋")
        break

    code = ask_ai(question)

    print("\nGenerated Pandas code:")
    print(code)

    try:
        allowed_builtins = {
            "__builtins__": {}
        }

        result = eval(code, allowed_builtins, {"df": df})

        print("\nAnswer:")

        if isinstance(result, pd.DataFrame):
            print(result.to_string(index=False))

        elif isinstance(result, pd.Series):
            for index, value in result.items():
                if isinstance(value, (int, float)):
                    print(f"{index}: {value:,.0f}")
                else:
                    print(f"{index}: {value}")

        elif isinstance(result, (int, float)):
            if isinstance(result, float) and not result.is_integer():
                print(f"{result:,.2f}")
            else:
                print(f"{result:,.0f}")

        else:
            print(result)

    except Exception as e:
        print("\nError while executing code:")
        print(e)