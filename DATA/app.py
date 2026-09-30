import ast
import operator

import streamlit as st
import pandas as pd
import ollama


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🤖 AI Data Analyst</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about your data using natural language</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# UPLOAD DATASET
# ============================================================

st.subheader("📁 Upload Your Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset '{uploaded_file.name}' uploaded successfully."
        )

    except Exception as e:

        st.error(
            f"Unable to read the uploaded CSV: {e}"
        )

        st.stop()

else:

    try:

        df = pd.read_csv("sales.csv")

        st.info(
            "No file uploaded. Using the default sales dataset."
        )

    except Exception as e:

        st.error(
            f"Unable to load sales.csv: {e}"
        )

        st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

columns = list(df.columns)

sample = df.head(5).to_dict()


# ============================================================
# AI FUNCTION
# ============================================================

def ask_ai(question):

    prompt = f"""
You are an expert Data Analyst and Pandas programmer.

You are given a Pandas DataFrame called df.

Dataset columns:
{columns}

Sample data:
{sample}

Convert the user's natural-language question into ONE valid
Pandas expression using only df.

User question:
{question}


IMPORTANT RULES:

1. Return ONLY the Pandas expression.
2. Do NOT use markdown.
3. Do NOT use ```.
4. Do NOT explain the code.
5. Use only the DataFrame called df.
6. Do not create variables.
7. Do not import anything.
8. Do not use Python functions such as eval, exec, open,
   compile, __import__, globals, locals, os, sys or subprocess.
9. Do not invent column names.
10. Use only columns that actually exist.
11. For total use .sum().
12. For average use .mean().
13. For minimum use .min().
14. For maximum use .max().
15. For unique values use .nunique().
16. For totals by category use groupby().sum().
17. For averages by category use groupby().mean().
18. For highest category use groupby().sum().idxmax().
19. For lowest category use groupby().sum().idxmin().
20. For sorting use sort_values().
21. For top results use sort_values().head().
22. For bottom results use sort_values().tail().
23. For filtering use boolean conditions.
24. Keep the expression simple.


EXAMPLES:


Question:
What is the total sales?

Code:
df['Sales'].sum()


Question:
What is the average sales?

Code:
df['Sales'].mean()


Question:
How many unique products are there?

Code:
df['Product'].nunique()


Question:
Which region has the highest sales?

Code:
df.groupby('Region')['Sales'].sum().idxmax()


Question:
Which region has the lowest sales?

Code:
df.groupby('Region')['Sales'].sum().idxmin()


Question:
What is the total sales by region?

Code:
df.groupby('Region')['Sales'].sum()


Question:
What is the total profit by region?

Code:
df.groupby('Region')['Profit'].sum()


Question:
Which product has the highest sales?

Code:
df.groupby('Product')['Sales'].sum().idxmax()


Question:
Show the top 3 products by sales.

Code:
df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(3)


Question:
Show the regions from highest sales to lowest sales.

Code:
df.groupby('Region')['Sales'].sum().sort_values(ascending=False)


Question:
Show sales greater than 50000.

Code:
df[df['Sales'] > 50000]


Return ONLY ONE Pandas expression.
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


# ============================================================
# ALLOWED METHODS
# ============================================================

ALLOWED_METHODS = {

    "sum",
    "mean",
    "min",
    "max",
    "nunique",
    "count",
    "groupby",
    "sort_values",
    "head",
    "tail",
    "idxmax",
    "idxmin",
    "reset_index",
    "value_counts"

}


# ============================================================
# ALLOWED ATTRIBUTES
# ============================================================

ALLOWED_ATTRIBUTES = {

    "shape",
    "columns"

}


# ============================================================
# ALLOWED OPERATORS
# ============================================================

BINARY_OPERATORS = {

    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod

}


COMPARISON_OPERATORS = {

    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne

}


# ============================================================
# CONTROLLED AST EXECUTOR
# ============================================================

def evaluate_node(node, dataframe):


    # --------------------------------------------------------
    # DataFrame name
    # --------------------------------------------------------

    if isinstance(node, ast.Name):

        if node.id == "df":

            return dataframe

        raise ValueError(
            f"Variable '{node.id}' is not allowed."
        )


    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    if isinstance(node, ast.Constant):

        if isinstance(
            node.value,
            (
                str,
                int,
                float,
                bool,
                type(None)
            )
        ):

            return node.value

        raise ValueError(
            "This constant type is not allowed."
        )


    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    if isinstance(node, ast.List):

        return [
            evaluate_node(
                element,
                dataframe
            )
            for element in node.elts
        ]


    # --------------------------------------------------------
    # Tuples
    # --------------------------------------------------------

    if isinstance(node, ast.Tuple):

        return tuple(
            evaluate_node(
                element,
                dataframe
            )
            for element in node.elts
        )


    # --------------------------------------------------------
    # Dictionaries
    # --------------------------------------------------------

    if isinstance(node, ast.Dict):

        return {

            evaluate_node(
                key,
                dataframe
            ):

            evaluate_node(
                value,
                dataframe
            )

            for key, value
            in zip(
                node.keys,
                node.values
            )

        }


    # --------------------------------------------------------
    # Subscript
    # --------------------------------------------------------

    if isinstance(node, ast.Subscript):

        obj = evaluate_node(
            node.value,
            dataframe
        )

        key = evaluate_node(
            node.slice,
            dataframe
        )

        try:

            return obj[key]

        except Exception as e:

            raise ValueError(
                f"Invalid DataFrame operation: {e}"
            )


    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    if isinstance(node, ast.Attribute):

        obj = evaluate_node(
            node.value,
            dataframe
        )

        attribute = node.attr

        if attribute not in ALLOWED_ATTRIBUTES:

            raise ValueError(
                f"Attribute '{attribute}' is not allowed."
            )

        return getattr(
            obj,
            attribute
        )


    # --------------------------------------------------------
    # Function / Method Calls
    # --------------------------------------------------------

    if isinstance(node, ast.Call):

        if not isinstance(
            node.func,
            ast.Attribute
        ):

            raise ValueError(
                "Only approved Pandas methods are allowed."
            )


        method_name = node.func.attr


        if method_name not in ALLOWED_METHODS:

            raise ValueError(
                f"Pandas method '{method_name}' is not allowed."
            )


        obj = evaluate_node(
            node.func.value,
            dataframe
        )


        args = [

            evaluate_node(
                arg,
                dataframe
            )

            for arg in node.args

        ]


        kwargs = {}


        for keyword in node.keywords:

            if keyword.arg is None:

                raise ValueError(
                    "Complex keyword arguments are not allowed."
                )


            kwargs[keyword.arg] = evaluate_node(
                keyword.value,
                dataframe
            )


        try:

            method = getattr(
                obj,
                method_name
            )

            return method(
                *args,
                **kwargs
            )

        except Exception as e:

            raise ValueError(
                f"Unable to execute '{method_name}': {e}"
            )


    # --------------------------------------------------------
    # Binary Operations
    # --------------------------------------------------------

    if isinstance(node, ast.BinOp):

        if type(node.op) not in BINARY_OPERATORS:

            raise ValueError(
                "This mathematical operation is not allowed."
            )


        left = evaluate_node(
            node.left,
            dataframe
        )


        right = evaluate_node(
            node.right,
            dataframe
        )


        return BINARY_OPERATORS[
            type(node.op)
        ](
            left,
            right
        )


    # --------------------------------------------------------
    # Unary Operations
    # --------------------------------------------------------

    if isinstance(node, ast.UnaryOp):

        operand = evaluate_node(
            node.operand,
            dataframe
        )


        if isinstance(
            node.op,
            ast.USub
        ):

            return -operand


        if isinstance(
            node.op,
            ast.UAdd
        ):

            return +operand


        if isinstance(
            node.op,
            ast.Not
        ):

            return not operand


        raise ValueError(
            "This unary operation is not allowed."
        )


    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------

    if isinstance(node, ast.Compare):

        left = evaluate_node(
            node.left,
            dataframe
        )

        result = None


        for operator_node, comparator_node in zip(
            node.ops,
            node.comparators
        ):

            if type(operator_node) not in COMPARISON_OPERATORS:

                raise ValueError(
                    "This comparison operation is not allowed."
                )


            right = evaluate_node(
                comparator_node,
                dataframe
            )


            comparison = COMPARISON_OPERATORS[
                type(operator_node)
            ](
                left,
                right
            )


            if result is None:

                result = comparison

            else:

                result = result & comparison


            left = right


        return result


    # --------------------------------------------------------
    # Boolean Operations
    # --------------------------------------------------------

    if isinstance(node, ast.BoolOp):

        values = [

            evaluate_node(
                value,
                dataframe
            )

            for value in node.values

        ]


        if isinstance(
            node.op,
            ast.And
        ):

            result = values[0]

            for value in values[1:]:

                result = result & value

            return result


        if isinstance(
            node.op,
            ast.Or
        ):

            result = values[0]

            for value in values[1:]:

                result = result | value

            return result


    # --------------------------------------------------------
    # Block everything else
    # --------------------------------------------------------

    raise ValueError(
        f"Expression type '{type(node).__name__}' is not allowed."
    )


# ============================================================
# CONTROLLED EXECUTION FUNCTION
# ============================================================

def controlled_execute(
    expression,
    dataframe
):

    expression = expression.strip()


    # Remove markdown code blocks
    if expression.startswith("```"):

        expression = expression.replace(
            "```python",
            ""
        )

        expression = expression.replace(
            "```",
            ""
        )

        expression = expression.strip()


    try:

        tree = ast.parse(
            expression,
            mode="eval"
        )

    except SyntaxError as e:

        raise ValueError(
            f"Invalid Pandas expression: {e}"
        )


    return evaluate_node(
        tree.body,
        dataframe
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Rows",
        df.shape[0]
    )


with col2:

    st.metric(
        "Total Columns",
        df.shape[1]
    )


with col3:

    if "Sales" in df.columns:

        try:

            total_sales = pd.to_numeric(
                df["Sales"],
                errors="coerce"
            ).sum()


            st.metric(
                "Total Sales",
                f"₹{total_sales:,.0f}"
            )

        except Exception:

            st.metric(
                "Total Sales",
                "N/A"
            )

    else:

        st.metric(
            "Total Sales",
            "N/A"
        )


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander("🔎 View Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )


st.divider()


# ============================================================
# ASK YOUR DATA
# ============================================================

st.subheader("💬 Ask Your Data")

st.caption(
    "Try one of these questions:"
)


example_questions = [

    "Select a question",

    "What is the total sales?",

    "Which region has the highest sales?",

    "Show the top 3 products by sales.",

    "What is the total profit by region?",

    "Show sales greater than 50000."

]


selected_question = st.selectbox(

    "Example questions",

    example_questions,

    key="example_question_select"

)


question = st.text_input(

    "Enter your question",

    placeholder="Example: What is the average sales?",

    key="user_question"

)


if (
    selected_question != "Select a question"
    and not question
):

    question = selected_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    with st.spinner(
        "🤖 AI is analyzing your data..."
    ):

        try:

            # ------------------------------------------------
            # Generate Pandas code
            # ------------------------------------------------

            code = ask_ai(
                question
            )


            st.subheader(
                "🐼 Generated Pandas Code"
            )


            st.code(
                code,
                language="python"
            )


            # ------------------------------------------------
            # Execute controlled expression
            # ------------------------------------------------

            result = controlled_execute(
                code,
                df
            )


            st.subheader(
                "✅ Answer"
            )


            # =================================================
            # DATAFRAME RESULT
            # =================================================

            if isinstance(
                result,
                pd.DataFrame
            ):

                st.dataframe(
                    result,
                    use_container_width=True
                )


                numeric_columns = result.select_dtypes(
                    include="number"
                ).columns.tolist()


                if len(numeric_columns) > 0:

                    st.subheader(
                        "📊 Visualization"
                    )


                    if len(result.columns) >= 2:

                        chart_column = st.selectbox(

                            "Select a numeric column to visualize",

                            numeric_columns,

                            key="chart_column_select"

                        )


                        chart_data = result.set_index(
                            result.columns[0]
                        )[chart_column]


                        st.bar_chart(
                            chart_data
                        )


            # =================================================
            # SERIES RESULT
            # =================================================

            elif isinstance(
                result,
                pd.Series
            ):

                result_df = result.reset_index()


                st.dataframe(
                    result_df,
                    use_container_width=True
                )


                if pd.api.types.is_numeric_dtype(
                    result
                ):

                    st.subheader(
                        "📊 Visualization"
                    )


                    st.bar_chart(
                        result
                    )


            # =================================================
            # SIMPLE RESULT
            # =================================================

            else:

                st.success(
                    str(result)
                )


            # ------------------------------------------------
            # ANALYSIS NOTE
            # ------------------------------------------------

            st.info(
                "💡 This result was generated by analyzing "
                "your uploaded dataset using Pandas."
            )


        except Exception as e:

            st.error(
                f"Unable to process the question: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • Pandas • Llama 3.2 • Ollama • Streamlit"
)