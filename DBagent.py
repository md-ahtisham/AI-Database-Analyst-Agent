
import streamlit as st
import sqlite3
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.chains import LLMMathChain
import google.generativeai as genai
import pygwalker as pyg
import tempfile
import os
import logging
import warnings
from typing import List, Tuple, Any
import streamlit.components.v1 as components

# Configuration
DB_PATH = r"sqlite:///C:/Users/acer/Desktop/AI DB analyst/titanic.db"
SQLITE_FILE_PATH = r"C:/Users/acer/Desktop/AI DB analyst/titanic.db"
ALLOWED_SQL_KEYWORDS = {'SELECT', 'FROM', 'WHERE', 'COUNT', 'SUM', 'AVG', 'JOIN', 'GROUP BY', 'ORDER BY', 'LIMIT'}
MAX_ITERATIONS = 10

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Streamlit page configuration
st.set_page_config(
    page_title="Database AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SecureSQLExecutor:
    """Safe SQL execution handler with query validation"""

    def __init__(self, db_uri: str):
        self.db = SQLDatabase.from_uri(db_uri)
        self.query_history = set()

    def validate_sql(self, query: str) -> Tuple[bool, str]:
        """Validate SQL query structure and keywords"""
        query = query.upper().strip()

        if any(word in query for word in {'DROP', 'DELETE', 'UPDATE', 'INSERT'}):
            return False, "DML operations forbidden"

        if not any(keyword in query for keyword in ALLOWED_SQL_KEYWORDS):
            return False, "Invalid SQL structure"

        return True, ""

    def execute(self, query: str) -> str:
        """Safe query execution with history tracking"""
        if query in self.query_history:
            return "Error: Repeated query detected"

        is_valid, msg = self.validate_sql(query)
        if not is_valid:
            return f"Validation Error: {msg}"

        try:
            result = self.db.run(query)
            self.query_history.add(query)
            return self._format_result(result)
        except Exception as e:
            return f"Execution Error: {str(e)}"

    def _format_result(self, raw_result: str) -> str:
        """Convert raw SQL output to natural language"""
        try:
            if "COUNT" in raw_result:
                count = int(raw_result.strip("[]()").split(",")[0])
                return f"Result: {count}"

            if "SUM" in raw_result or "AVG" in raw_result:
                value = float(raw_result.strip("[]()").split(",")[0])
                return f"Result: {value:.2f}"

            return f"Data: {raw_result}"
        except:
            return raw_result

class VisualizationTool:
    """Data visualization handler with automatic CSV conversion"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.csv_path = "combined_database.csv"

    def export_and_visualize(self, _):
        """Handles CSV conversion and visualization in one step"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]

            dfs = []
            for table in tables:
                df = pd.read_sql(f"SELECT * FROM {table};", conn)
                df["Table_Source"] = table
                dfs.append(df)

            combined_df = pd.concat(dfs)
            combined_df.to_csv(self.csv_path, index=False)
            conn.close()

            # Generate Pygwalker visualization
            vis_df = pd.read_csv(self.csv_path)
            pyg_html = pyg.to_html(vis_df)

            return f"Visualization generated successfully! Data exported to {self.csv_path}"

        except Exception as e:
            return f"Visualization failed: {str(e)}"

def get_agent_for_dbfile(dbfile_path=None, api_key=None):
    """Returns a new agent instance for the given db file path"""

    if not api_key:
        return None

    # Configure Google API
    genai.configure(api_key=api_key)

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0.4,
        convert_system_message_to_human=True
    )

    db_uri = f"sqlite:///{dbfile_path}" if dbfile_path else DB_PATH
    sql_executor = SecureSQLExecutor(db_uri)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
    visualization_tool = VisualizationTool(dbfile_path or SQLITE_FILE_PATH)

    # Define tools
    tools = [
        Tool(
            name="MathSolver",
            func=llm_math_chain.run,
            description="Use for mathematical calculations in natural language question as input. For example, What is 10 plus 5?"
            "only capable of do basic Arithmetic operations, Exponents and logarithms, Trigonometry, Calculus only so take action accordingly "
        ),
        Tool(
            name="MusicDB",
            func=sql_executor.execute,
            description=(
                "Use for music database queries. Input MUST be a valid SQL SELECT query."
                " specily Ensure while Entering new SQLDatabaseChain chain... the SQLQuery does not include Markdown syntax like ```sqlite...``` or other formatting that cause OperationalError. Simply write raw SQL."
                "Overview of the tatanic Database:The titanic database is a sample database that represents information of passenger of tatanic, including entities such as 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'. Here's how the data is organized and structured:"
                "'titanic': 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked' "
                "Always use exact column names. Never use JOINs unless absolutely necessary."
            )
        ),
        Tool(
            name="DataVisualizer",
            func=visualization_tool.export_and_visualize,
            description="Generates interactive visualizations. Input should describe desired chart type and data to visualize."
        )
    ]

    # Custom agent prompt template
    CUSTOM_PROMPT = """You are a professional database analyst. Follow these steps:

    1. Analyze the question carefully
    2. Generate EXACT SQL query for MusicDB tool when needed
    3. Use visualizations when asked for charts/graphs
    4. Return final answer in natural language

    Rules:
    - NEVER invent table/column names
    - Use only provided tools
    - Format SQL queries as plain text without markdown
    - Handle numbers with MathSolver when needed
    - Use for mathematical calculations in natural language question as input in MathSolver . For example, What is 10 plus 5?, only capable of do basic Arithmetic operations, Exponents and logarithms, Trigonometry, Calculus not cable of of rounding or anything so take action accordingly.

    Question: {input}

    {agent_scratchpad}"""

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": CUSTOM_PROMPT,
            "format_instructions": """Use this EXACT format:

            Thought: [your analysis]
            Action: [tool name]
            Action Input: [input]
            Observation: [result]
            ...repeat if needed...
            Final Answer: [final response]""",
            "stop": ["\nObservation:"]
        }
    )

    return agent

def main():
    """Main Streamlit application"""

    # Title and description
    st.title("🤖 Database AI Agent")
    st.markdown("Ask natural language questions about your database and get AI-powered answers with visualizations!")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google Generative AI API key"
    )

    # Database file uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload SQLite Database (optional)",
        type=["db", "sqlite", "sqlite3"],
        help="Upload a custom SQLite database file, or use the default Titanic dataset"
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask Your Question")

        # Question input
        question = st.text_area(
            "Enter your question about the database:",
            height=100,
            placeholder="e.g., How many passengers survived? or Show me a visualization of passenger ages"
        )

        # Submit button
        if st.button("🔍 Get Answer", type="primary"):
            if not api_key:
                st.error("Please enter your Google API key in the sidebar.")
                return

            if not question.strip():
                st.error("Please enter a question.")
                return

            # Handle file upload
            dbfile_path = None
            temp_file = None

            try:
                if uploaded_file is not None:
                    # Save uploaded file to temporary location
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
                    temp_file.write(uploaded_file.getvalue())
                    temp_file.close()
                    dbfile_path = temp_file.name
                    st.sidebar.success(f"Using uploaded database: {uploaded_file.name}")
                else:
                    st.sidebar.info("Using default Titanic database")

                # Get agent and process question
                with st.spinner("Processing your question..."):
                    agent = get_agent_for_dbfile(dbfile_path, api_key)

                    if agent is None:
                        st.error("Failed to initialize the AI agent. Please check your API key.")
                        return

                    # Execute query
                    answer = agent.run(question)

                    # Display answer
                    st.subheader("🎯 Answer")
                    st.write(answer)

                    # Check if visualization was requested
                    if "visualization" in question.lower() or "chart" in question.lower() or "graph" in question.lower():
                        st.subheader("📊 Visualization")

                        # Try to load and display the generated CSV
                        try:
                            if os.path.exists("combined_database.csv"):
                                df = pd.read_csv("combined_database.csv")
                                st.dataframe(df.head(10))

                                # Generate Pygwalker visualization
                                st.subheader("🎨 Interactive Visualization")
                                pyg_html = pyg.to_html(df)
                                components.html(pyg_html, height=600, scrolling=True)

                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

    with col2:
        st.subheader("ℹ️ Information")

        st.markdown("""
        **Database Schema (Titanic):**
        - PassengerId
        - Survived
        - Pclass
        - Name
        - Sex
        - Age
        - SibSp
        - Parch
        - Ticket
        - Fare
        - Cabin
        - Embarked
        """)

        st.markdown("""
        **Example Questions:**
        - How many passengers survived?
        - What was the average age of passengers?
        - Show me a visualization of passenger classes
        - Calculate the survival rate by gender
        """)

        st.markdown("""
        **Features:**
        - 🔍 Natural language SQL queries
        - 🧮 Mathematical calculations
        - 📊 Interactive visualizations
        - 🛡️ Secure query validation
        - 📁 Custom database upload
        """)

if __name__ == "__main__":
    main()
