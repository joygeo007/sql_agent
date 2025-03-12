#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQL Query Agent CLI
A command-line interface for interacting with the SQL Query Agent.
"""

import argparse
import os
import sys
import json
import hashlib
import requests
from typing import Dict, List, Optional, Union
import readline  # Enable command history and editing

# Import necessary packages
try:
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph
    import pandas as pd
except ImportError:
    print("Required packages not found. Installing dependencies...")
    os.system("pip install langgraph langchain_community langchain_openai pandas")
    # Re-import after installation
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph
    import pandas as pd

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color):
    """Print text with color"""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text):
    """Print a formatted header"""
    width = min(80, max(len(text) + 4, 50))
    print("\n" + "=" * width)
    print_colored(f"  {text}", Colors.BOLD + Colors.HEADER)
    print("=" * width + "\n")

def print_error(text):
    """Print an error message"""
    print_colored(f"ERROR: {text}", Colors.RED)

def print_success(text):
    """Print a success message"""
    print_colored(f"SUCCESS: {text}", Colors.GREEN)

def print_info(text):
    """Print an info message"""
    print_colored(f"INFO: {text}", Colors.BLUE)

def print_warning(text):
    """Print a warning message"""
    print_colored(f"WARNING: {text}", Colors.YELLOW)

def download_sample_db(db_path: str = "Chinook.db") -> bool:
    """Download the sample Chinook database if it doesn't exist."""
    if os.path.exists(db_path):
        print_info(f"Using existing database at {db_path}")
        return True
    
    print_info("Downloading sample database...")
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    
    try:
        response = requests.get(db_url)
        if response.status_code == 200:
            with open(db_path, "wb") as file:
                file.write(response.content)
            print_success(f"Database downloaded successfully as {db_path}")
            return True
        else:
            print_error(f"Download failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error downloading database: {str(e)}")
        return False

def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or prompt user."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    try:
        api_key = input("Enter your OpenAI API key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            return api_key
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return None
    
    return None

def setup_db_connection(db_path: str) -> Optional[SQLDatabase]:
    """Set up database connection."""
    try:
        db_uri = f"sqlite:///{db_path}"
        db_connection = SQLDatabase.from_uri(db_uri)
        print_success(f"Connected to database with dialect: {db_connection.dialect}")
        print_info(f"Available tables: {', '.join(db_connection.get_usable_table_names())}")
        return db_connection
    except Exception as e:
        print_error(f"Failed to connect to database: {str(e)}")
        return None

def create_agent(db_connection: SQLDatabase, api_key: str) -> StateGraph:
    """Create and initialize the SQL agent."""
    from typing import Annotated, Literal
    from langchain_core.messages import AIMessage, ToolMessage
    from pydantic import BaseModel, Field
    from typing_extensions import TypedDict
    from langgraph.graph import END, StateGraph, START
    from langgraph.graph.message import AnyMessage, add_messages

    # Initialize query cache
    query_cache = {}

    # Utility functions
    def surface_tool_error(state) -> dict:
        """Handle and surface tool errors back to the agent."""
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error encountered: {repr(error)}\nPlease revise your approach.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def create_robust_tool_node(tools: list):
        """Create a tool node with error handling capabilities."""
        return ToolNode(tools).with_fallbacks(
            [lambda state: surface_tool_error(state)], 
            exception_key="error"
        )

    # State definition
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        cache_info: dict  # For tracking cache performance

    # Create tools
    toolkit = SQLDatabaseToolkit(
        db=db_connection, 
        llm=ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    )
    all_tools = toolkit.get_tools()

    # Extract specific tools
    table_listing_tool = next(tool for tool in all_tools if tool.name == "sql_db_list_tables")
    schema_tool = next(tool for tool in all_tools if tool.name == "sql_db_schema")

    # Create database query tool with caching
    @tool
    def database_query_tool(query: str) -> str:
        """
        Execute a SQL query against the database and return the results.
        Uses caching to improve performance for repeated queries.
        
        If the query has syntax errors, an error message will be returned.
        """
        # Check cache first
        cache_key = hashlib.md5(query.strip().lower().encode()).hexdigest()
        if cache_key in query_cache:
            print_info("Cache hit! Using cached results.")
            return query_cache[cache_key]
        
        # If not in cache, execute the query
        result = db_connection.run_no_throw(query)
        
        if not result:
            return "Error: Query execution failed. Please revise your SQL syntax."
        
        # Store successful query results in cache
        query_cache[cache_key] = result
        return result

    # Create result formatting tool
    @tool
    def format_query_results(query_results: str, format_type: str = "table") -> str:
        """Format query results in different presentation styles."""
        try:
            # Convert string results to DataFrame
            lines = query_results.strip().split('\n')
            if len(lines) <= 1:
                return "No results to format or empty result set."
                
            # Parse header and rows
            header = lines[0].split('|')
            header = [h.strip() for h in header if h.strip()]
            
            rows = []
            for line in lines[2:]:  # Skip header and separator line
                if line.strip():
                    cells = line.split('|')
                    cells = [c.strip() for c in cells if c.strip()]
                    if cells:
                        rows.append(cells)
            
            df = pd.DataFrame(rows, columns=header)
            
            # Return formatted results based on requested type
            if format_type == "json":
                return df.to_json(orient="records", indent=2)
            elif format_type == "csv":
                return df.to_csv(index=False)
            elif format_type == "markdown":
                return df.to_markdown(index=False)
            else:  # Default table format
                return query_results
                
        except Exception as e:
            return f"Error formatting results: {str(e)}\nReturning original results:\n{query_results}"

    # Define FinalResponse class
    class FinalResponse(BaseModel):
        """Submit the final answer based on database query results."""
        answer: str = Field(..., description="Comprehensive answer to the user's question")
        format_preference: str = Field(
            default="markdown", 
            description="Preferred format for displaying results (table, json, csv, markdown)"
        )

    # Initialize workflow graph
    workflow = StateGraph(AgentState)

    # Add query cache check node
    def check_query_cache(state: AgentState) -> dict:
        """Check if the current question has a cached response."""
        messages = state["messages"]
        user_question = next((m.content for m in messages if hasattr(m, 'content') and not hasattr(m, 'tool_calls')), None)
        
        # Initialize cache info if not present
        if "cache_info" not in state:
            state["cache_info"] = {"hits": 0, "misses": 0}
        
        cache_key = hashlib.md5(str(user_question).encode()).hexdigest()
        
        if cache_key in query_cache:
            state["cache_info"]["hits"] += 1
            return {
                "messages": [
                    AIMessage(content=f"Using cached results for similar query. Cache hit #{state['cache_info']['hits']}.")
                ]
            }
        else:
            state["cache_info"]["misses"] += 1
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sql_db_list_tables",
                                "args": {},
                                "id": "tool_tables_listing",
                            }
                        ],
                    )
                ]
            }

    # Add nodes to workflow
    workflow.add_node("check_query_cache", check_query_cache)
    workflow.add_node("list_tables_tool", create_robust_tool_node([table_listing_tool]))

    # Schema retrieval node
    schema_retriever = ChatOpenAI(
        model="gpt-4o", 
        api_key=api_key, 
        temperature=0
    ).bind_tools([schema_tool])

    workflow.add_node(
        "determine_relevant_tables",
        lambda state: {
            "messages": [schema_retriever.invoke(state["messages"])],
        },
    )

    workflow.add_node("get_schema_tool", create_robust_tool_node([schema_tool]))

    # Query generation node
    from langchain_core.prompts import ChatPromptTemplate

    query_generator_system = """You are an expert SQL database analyst who specializes in translating natural language questions into precise SQL queries.

    Based on the user's question and database schema information:
    1. Generate a syntactically correct SQLite query that addresses the question
    2. Focus on retrieving only the necessary columns for answering the question
    3. Use appropriate aggregations, joins, and filters
    4. Limit results to 5 rows unless explicitly asked for more
    5. Order results by relevant columns to show the most informative examples first

    Important Guidelines:
    - Output the raw SQL query directly without using a tool call
    - NEVER attempt DML operations (INSERT, UPDATE, DELETE, DROP)
    - If query execution fails, analyze the error and revise your approach
    - Empty result sets may indicate a need to broaden search criteria
    - Never fabricate information - if data is insufficient, acknowledge limitations
    - When you have the final answer, use the FinalResponse tool to submit it with an appropriate format preference

    Your goal is to provide accurate, concise answers based solely on available database information.
    """

    query_generator_prompt = ChatPromptTemplate.from_messages(
        [("system", query_generator_system), ("placeholder", "{messages}")]
    )

    query_generator = query_generator_prompt | ChatOpenAI(
        model="gpt-4o", 
        api_key=api_key, 
        temperature=0
    ).bind_tools([FinalResponse])

    def query_generation_node(state: AgentState):
        """Generate SQL query based on user question and schema information."""
        message = query_generator.invoke(state)

        # Validate tool calls
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "FinalResponse":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: Invalid tool call: {tc['name']}. Please use FinalResponse for final answers only. SQL queries should be output directly without tool calls.",
                            tool_call_id=tc["id"],
                        )
                    )
        
        return {"messages": [message] + tool_messages}

    workflow.add_node("generate_query", query_generation_node)

    # Query validation node
    query_validator_system = """You are an expert SQL reviewer specializing in query optimization and error detection.

    Carefully analyze the provided SQLite query for the following issues:
    1. Syntax errors and incorrect SQL structure
    2. Performance pitfalls (missing indexes, inefficient joins, etc.)
    3. Logical flaws in predicates or conditionals
    4. Null handling issues (NOT IN with NULL values)
    5. Data type inconsistencies or implicit conversions
    6. Proper quoting of identifiers and string values
    7. Appropriate use of functions and correct argument counts
    8. Join condition correctness and potential cartesian products
    9. Appropriate use of aggregation functions

    If you identify any issues, rewrite the query to fix them while maintaining the original intent.
    If the query appears correct and optimal, return it unchanged.

    After your review, you'll execute the query using the appropriate database tool.
    """

    query_validator_prompt = ChatPromptTemplate.from_messages(
        [("system", query_validator_system), ("placeholder", "{messages}")]
    )

    query_validator = query_validator_prompt | ChatOpenAI(
        model="gpt-4o", 
        api_key=api_key, 
        temperature=0
    ).bind_tools([database_query_tool], tool_choice="required")

    workflow.add_node("validate_query", lambda state: {
        "messages": [query_validator.invoke({"messages": [state["messages"][-1]]})]
    })

    # Add execution and formatting nodes
    workflow.add_node("execute_query", create_robust_tool_node([database_query_tool]))
    workflow.add_node("format_results", create_robust_tool_node([format_query_results]))

    # Conditional edge function
    def determine_next_step(state: AgentState) -> Literal[END, "validate_query", "generate_query", "format_results"]:
        """Determine the next step in the workflow based on current state."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If final answer tool was called, end workflow
        if getattr(last_message, "tool_calls", None):
            tool_name = last_message.tool_calls[0].get("name", "")
            if tool_name == "FinalResponse":
                return END
        
        # If error occurred, go back to query generation
        if hasattr(last_message, "content") and isinstance(last_message.content, str):
            if last_message.content.startswith("Error"):
                return "generate_query"
        
        # If we have query results, format them
        if hasattr(last_message, "content") and isinstance(last_message.content, str):
            if "|" in last_message.content and not last_message.content.startswith("Error"):
                return "format_results"
        
        # Default: proceed to validation
        return "validate_query"

    # Define workflow connections
    workflow.add_edge(START, "check_query_cache")
    workflow.add_conditional_edges(
        "check_query_cache",
        lambda state: "list_tables_tool" if "Cache hit" not in state["messages"][-1].content else END
    )
    workflow.add_edge("list_tables_tool", "determine_relevant_tables")
    workflow.add_edge("determine_relevant_tables", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "generate_query")
    workflow.add_conditional_edges("generate_query", determine_next_step)
    workflow.add_edge("validate_query", "execute_query")
    workflow.add_edge("execute_query", "generate_query")
    workflow.add_edge("format_results", "generate_query")

    # Compile workflow
    return workflow.compile()

def run_cli(agent):
    """Run the interactive CLI for the SQL agent."""
    print_header("SQL Query Agent CLI")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("Type 'help' for available commands.")
    
    history = []
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['exit', 'quit']:
                print_info("Exiting CLI. Goodbye!")
                break
                
            elif command.lower() == 'help':
                print_info("Available commands:")
                print("  help             - Show this help message")
                print("  clear            - Clear the screen")
                print("  history          - Show previous queries")
                print("  exit/quit        - Exit the CLI")
                print("  Any other input  - Query the database in natural language")
                
            elif command.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                
            elif command.lower() == 'history':
                if not history:
                    print_info("No previous queries.")
                else:
                    print_info("Previous queries:")
                    for i, query in enumerate(history, 1):
                        print(f"  {i}. {query}")
                        
            elif command:
                # Add to history
                history.append(command)
                
                print_info("Processing query...")
                result = agent.invoke({
                    "messages": [("user", command)],
                    "cache_info": {"hits": 0, "misses": 0}
                })
                
                # Extract the answer
                for message in reversed(result["messages"]):
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if tool_call.get("name") == "FinalResponse":
                                args = tool_call.get("args", {})
                                answer = args.get("answer", "No answer found.")
                                print("\n" + "=" * 80)
                                print(answer)
                                print("=" * 80)
                                break
                
        except KeyboardInterrupt:
            print("\nOperation cancelled. Type 'exit' to quit.")
        except Exception as e:
            print_error(f"An error occurred: {str(e)}")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SQL Query Agent CLI")
    parser.add_argument("--db", type=str, default="Chinook.db", help="Path to SQLite database file")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (optional, will prompt if not provided)")
    args = parser.parse_args()
    
    # Download sample DB if needed
    if not os.path.exists(args.db):
        if not download_sample_db(args.db):
            return 1
    
    # Get API key
    api_key = args.api_key or get_api_key()
    if not api_key:
        print_error("API key is required to use the SQL agent.")
        return 1
    
    # Connect to database
    db_connection = setup_db_connection(args.db)
    if not db_connection:
        return 1
    
    # Create agent
    try:
        print_info("Initializing SQL agent...")
        agent = create_agent(db_connection, api_key)
        print_success("Agent initialized successfully!")
        
        # Run CLI
        run_cli(agent)
        
    except Exception as e:
        print_error(f"Failed to initialize agent: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
