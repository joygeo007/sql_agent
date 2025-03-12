# SQL Query Agent with LangGraph

A conversational AI agent that translates natural language questions into SQL queries, executes them, and returns formatted results.

## Overview

This project implements an intelligent SQL query agent using LangGraph and LangChain. The agent takes natural language questions about data stored in a SQL database, automatically generates and validates SQL queries, executes them, and returns formatted results.

## Features

- **Natural Language to SQL Translation**: Converts user questions into precise SQL queries
- **Multi-step Processing Pipeline**: 
  - Database exploration
  - Schema analysis
  - Query generation
  - Validation
  - Execution
  - Error handling
- **Query Caching**: Improves performance for repeated queries
- **Multiple Result Formats**: Supports JSON, CSV, and Markdown output
- **Robust Error Handling**: Provides fallback mechanisms and helpful error messages
- **Conditional Workflow Routing**: Dynamically adjusts processing based on query context and execution results

<img src="https://github.com/user-attachments/assets/f864eeb7-cb8a-4b52-91cb-3242122a20cc" width="400" height="600" alt="SQL Query Agent Architecture">

## Technical Implementation

The agent is built with the following technologies:
- **LangGraph**: For creating the agent workflow and handling state management
- **LangChain**: For connecting to database utilities and building tools
- **OpenAI GPT Models**: For natural language understanding and SQL generation
- **SQLDatabase**: For database connection and query execution
- **Pandas**: For data manipulation and formatting
- **SQLite**: As the demonstration database backend

## Installation

```bash
# Install required packages
pip install langgraph langchain_community langchain_openai

# Download the sample Chinook database
# This is included in the repository code
```

## Usage

```python
# Initialize the agent
agent = workflow.compile()

# Query the database with natural language
result = agent.invoke({
    "messages": [("user", "Which sales agent made the most in sales in 2009?")],
    "cache_info": {"hits": 0, "misses": 0}
})

# Extract the answer
print(result["messages"][-1].tool_calls[0]["args"]["answer"])
```

## Architecture

The agent follows a directed workflow with conditional branches:
1. Check query cache
2. List available database tables
3. Determine relevant tables for the query
4. Retrieve schema information
5. Generate SQL query
6. Validate query
7. Execute query
8. Format results
9. Generate final response

## Evaluation

The agent has been tested on the WikiSQL dataset to evaluate its performance across a variety of natural language queries.

## Future Improvements

- Semantic cache matching for similar queries
- Support for more complex SQL operations
- Additional database backends
- Performance optimization for large datasets
- Integration with visualization tools
