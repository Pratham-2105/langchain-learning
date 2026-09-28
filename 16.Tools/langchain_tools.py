# LangChain provides some built-in tools that can be used by agents.
from typing import Type

from langchain_community.tools import (
    DuckDuckGoSearchResults,
    DuckDuckGoSearchRun,
    ShellTool,
)

# Import the @tool decorator to create our own custom tools.
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field

# ============================================================
# 1. SEARCHING USING DUCKDUCKGO
# ============================================================

"""
# Create a DuckDuckGo search tool.
# DuckDuckGoSearchRun returns the search result as text.
search_tool = DuckDuckGoSearchRun()

# Run the search tool.
results = search_tool.invoke("India latest news September 28 2026")

# Print the search results.
print(results)

# Tools also contain useful information about themselves.
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)
"""


# ============================================================
# 2. USING SHELL IN LANGCHAIN
# ============================================================

"""
# Create a shell tool.
# This allows LangChain to execute shell commands.
shell_tool = ShellTool()

# Execute a shell command.
results = shell_tool.invoke("dir")

# Print the output of the command.
print(results)
"""


# ============================================================
# 3. CREATING A CUSTOM TOOL
# ============================================================

# @tool converts a normal Python function into a LangChain tool.
# @tool
# def multiply(a: int, b: int) -> int:
#     """Multiply two integers."""
#     return a * b


# # Call the custom tool using a dictionary of arguments.
# result = multiply.invoke({"a": 3, "b": 5})

# # Print the result.
# print(result)

# # Print information about our custom tool.
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# ============================================================
# 4. STRUCTURED TOOLS
# ============================================================

"""
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
"""

# ============================================================
# 5. USING BASETOOLS CLASS
# ============================================================

"""
class MutiplyInput(BaseModel):
    a: int = Field(description="The first number to add")
    b: int = Field(description="The second number to add")

class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MutiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()
result = multiply_tool.invoke({'a':3, 'b':4})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)
"""

# ============================================================
# 6. TOOLKITS 
# ============================================================

# Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers in INT"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers in INT"""
    return a * b 

class MathToolKit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolKit()
tools = toolkit.get_tools()

for tool in tools:

    print(tool.name, "=>", tool.description)