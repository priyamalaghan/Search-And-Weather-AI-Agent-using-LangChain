# %%
import os
import certifi
from dotenv import load_dotenv
from IPython.display import display, HTML
import requests
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults #Used to access prompts stored in LangChain Hub.
from langchain import hub # provides reusable prompts


# %%
#Agent related functionalities
#Used to create and run an AI agent that can decide when to use tools.
from langchain.agents import create_react_agent, AgentExecutor

# %%

from langchain_core.tools import tool
## @tool turns a normal Python function into a tool that an AI agent can understand and use.
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
#Now function add becomes a LangChain Tool.

# %%
#HTTPS = a secure connection between your Python program and a website/API.
#Certificate = a digital ID that helps verify that the website/API is genuine.
#certifi = provides a collection of trusted certificates.
#certifi.where() = finds where that certificate file is stored.

os.environ["SSL_CERT_FILE"] = certifi.where()

#Load ENV Variables
load_dotenv()

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

#Streamlit UI for displaying responses
st.set_page_config(
    page_title="LangChain Agent Demo",
    page_icon="🤖",
    layout="centered",
)

st.title("Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

# %%
#Initializing LLM
#llm = ChatOpenAI(model="gpt-6-luna", temperature=0, openai_api_key=OPEN_API_KEY)
llm = ChatOllama(model="llama3.2:latest", temperature=0)
#ollama lis

# %%
from datetime import datetime
from langchain_core.tools import tool

@tool
def get_current_date(query: str="") -> str:
    """ONLY use this tool when the user asks for today's date,
    current date, or today's day/date."""
    return datetime.now().strftime("%d-%b-%Y")

# %%
@tool
def get_weather_data(location: str) -> str:
    """ONLY use this tool when the user asks for current weather,
    temperature, humidity, or weather conditions for a specific city.
    The location should be a city name such as New Delhi, London, or Chicago."""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={location}"
    )
    response = requests.get(url)
    data = response.json()
    
    if "current" not in data:
        return f"Could not fetch weather data for {location}"
    return (
        f"City: {location}\n"
        f"Temperature: {data['current']['temperature']}deg C \n"
        f"Weather: {data['current']['weather_descriptions'][0]} \n"
        f"Humidity: {data['current']['humidity']}%"
    )
    
# llm_with_weather_tool = llm.bind_tools([get_weather_data])
# response = llm_with_weather_tool.invoke("What is the current weather in New Delhi?")
# print(response.tool_calls)

# # %%
# llm_with_date_tool = llm.bind_tools([get_current_date])
# response = llm_with_date_tool.invoke("What is today's date")
# if response.tool_calls:
#     date = get_current_date.invoke(response.tool_calls[0]["args"])
#     print(date)


# %%
#Initializing Tavily Search Result Object
search_tool = TavilySearchResults(max_results=2)
result = search_tool.invoke("Give me the latest news on AI")
result

# %%
# llm_with_tavily_tool = llm.bind_tools([search_tool])

# response = llm_with_tavily_tool.invoke("What is the top latest news on AI?")

# if response.tool_calls:
#     result = search_tool.invoke(response.tool_calls[0]["args"])

#     for item in result:
#         print(item)

#         if "url" in item:
#             display(
#                 HTML(
#                     f'<a href="{item["url"]}" target="_blank">{item["url"]}</a>'
#                 )
#             )

#search_tool = the tool
#llm_with_tavily_tool = LLM with that tool attached.

# %%
#print(response.tool_calls)

# %%
import yfinance as yf

@tool
def get_top_5_gainers(query: str="") -> str:
    """ONLY use this tool when the user asks for stock market gainers,
    top gaining stocks, biggest stock winners, or today's stock gainers.
    Do NOT use this tool for weather, dates, cities, countries, or general questions."""

    data = yf.screen("day_gainers")

    top_5 = data["quotes"][:5]

    return [
        {
            "symbol": stock["symbol"],
            "name": stock["shortName"],
            "price": stock["regularMarketPrice"],
            "change_percent": stock["regularMarketChangePercent"]
        }
        for stock in top_5
    ]


# llm_with_stock_tool = llm.bind_tools([get_top_5_gainers])
# response = llm_with_stock_tool.invoke("What are the top 5 gainers in the US stock market today?")
# if response.tool_calls:
    
#     result = get_top_5_gainers.invoke(
#         response.tool_calls[0]["args"]
#     )

#     for stock in result:
#         print(
#             stock["symbol"],
#             "-",
#             stock["name"],
#             "- $",
#             stock["price"],
#             "-",
#             round(stock["change_percent"], 2),
#             "%"
#         )



# %%
tools = [search_tool, get_current_date, get_top_5_gainers, get_weather_data]

# %%
#Prompt
#For create react agent functionality, below is recommended prompt
prompt1 = hub.pull("hwchase17/react")

# %%
from langchain_core.prompts import PromptTemplate

template = """Answer the question using the available tool when necessary.

TOOLS:
{tools}

AVAILABLE TOOL NAMES:
{tool_names}

STRICT RULES:

1. The Action field MUST contain ONLY a tool name.
2. The tool name MUST be exactly one of [{tool_names}].
3. Action Input MUST contain only the input for that tool.
4. Use the tool AT MOST ONCE.
5. After you receive an Observation, DO NOT call any tool again.
6. After the Observation, immediately produce:
   Thought: I now know the final answer
   Final Answer: <answer>
7. Never repeat the Question.
8. Never repeat the Observation.
9. Never repeat the search.
10. Final Answer must directly answer the original question.

EXACT FORMAT:

Question: {input}
Thought: I need to find the answer.
Action: <exact tool name>
Action Input: <tool input>
Observation: <tool result>
Thought: I now know the final answer
Final Answer: <answer> the final answer tothe original question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# %%
from langchain_core.prompts import PromptTemplate

template = """Answer the question using the available tools when necessary.

TOOLS:
{tools}

AVAILABLE TOOL NAMES:
{tool_names}

Follow this format:

Question: the user's question
Thought: think about what to do
Action: one of [{tool_names}]
Action Input: input for the selected tool
Observation: tool result
Thought: I now know the final answer
Final Answer: the answer to the user's question

RULES:
- Action must contain ONLY the exact tool name.
- Action Input must contain only the tool input.
- Choose only the tool that is relevant to the question.
- After receiving an Observation that answers the question, do NOT call another tool.
- Give the Final Answer immediately.
- Do not repeat the Question.
- Do not repeat the Observation.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt2 = PromptTemplate.from_template(template)

# %%
prompt3 = PromptTemplate.from_template("""
Answer the user's question using the available tools.

TOOLS:
{tools}

Tool names:
{tool_names}

Use this format:

Question: {input}
Thought: decide what to do
Action: tool name
Action Input: tool input
Observation: tool result

Repeat Thought → Action → Action Input → Observation
if another tool is needed.

Important:
- Complete every part of the user's question.
- Use the result of one tool as input to another tool when needed.
- Do not repeat a tool call if its result already answers that part.
- Give Final Answer only after all requested tasks are complete.

When finished:
Thought: I now know the final answer
Final Answer: your complete answer

Question: {input}
Thought: {agent_scratchpad}
""")


# %%
search_tool.name

# %%
response = llm_with_tavily_tool.invoke(
    "Search for the capital of India."
)

print(response.tool_calls)

# %%
#Create Agent
agent = create_react_agent(
    llm=llm ,
    tools=tools,
    prompt=prompt2
    )
#agent = agent object

# %%
#Executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
    )
#agent_executor = agent executor object

#UI Input
user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and get the current weather there."
)


# %%
#Run Agent with User Query

if st.button("Run Agent"):
    if user_query:
        with st.spinner("Running agent..."):
            try: 
                response = agent_executor.invoke({
                    "input":user_query                        
                    })
                st.success("Response Generated.")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query before running the agent.")


# %%
print([tool.name for tool in agent_executor.tools])