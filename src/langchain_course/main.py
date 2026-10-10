from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool # the decorator to create a tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic

from langchain_tavily import TavilySearch


llm = ChatAnthropic(model="claude-haiku-5-5")
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from LangChain Course!")
    result = agent.invoke({
        "messages":[HumanMessage(content="Search for 3 job openings for a remote react native software engineer in Mexico, summarize the results and include links.")],
    })
    print(result)

if __name__ == "__main__":
    main()