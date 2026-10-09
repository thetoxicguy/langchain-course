import os
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_API_KEY"]


def check_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)}")


def main() -> None:
    print("Hello from langchain-course!")
    check_env()

    # A Chat Model object: configured once, then called with .invoke().
    # With LANGSMITH_TRACING=true, every call is sent to LangSmith automatically.
    llm = ChatAnthropic(model="claude-haiku-5-5", max_tokens=256)

    # A prompt template with a {topic} variable, filled in at invoke time.
    prompt = ChatPromptTemplate.from_messages([("user", "Write a haiku about {topic}.")])
    chain = prompt | llm

    # Topic comes from the command line, or we ask for it.
    topic = " ".join(sys.argv[1:]) or input("What should the haiku be about? ")
    response = chain.invoke({"topic": topic})

    print(response.content)
    print("\nToken usage:", response.usage_metadata)


if __name__ == "__main__":
    main()
