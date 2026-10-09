import os

from dotenv import load_dotenv

load_dotenv()

def main() -> None:
    print("Hello from langchain-course!")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    print("Environment loaded: ANTHROPIC_API_KEY found.")


if __name__ == "__main__":
    main()
