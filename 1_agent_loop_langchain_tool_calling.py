# Sample code demonstrating how to use LangChain's agent loop with tool calling with manual boilerplate.
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"  # Replace with your desired model name

# --- Tools (LangChain @tool decorator) ---

@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product."""
    # Simulate fetching product price (replace with actual logic)
    prices = {
        "laptop": 999.0,
        "smartphone": 699.0,
        "headphones": 199.0,
        "keyboard": 49.0,
    }
    print(f"Fetching price for product: {product_name}")
    return prices.get(product_name.lower(), 0.0)  # Return 0.0 if product not found

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to the price and return the discounted price.
    Available discount tiers: 'bronze', 'silver', 'gold'."""

    discount_percentages = {
        "bronze": 5,
        "silver": 12,
        "gold": 23
    }

    discount = discount_percentages.get(discount_tier.lower(), 0)

    print(f"Applying discount: {discount}% to price: {price}")
    return round(price * (1 - discount / 100), 2)  # Return discounted price rounded to 2 decimal places

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop with Tool Calling") # Manual traceable decorator for the agent loop
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm = init_chat_model(f"anthropic:{MODEL}", temperature=0) # This can be another option if you have access to Anthropic models

    llm_with_tools = llm.bind_tools(tools)

    print(f"Running agent with question: {question}")
    print("=" * 60)

    messages = [SystemMessage(
        content=(
            "You are a helpful shopping assistant that can answer questions about product prices and apply discounts. "
            "You have access to the following tools: get_product_price and apply_discount. "
            "STRICT RULES - you must follow these exact rules when responding:\n"
            "1. NEVER make up product prices. Use the get_product_price tool to fetch the price of a product.\n"
            "2. NEVER apply discounts without first fetching the product price.\n"
            "3. Do NOT pass a made-up number to the apply_discount tool. Always use the price fetched from get_product_price.\n"
            "4. Never calculate discoumts yourself using math. Always use the apply_discount tool to calculate the discounted price.\n"
            "5. If the user does not specify a discount tier, ask them which tier they would like to use - do NOT assume one.\n"
            )
        ),
        HumanMessage(content=question)
        ]

    for iteration in range(MAX_ITERATIONS+1):
        print(f"--- Iteration {iteration + 1} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call for simplicity - force one tool call per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")

        # TODO(human): wrap this call in try/except and set `observation` to an error string on failure
        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool Result] {observation}")

        # Append the AI message and the tool observation to the messages for the next iteration
        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call_id
        ))

    print("ERROR: Max iterations reached without a final answer.")
    return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")