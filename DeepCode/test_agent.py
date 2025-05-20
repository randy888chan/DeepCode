import asyncio
from mcp_agent.core.fastagent import FastAgent

# ==================== Agent Prompts ====================

MATH_QUESTION_AGENT_PROMPT = """You are a simple math question agent.
Your task is to provide the question: "What is 1 + 1?". You will receive the user answer.

Rules:
1. Keep your response simple and direct
2. Do not include any explanations or additional text

Output format:
{
    "answer": "useranswer"
}
And then go to the answer verifier agent.
"""

ANSWER_VERIFIER_AGENT_PROMPT = """You are an answer verification agent.
Your task is to verify if user answer for the question "What is 1 + 1?" is correct.

Rules:
1. Check if the answer equals 2
2. If the answer is not 2, return to the question agent
3. If the answer is 2, confirm it is correct

Input format:
{
    "answer": "answer to verify"
}

Output format:
{
    "is_correct": true|false,
    "message": "verification message",
    "should_retry": true|false
}
Then, if incorrect,go back to the question agent and ask for the answer again.
"""

# ==================== Agent Definitions ====================

agents = FastAgent(name="MathTest")

@agents.agent(
    name="MathQuestionAgent",
    model="sonnet",
    instruction=MATH_QUESTION_AGENT_PROMPT,
)

@agents.agent(
    name="AnswerVerifierAgent",
    model="sonnet",
    instruction=ANSWER_VERIFIER_AGENT_PROMPT,
)

# ==================== Workflow Definition ====================

@agents.chain(
    name="MathTestWorkflow",
    sequence=[
        "MathQuestionAgent",
        "AnswerVerifierAgent",
    ],
    instruction="A workflow to test basic arithmetic and verify answers",
    cumulative=True  # Enable context preservation
)

async def main() -> None:
    print("\n=== Math Test Workflow ===")
    print("Testing 1 + 1 question...")
    
    # Start the agent workflow
    async with agents.run() as agent:
        result = await agent.prompt("MathTestWorkflow", "What is 1 + 1?")
        
        # Process the result
        if isinstance(result, dict):
            if result.get("should_retry", False):
                print("\nAnswer was incorrect. Retrying with MathQuestionAgent...")
                await agent.prompt("MathQuestionAgent", "Please try again: What is 1 + 1?")
            else:
                print(f"\nResult: {result.get('message', 'No message provided')}")
        else:
            print(f"\nUnexpected result format: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 