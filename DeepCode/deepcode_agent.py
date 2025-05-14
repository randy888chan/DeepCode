import asyncio

from mcp_agent.core.fastagent import FastAgent
from Deepcode.agents.code_agents import *
# from rich import print

# agents = FastAgent(name="Researcher Agent")



async def main() -> None:
    research_prompt = """
Produce an investment report for the company Eutelsat. The final report should be saved in the filesystem in markdown format, and
contain at least the following: 
1 - A brief description of the company
2 - Current financial position (find data, create and incorporate charts)
3 - A PESTLE analysis
4 - An investment thesis for the next 3 years. Include both 'buy side' and 'sell side' arguments, and a final 
summary and recommendation.
Todays date is 15 February 2025. Include the main data sources consulted in presenting the report."""  # noqa: F841

    async with agents.run() as agent:
        await agent.CodeAssistAgent.prompt()


if __name__ == "__main__":
    asyncio.run(main())  # type: ignore
